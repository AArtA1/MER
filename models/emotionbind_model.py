import os
from functools import partial
from types import SimpleNamespace
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from emotionbind.imagebind.models.helpers import (EinOpsRearrange, LearnableLogitScaling, Normalize,
                                                  SelectElement, SelectEOSAndProject)
from emotionbind.imagebind.models.multimodal_preprocessors import (AudioPreprocessor,
                                                                   PadIm2Video,
                                                                   PatchEmbedGeneric,
                                                                   RGBDTPreprocessor,
                                                                   SpatioTemporalPosEmbeddingHelper,
                                                                   TextPreprocessor)

from emotionbind.utils.vad_utils import find_closest_key, find_closest_label

from emotionbind.imagebind.models.transformer import MultiheadAttention, SimpleTransformer

ModalityType = SimpleNamespace(
    VISION="vision",
    VISION1="vision1",
    VISION2="vision2",
    VISION3="vision3",
    VISION4="vision4",
    VISION5="vision5",
    TEXT="text",
    AUDIO="audio",
    EEG="eeg",
    ECG="ecg",
    POSE="pose",                   # Skeleton Pose Data
    FACES="faces"
)

# Datasets: MANHOB-HCI, AMIGOS, K-EMOCON, IMIGUE, SMG, IEMOCAP

DATASET_WEIGHTS = {
    "mahnob": torch.tensor([0.3, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1]),
    "amigos": torch.tensor([0.3, 0.3, 0.2, 0.2]),
    "iemocap": torch.tensor([0.4, 0.3, 0.3]),
}


class EmotionBindModel(nn.Module):
    def __init__(
        self,
        dataset_name,
        video_frames=4,
        video_kernel_size=(2, 14, 14),
        audio_kernel_size=16,
        audio_stride=10,
        out_embed_dim=768,
        eeg_embed_dim=768,
        ecg_embed_dim=768,
        pose_embed_dim=768,
        vision_embed_dim=768,
        text_embed_dim=768,
        audio_embed_dim=768,
        faces_embed_dim=768,
    ):
        super().__init__()

        #self.log_logit_scale = nn.Parameter(torch.tensor(0.0))

        self.dataset_name = dataset_name

        # modality preprocessors

        self.modality_preprocessors = self._create_modality_preprocessors(
            video_frames,
            vision_embed_dim,
            video_kernel_size,
            text_embed_dim,
            audio_embed_dim,
            audio_kernel_size,
            audio_stride,
        )

        # modality trunks
        self.modality_trunks = self._create_modality_trunks(
            #vision_embed_dim,
            #text_embed_dim,
            #audio_embed_dim,
            #eeg_embed_dim,
            #ecg_embed_dim,
            #pose_embed_dim,
        )

        # modality-specific heads for embeddings
        self.modality_heads = self._create_modality_heads(
            out_embed_dim,
            vision_embed_dim,
            text_embed_dim,
            audio_embed_dim,
            eeg_embed_dim,
            ecg_embed_dim,
            #pose_embed_dim,
            faces_embed_dim,
        )

        # postprocessors for embeddings
        self.modality_postprocessors = self._create_modality_postprocessors(
            out_embed_dim
        )

    def _create_modality_preprocessors(
            self,
            video_frames=2,
            vision_embed_dim=1024,
            video_kernel_size=(2, 14, 14),
            text_embed_dim=768,
            audio_embed_dim=768,
            audio_kernel_size=16,
            audio_stride=10,
            audio_num_mel_bins=128,
            audio_target_len=204,
    ):
        # Video Preprocessor
        video_stem = PatchEmbedGeneric(
            proj_stem=[
                PadIm2Video(pad_type="repeat", ntimes=2),
                nn.Conv3d(
                    in_channels=3,
                    kernel_size=video_kernel_size,
                    out_channels=vision_embed_dim,
                    stride=video_kernel_size,
                    bias=False,
                ),
            ]
        )
        video_preprocessor = RGBDTPreprocessor(
            img_size=[3, video_frames, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=video_stem,
            depth_stem=None,
        )

        # Text Preprocessor
        text_preprocessor = TextPreprocessor(
            context_length=512,
            vocab_size=50265,
            embed_dim=text_embed_dim,
            causal_masking=False,
        )

        # Audio Preprocessor
        audio_stem = PatchEmbedGeneric(
            proj_stem=[
                nn.Conv2d(
                    in_channels=1,
                    kernel_size=audio_kernel_size,
                    stride=audio_stride,
                    out_channels=audio_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=audio_embed_dim),
        )
        audio_preprocessor = AudioPreprocessor(
            img_size=[1, audio_num_mel_bins, audio_target_len],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            audio_stem=audio_stem,
        )


        modality_preprocessors = {
            ModalityType.VISION: video_preprocessor,
            ModalityType.TEXT: text_preprocessor,
            ModalityType.AUDIO: audio_preprocessor,
            #ModalityType.ECG: ecg_preprocessor,
            #ModalityType.POSE: pose_preprocessor,
        }

        return nn.ModuleDict(modality_preprocessors)

    def _create_modality_trunks(
        self,
        vision_embed_dim=768,
        vision_num_blocks=24,
        vision_num_heads=16,
        text_embed_dim=768,
        text_num_blocks=12,
        text_num_heads=12,
        audio_embed_dim=768,
        audio_num_blocks=12,
        audio_num_heads=12,
        audio_drop_path=0.0,
        eeg_embed_dim=768,
        eeg_num_blocks=12,
        eeg_num_heads=12,
        ecg_embed_dim=768,
        ecg_num_blocks=12,
        ecg_num_heads=12,
        faces_embed_dim=768,
        faces_num_blocks=24,
        faces_num_heads=16,
    ):
        def instantiate_trunk(
            embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path
        ):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6)
                    if pre_transformer_ln
                    else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            )

        modality_trunks = {}
        modality_trunks[ModalityType.VISION] = instantiate_trunk(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            pre_transformer_ln=True,
            add_bias_kv=False,
            drop_path=0.0,
        )

        modality_trunks[ModalityType.VISION1] = instantiate_trunk(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            pre_transformer_ln=True,
            add_bias_kv=False,
            drop_path=0.0,
        )

        modality_trunks[ModalityType.VISION2] = instantiate_trunk(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            pre_transformer_ln=True,
            add_bias_kv=False,
            drop_path=0.0,
        )

        modality_trunks[ModalityType.VISION3] = instantiate_trunk(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            pre_transformer_ln=True,
            add_bias_kv=False,
            drop_path=0.0,
        )

        modality_trunks[ModalityType.VISION4] = instantiate_trunk(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            pre_transformer_ln=True,
            add_bias_kv=False,
            drop_path=0.0,
        )

        modality_trunks[ModalityType.VISION5] = instantiate_trunk(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            pre_transformer_ln=True,
            add_bias_kv=False,
            drop_path=0.0,
        )

        modality_trunks[ModalityType.TEXT] = instantiate_trunk(
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=False,
            drop_path=0.0,
        )
        modality_trunks[ModalityType.AUDIO] = instantiate_trunk(
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=audio_drop_path,
        )

        modality_trunks[ModalityType.EEG] = instantiate_trunk(
            eeg_embed_dim,
            eeg_num_blocks,
            eeg_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=0.0,
        )

        modality_trunks[ModalityType.ECG] = instantiate_trunk(
            ecg_embed_dim,
            ecg_num_blocks,
            ecg_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=0.0,
        )

        modality_trunks[ModalityType.FACES] = instantiate_trunk(
            faces_embed_dim,
            faces_num_blocks,
            faces_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=0.0,
        )


        return nn.ModuleDict(modality_trunks)

    def _create_modality_heads(
        self,
        out_embed_dim,
        vision_embed_dim,
        text_embed_dim,
        audio_embed_dim,
        eeg_embed_dim,
        ecg_embed_dim,
        faces_embed_dim,
    ):
        modality_heads = {}

        modality_heads[ModalityType.VISION] = nn.Sequential(
            nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(vision_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.VISION1] = nn.Sequential(
            nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(vision_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.VISION2] = nn.Sequential(
            nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(vision_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.VISION3] = nn.Sequential(
            nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(vision_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.VISION4] = nn.Sequential(
            nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(vision_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.VISION5] = nn.Sequential(
            nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(vision_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.TEXT] = SelectEOSAndProject(
            proj=nn.Sequential(
                nn.LayerNorm(normalized_shape=text_embed_dim, eps=1e-6),
                nn.Linear(text_embed_dim, out_embed_dim, bias=False),
            )
        )

        modality_heads[ModalityType.AUDIO] = nn.Sequential(
            nn.LayerNorm(normalized_shape=audio_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(audio_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.EEG] = nn.Sequential(
            nn.LayerNorm(normalized_shape=eeg_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(eeg_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.ECG] = nn.Sequential(
            nn.LayerNorm(normalized_shape=ecg_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(ecg_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.FACES] = nn.Sequential(
            nn.LayerNorm(normalized_shape=faces_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(faces_embed_dim, out_embed_dim, bias=False),
        )



        return nn.ModuleDict(modality_heads)

    def _create_modality_postprocessors(self, out_embed_dim):
        modality_postprocessors = {}

        modality_postprocessors[ModalityType.VISION] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.VISION1] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.VISION2] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.VISION3] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.VISION4] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.VISION5] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.FACES] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.TEXT] = nn.Sequential(
            Normalize(dim=-1), LearnableLogitScaling(learnable=True)
        )
        modality_postprocessors[ModalityType.AUDIO] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=20.0, learnable=False),
        )

        modality_postprocessors[ModalityType.EEG] = nn.Sequential(
            Normalize(dim=-1), LearnableLogitScaling(learnable=True)
        )

        modality_postprocessors[ModalityType.ECG] = nn.Sequential(
            Normalize(dim=-1), LearnableLogitScaling(learnable=True)
        )

        return nn.ModuleDict(modality_postprocessors)

    def clamp_log_logit_scale(model):
        with torch.no_grad():
            model.log_logit_scale.clamp_(-5, 5)

    def forward(self, inputs, vad_true=None):
        outputs = {}
        vad_losses = {}
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.log_logit_scale.data.clamp_(-5, 5)

        for modality_key, modality_value in inputs.items():
            if modality_value is not None:

                reduce_list = modality_value.ndim >= 5
                if reduce_list:
                    B, S = modality_value.shape[:2]
                    modality_value = modality_value.reshape(B * S, *modality_value.shape[2:])

                modality_value = self.modality_trunks[modality_key](modality_value)

                if modality_key == "text":
                    seq_len = modality_value.shape[1] - 1
                    seq_len = torch.clamp(torch.tensor(seq_len, dtype=torch.long, device=modality_value.device), min=0)
                    modality_value = self.modality_heads[modality_key](modality_value, seq_len=seq_len)
                else:
                    modality_value = self.modality_heads[modality_key](modality_value)


                modality_value = self.modality_postprocessors[modality_key](modality_value)

                if reduce_list:
                    modality_value = modality_value.reshape(B, S, -1).mean(dim=1)

                outputs[modality_key] = modality_value

                if self.training and vad_true is not None:
                    vad_losses[modality_key] = F.mse_loss(modality_value, vad_true[modality_key])

        if self.training:
            return outputs, vad_losses
        else:
            vad_projector = ExactInvertibleVADProjection().to(DEVICE)
            vad_predictions = {key: vad_projector.inverse(emb) for key, emb in outputs.items()}

            dataset_weights = DATASET_WEIGHTS[self.dataset_name]

            weights = torch.tensor(dataset_weights, device=DEVICE)
            vad_predictions = torch.stack(list(vad_predictions.values()), dim=0)
            vad_predictions = (vad_predictions * weights[:, None, None]).sum(dim=0)

            return outputs, vad_predictions

class AttentionPooling(nn.Module):
    def __init__(self, feature_dim, output_len):
        super().__init__()
        self.output_len = output_len
        self.attn = nn.Linear(feature_dim, 1)

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (seq_len, feature_dim)

        Returns:
            Tensor: Shape (output_len, feature_dim)
        """
        seq_len, feat_dim = x.shape

        attn_scores = self.attn(x)
        attn_weights = torch.softmax(attn_scores, dim=0)
        weighted_sum = torch.sum(x * attn_weights, dim=0)

        return weighted_sum.unsqueeze(0).repeat(self.output_len, 1)

class ExactInvertibleVADProjection(nn.Module):
    def __init__(self, input_dim=3, output_dim=768):
        super().__init__()

        path = f"../datasets/labels/vad_projection_{input_dim}.pt"

        self.W = nn.Parameter(torch.load(path))
        self.W.requires_grad = False  # Ensure it's frozen

    def forward(self, vad):
        """ Forward: Vector (XD) -> Embedding (768D) """
        return torch.matmul(vad, self.W.T)

    def inverse(self, embedding):
        """ Inverse: Embedding (768D) -> Vector (XD) """
        vad_reconstructed = torch.matmul(embedding, self.W)
        return torch.clamp(vad_reconstructed, min=-1, max=1)




