import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys

sys.path.append("C:/Users/aslan/Documents/MER/emotionbind")

# sys.path.append("C:/Users/aslan/Documents/MER/emotionbind/emotionbind")

from emotionbind.datasets.dataset_iemocap import IEMOCAPDatasetHandler
from emotionbind.datasets.dataset_amigos import AMIGOSDatasetHandler
from emotionbind.datasets.dataset_mahnob_hci import MAHNOBHCIDatasetHandler
#from emotionbind.datasets.dataset_kemocon import KEMOCONDatasetHandler
# from emotionbind.datasets.dataset_imigue import IMIGUEDatasetHandler
from emotionbind.datasets.dataset_smg import SMGDatasetHandler
from emotionbind.models.emotionbind_model import EmotionBindModel
from torch.utils.data import ConcatDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "./checkpoints"


writer = SummaryWriter(log_dir="/home/razzhivina/emotionbind/emotionbind/train/logs")

DATASET_MODALITIES = {
    "iemocap": ["video", "audio", "text"],
    "amigos": ["video", "faces", "eeg", "ecg"],
    "kemocon": ["video", "audio", "text", "eeg", "ecg"],
    "mahnob": ["video", "eeg"],
    "smg" : ["video"]
}

iemocap_root = "/space/emotion_data/IEMOCAP_full_release"
feature_dirs_iemocap = {}
feature_dirs_iemocap["video"] = "/space/emotion_data/IEMOCAP_full_release/mvit_v2_scene_features"
feature_dirs_iemocap["audio"] = "args.audio_feature_dir"
feature_dirs_iemocap["text"] = "/space/emotion_data/IEMOCAP/text_features"
amigos_root = "/space/emotion_data/microgestures/AMIGOS"
feature_dirs_amigos = {}
feature_dirs_amigos["video"] = "/space/emotion_data/microgestures/AMIGOS/mvit2_features/rgb"
feature_dirs_amigos["faces"] = "/space/emotion_data/microgestures/AMIGOS/mvit2_features/face"
feature_dirs_amigos["eeg"] = "/space/emotion_data/microgestures/AMIGOS/unzipped/EEG/features"
feature_dirs_amigos["ecg"] = "/space/emotion_data/microgestures/AMIGOS/unzipped/ECG/features"

mahnob_root = "/space/emotion_data/microgestures/HCI_tagging"

def info_nce_loss(embeddings, label, temperature=0.07):
    """
    Computes InfoNCE loss between multiple modality embeddings and a shared label.
    Args:
        embeddings: Dictionary of modality embeddings from the model.
        label: 768D projected VAD label (shared target).
        temperature: Scalar to scale the similarity (softmax temperature).
    """
    all_embeddings = torch.stack(list(embeddings.values()))  # Stack all modality embeddings
    label = label.unsqueeze(0)  # Ensure label is a 2D tensor (batch_size=1)

    # Compute cosine similarities (inner product) between label and each modality
    similarities = F.cosine_similarity(all_embeddings, label, dim=-1) / temperature

    # Apply softmax to get similarity distribution
    logits = similarities  # Since we're comparing to the same label, no need for negtives here
    loss = -torch.log(torch.exp(logits) / torch.sum(torch.exp(logits), dim=0))  # Normalized softmax

    return loss.mean()

def info_nce_loss_pairs(embeddings, labels, temperature=0.07):
    """
    Computes InfoNCE loss using positive pairs (modality embedding x corresponding label)
    and negatives selected as the furthest label from the current label in the batch.

    Args:
        embeddings: Dictionary of modality embeddings from the model.
        labels: Tensor of shape (batch_size, embedding_dim), containing projected VAD labels.
        temperature: Scalar to scale the similarity (softmax temperature).
    """
    batch_size = labels.shape[0]  # Batch size
    all_embeddings = torch.stack(list(embeddings.values()))  # Shape: (num_modalities, batch_size, embedding_dim)
    num_modalities = all_embeddings.shape[0]

    # Expand labels to match modality embeddings shape
    labels = labels.unsqueeze(0).expand(num_modalities, -1, -1)  # Shape: (num_modalities, batch_size, embedding_dim)

    # Compute cosine similarity between each modality embedding and its corresponding label
    pos_similarities = F.cosine_similarity(all_embeddings, labels,
                                           dim=-1) / temperature  # Shape: (num_modalities, batch_size)

    # Compute pairwise cosine distances between all labels
    label_sim_matrix = F.cosine_similarity(labels.unsqueeze(2), labels.unsqueeze(1),
                                           dim=-1)  # Shape: (num_modalities, batch_size, batch_size)

    # Find the furthest (most dissimilar) label in the batch for each label
    furthest_label_indices = torch.argmin(label_sim_matrix, dim=-1)  # Shape: (num_modalities, batch_size)

    # Gather the furthest labels to use as negatives
    negative_labels = labels.gather(1, furthest_label_indices.unsqueeze(-1).expand(-1, -1, labels.shape[-1]))

    # Compute cosine similarity between each modality embedding and the furthest negative label
    neg_similarities = F.cosine_similarity(all_embeddings, negative_labels,
                                           dim=-1) / temperature  # Shape: (num_modalities, batch_size)

    # Compute InfoNCE loss
    logits = torch.cat([pos_similarities.unsqueeze(-1), neg_similarities.unsqueeze(-1)],
                       dim=-1)  # Shape: (num_modalities, batch_size, 2)
    labels = torch.zeros(num_modalities, batch_size, dtype=torch.long, device=logits.device)  # Positive at index 0
    loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))

    return loss

def contrastive_loss(embeddings, projected_vad_labels, alpha, gamma=0.5, lambda_reg=0.1):
    """
    Computes contrastive loss using projected VAD labels.

    Args:
    - embeddings: Dict of modality embeddings from model.
    - projected_vad_labels: Already projected VAD values (from dataset).
    - alpha: Weight vector for modalities.
    - gamma: Margin for contrastive loss.
    - lambda_reg: Regularization weight.

    Returns:
    - loss: Computed contrastive loss.
    """
    loss = 0.0
    num_modalities = len(embeddings)

    for i, (modality_i, fi) in enumerate(embeddings.items()):
        modality_loss = 0.0

        for j, (modality_j, fj) in enumerate(embeddings.items()):
            if i != j:
                dist = torch.norm(fi - fj, p=2, dim=1) ** 2
                margin_loss = torch.clamp(dist - gamma, min=0).mean()
                modality_loss += margin_loss

        alignment_loss = lambda_reg * torch.norm(fi - projected_vad_labels, p=2, dim=1).mean()
        loss += alpha[i] * (modality_loss + alignment_loss)

    return loss

def log_gradients(model):
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = torch.norm(param.grad).item()
    print("Gradient norms:", grad_norms)

def log_weights(model):
    weight_norms = {}
    for name, param in model.named_parameters():
        weight_norms[name] = torch.norm(param).item()
    print("Weight norms:", weight_norms)


def get_dataset_handler(dataset_name, root_dir, feature_dirs, split):
    dataset_handlers = {
        "iemocap": IEMOCAPDatasetHandler,
        "amigos": AMIGOSDatasetHandler,
        "mahnob": MAHNOBHCIDatasetHandler,
        #"kemocon": KEMOCONDatasetHandler,
        "smg" : SMGDatasetHandler
    }
    dataset_handler_cls = dataset_handlers.get(dataset_name.lower())
    if not dataset_handler_cls:
        raise ValueError(f"Dataset handler for {dataset_name} not found!")

    dataset_args = {"root_dir": root_dir, "split": split}

    modalities = DATASET_MODALITIES.get(dataset_name, [])

    for modality in modalities:
        if modality in feature_dirs:
            dataset_args[f"{modality}_dir"] = feature_dirs[modality]

    # print(dataset_args)

    return dataset_handler_cls(**dataset_args)


def train_process(process_id, dataset_name, dataset_root, feature_dirs, num_processes, batch_size, learning_rate,
                  epochs, multi_mode):
    print(f"Starting process {process_id}...")
    split = "train"

    if multi_mode:
        dataset_iemocap = get_dataset_handler("iemocap", iemocap_root, feature_dirs_iemocap, split)
        dataset_amigos = get_dataset_handler("amigos", amigos_root, feature_dirs_amigos, split)
        dataset_mahnob = get_dataset_handler("mahnob", mahnob_root, feature_dirs, split)
        dataset = ConcatDataset([dataset_iemocap, dataset_amigos, dataset_mahnob])
    else:
        dataset = get_dataset_handler(dataset_name, dataset_root, feature_dirs, split)

    sampler = RandomSampler(dataset, num_samples=len(dataset) // num_processes, replacement=False)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    model = EmotionBindModel(dataset_name=dataset_name).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_checkpoint_proc{process_id}.pth")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        try:
            for batch_idx, batch in enumerate(
                    tqdm(train_loader, desc=f"Process {process_id} | Epoch {epoch + 1}/{epochs}")):
                pass
        except Exception as e:
            # print(f"Error occurred at batch index {batch_idx}")
            print(f"Error: {e}")
            print(f"Batch content: {batch}")

        for batch_idx, batch in enumerate(
                tqdm(train_loader, desc=f"Process {process_id} | Epoch {epoch + 1}/{epochs}")):
            optimizer.zero_grad()
            inputs = {}
            if "video" in batch:
                if dataset_name == "mahnob":
                    inputs["vision"] = batch["video"].to(DEVICE)
                    inputs["vision1"] = batch["vbw1"].to(DEVICE)
                    inputs["vision2"] = batch["vbw2"].to(DEVICE)
                    inputs["vision3"] = batch["vbw3"].to(DEVICE)
                    inputs["vision4"] = batch["vbw4"].to(DEVICE)
                    inputs["vision5"] = batch["vbw5"].to(DEVICE)
                else:
                    inputs["vision"] = batch["video"].to(DEVICE)
            if "audio" in batch:
                inputs["audio"] = batch["audio"].to(DEVICE)
            if "text" in batch:
                inputs["text"] = batch["text"].to(DEVICE)
            if "faces" in batch:
                inputs["faces"] = batch["faces"].to(DEVICE)
            if "eeg" in batch:
                inputs["eeg"] = batch["eeg"].to(DEVICE)
            if "ecg" in batch:
                inputs["ecg"] = batch["ecg"].to(DEVICE)
            vad_labels = batch["label"].to(DEVICE)

            embeddings, _ = model(inputs)
            # alpha = torch.ones(len(embeddings)).to(DEVICE)
            loss = info_nce_loss_pairs(embeddings, vad_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            print(f"Batch Loss: {loss.item():.6f}")

            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Loss/train", loss.item(), step)

        avg_loss = total_loss / len(train_loader)
        print(f"Process {process_id} | Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        scheduler.step(avg_loss)

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "loss": best_val_loss},
                       best_checkpoint_path)
            print(f"Process {process_id} | New best model saved with Validation Loss: {best_val_loss:.6f}")
    writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dataset_dir", type=str, default="D:/IEMOCAP", help="Path to training dataset directory")
    parser.add_argument("--text_feature_dir", type=str, default="D:/IEMOCAP/text_features",
                        help="Path to text feature directory")
    parser.add_argument("--video_feature_dir", type=str, default="D:/IEMOCAP/mvit_v2_scene_features",
                        help="Path to video feature directory")
    parser.add_argument("--audio_feature_dir", type=str, default="D:/IEMOCAP/wav2vec2_audio_features",
                        help="Path to audio feature directory")
    parser.add_argument("--faces_feature_dir", type=str, default="D:/AMIGOS/face_features",
                        help="Path to faces video feature directory")
    parser.add_argument("--eeg_feature_dir", type=str, default="D:/AMIGOS/eeg_features",
                        help="Path to eeg video feature directory")
    parser.add_argument("--ecg_feature_dir", type=str, default="D:/AMIGOS/ecg_features",
                        help="Path to ecg video feature directory")
    parser.add_argument("--dataset_name", type=str, default="iemocap", choices=['iemocap', 'amigos', 'kemocon', 'mahnob', 'smg', 'imigue'], help="Name of the dataset")
    parser.add_argument("--num_processes", type=int, default=1, help="Number of parallel training processes")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--multi_mode", type=bool, default=False, help="Whether to enable multi-dataset training")
    args = parser.parse_args()

    feature_dirs = {}
    if args.video_feature_dir:
        feature_dirs["video"] = args.video_feature_dir
    if args.audio_feature_dir:
        feature_dirs["audio"] = args.audio_feature_dir
    if args.text_feature_dir:
        feature_dirs["text"] = args.text_feature_dir
    if args.faces_feature_dir:
        feature_dirs["faces"] = args.faces_feature_dir
    if args.eeg_feature_dir:
        feature_dirs["eeg"] = args.eeg_feature_dir
    if args.ecg_feature_dir:
        feature_dirs["ecg"] = args.ecg_feature_dir



    if not os.path.exists(args.root_dataset_dir):
        raise FileNotFoundError(f"The dataset directory does not exist: {args.root_dataset_dir}")

    mp.set_start_method("spawn", force=True)
    processes = []
    for process_id in range(args.num_processes):
        p = mp.Process(target=train_process, args=(process_id,
                                                   args.dataset_name,
                                                   args.root_dataset_dir,
                                                   feature_dirs,
                                                   args.num_processes,
                                                   args.batch_size,
                                                   args.learning_rate,
                                                   args.epochs,
                                                   args.multi_mode))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    #print(os.getcwd())
    main()

