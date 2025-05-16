import torch.nn as nn
from transformers import RobertaModel, AutoModel
from ..utils import misc_utils

misc_utils.set_seed(seed_value=42)


class RobertaFeatureExtractor(nn.Module): 
    def __init__(self, freeze_weights=True, **kwargs):
        super().__init__()
        self.vectorizer = RobertaModel.from_pretrained("roberta-base", add_pooling_layer=False)
        if freeze_weights:
            for param in self.vectorizer.parameters():
                param.requires_grad = False

    def forward(self, encoded):
        """Vectorizes text using RoBERTa model.

        Args:
            input (str or list of str): Text or texts to vectorize.

        Returns:
            torch.Tensor: Tensor containing the vectorized text.
        """
        return self.vectorizer(**encoded).last_hidden_state

def roberta(**kwargs):
    return RobertaFeatureExtractor(**kwargs)


class ModernBERTFeatureExtractor(nn.Module):
    def __init__(self, freeze_weights=True, **kwargs):
        super().__init__()
        self.vectorizer = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
        # Set the model to eval mode and move it to CPU by default if memory is a concern
        if freeze_weights:
            for param in self.vectorizer.parameters():
                param.requires_grad = False

    def forward(self, encoded):
        """Vectorizes text using ModernBERT model.

        Args:
            encoded: Dictionary containing tokenized text batches for ModernBERT.
            
        Returns:
            torch.Tensor: The last hidden states of ModernBERT.
        """
        return self.vectorizer(**encoded).last_hidden_state


def modernbert(**kwargs):
    return ModernBERTFeatureExtractor(**kwargs)


text_feature_extractors = {
    "roberta" : roberta,
    "modernbert": modernbert,
}