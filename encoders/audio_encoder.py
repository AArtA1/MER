import torch
from encoder import EncoderBase
from transformers import AutoModel

class AudioEncoder(EncoderBase):
    def __init__(self, model_path, output_dim):
        super().__init__(model_path, output_dim)
        self.model = AutoModel.from_pretrained(model_path)

    def get_output_dim(self):
        return self.model.config.hidden_size

    def encode(self, x):
        return self.model(x).last_hidden_state.mean(dim=1)
