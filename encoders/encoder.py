import torch
import torch.nn as nn
class EncoderBase(nn.Module):
    def __init__(self, model_path, output_dim):
        super().__init__()
        self.model_path = model_path
        self.output_dim = output_dim
        self.model = None
        self.projection = nn.Linear(self.get_output_dim(), output_dim)

    def get_output_dim(self):
        raise NotImplementedError

    def forward(self, x):
        embeddings = self.encode(x)
        return self.projection(embeddings)

    def encode(self, x):
        raise NotImplementedError