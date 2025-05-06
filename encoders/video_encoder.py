import torch
from encoder import EncoderBase
class VideoEncoder(EncoderBase):
    def __init__(self, model_path, output_dim):
        super().__init__(model_path, output_dim)
        self.model = torch.hub.load("pytorch/vision", "mvit_v2_s", pretrained=True)

    def get_output_dim(self):
        return self.model.head.in_features

    def encode(self, x):
        return self.model(x)