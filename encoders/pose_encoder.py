import torch
from encoder import EncoderBase
class PoseEncoder(EncoderBase):
    def __init__(self, model_path, output_dim):
        super().__init__(model_path, output_dim)
        checkpoint = torch.load(model_path)
        self.model = checkpoint['model']  # Assuming it's stored in the 'model' key

    def get_output_dim(self):
        return self.model.output_dim  # Adjust based on how dimensions are stored

    def encode(self, x):
        return self.model(x)