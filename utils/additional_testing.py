import torch
import torch.nn as nn

class ExactInvertibleVADProjectionOLD(nn.Module):
    def __init__(self, input_dim=3, output_dim=768, path="projection_matrix.pt"):
        super().__init__()

        # Load pre-saved matrix from file
        self.W = nn.Parameter(torch.load(path))
        self.W.requires_grad = False  # Ensure it's frozen

    def forward(self, vad):
        """ Forward: VAD (560D) -> Embedding (768D) """
        return torch.matmul(vad, self.W.T)

    def inverse(self, embedding):
        """ Inverse: Embedding (768D) -> VAD (560D) """
        vad_reconstructed = torch.matmul(embedding, self.W)
        return torch.clamp(vad_reconstructed, min=-1, max=1)

class ExactInvertibleVADProjectionSAVING(nn.Module):
    def __init__(self, input_dim=3, output_dim=768):
        super().__init__()

        # Define W as a learnable parameter
        self.W = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.orthogonal_(self.W)  # Ensure W starts orthogonal

    def forward(self, vad):
        """ Forward: VAD (3D) -> Embedding (768D) """
        return torch.matmul(vad, self.W.T)

    def inverse(self, embedding):
        """ Inverse: Embedding (768D) -> VAD (560D) """
        # Orthogonal W, so inverse is just transpose
        vad_reconstructed = torch.matmul(embedding, self.W)
        return torch.clamp(vad_reconstructed, min=-1, max=1)

    def save_projection(self, path="projection_matrix.pt"):
        """ Save the learned projection matrix """
        torch.save(self.W.detach().cpu(), path)

    def load_projection(self, path="projection_matrix.pt"):
        """ Load the saved projection matrix """
        self.W.data = torch.load(path).to(self.W.device)

# Example Usage
model = ExactInvertibleVADProjectionSAVING()
model.save_projection("vad_projection.pt")



torch.manual_seed(42)

# Generate random VAD input (batch_size=5, input_dim=3)
vad_input = torch.rand(5, 3) * 2 - 1  # Random values in range [-1, 1]

# 1️⃣ Train Model: Save Projection Matrix
train_model = ExactInvertibleVADProjectionSAVING()
train_model.save_projection("vad_projection.pt")

# 2️⃣ First Model: Load and Forward Pass
eval_model_1 = ExactInvertibleVADProjectionOLD(path="vad_projection.pt")
embedding_1 = eval_model_1.forward(vad_input)
vad_reconstructed_1 = eval_model_1.inverse(embedding_1)

# 3️⃣ Second Model: Load and Forward Pass Again
eval_model_2 = ExactInvertibleVADProjectionOLD(path="vad_projection.pt")
embedding_2 = eval_model_2.forward(vad_input)
vad_reconstructed_2 = eval_model_2.inverse(embedding_2)

# 4️⃣ Compare Outputs
print("Embeddings Match:", torch.allclose(embedding_1, embedding_2))
print("Reconstructed VAD Match:", torch.allclose(vad_reconstructed_1, vad_reconstructed_2))