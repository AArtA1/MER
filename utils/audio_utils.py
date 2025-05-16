import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from emotionbind.utils.vad_utils import find_closest_key, find_closest_label




class ExactInvertibleVADProjection(nn.Module):
    def __init__(self, input_dim=3, output_dim=768, num_classes=10, label_path="../datasets/labels/iemocap_cat_labels.pkl"):
        super().__init__()

        # Create multiple orthogonal weight matrices for different emotions
        #self.W_blocks = nn.ParameterList([
        #    nn.Parameter(self._init_orthogonal(input_dim, output_dim)) for _ in range(num_classes)
        #])

        self.num_classes = num_classes
        self.label_path = label_path

        loaded_data = np.load("vad_projection_matrices.npz")
        W_matrices = [loaded_data[key] for key in loaded_data.files]
        print(W_matrices)

        with open(self.label_path, "rb") as f:
            self.vad_to_label = pickle.load(f)


            # Print available keys for debugging
            print("Keys in vad_projection_matrices.npz:", loaded_data.files)

            for key in loaded_data.files:
                W_np = loaded_data[key]
                print(W_np)
                W_torch = torch.tensor(W_np, dtype=torch.float32)

                I_approx_np = W_np @ W_np.T
                I_approx_torch = W_torch @ W_torch.T

                np_error = np.linalg.norm(I_approx_np - np.eye(W_np.shape[0]))  # Frobenius norm
                torch_error = torch.norm(I_approx_torch - torch.eye(W_torch.shape[0]))  # PyTorch equivalent

                print(f"Orthogonality Error (NumPy)  for {key}: {np_error:.6f}")
                print(f"Orthogonality Error (Torch)  for {key}: {torch_error.item():.6f}")

            for i, W in enumerate(W_matrices):
                I_approx = W.T @ W  # (3,3)
                I_approx_other = W @ W.T  # (768,768)

                # Correct identity matrix sizes
                identity_error = np.linalg.norm(I_approx - np.eye(W.shape[1]))  # (3,3)
                identity_error_o = np.linalg.norm(I_approx_other - np.eye(W.shape[0]))  # (768,768)

                print(
                    f"DEBUG: Orthogonality Error in dataset handler for W[{i}]: {identity_error:.6f} +  {identity_error_o:.6f}")

            # Load W_blocks dynamically from arr_0, arr_1, ..., arr_N
            self.W_blocks = nn.ParameterList([
                nn.Parameter(torch.tensor(loaded_data[key], dtype=torch.float32).clone()) for key in loaded_data.files
            ])

            for i, W in enumerate(self.W_blocks):
                W_np = W.detach().cpu().numpy()
                I_approx = W_np @ W_np.T  # Should be close to identity
                identity_error = np.linalg.norm(I_approx - np.eye(W_np.shape[0]))  # Frobenius norm
                print(f"Orthogonality Error for W[{i}]: {identity_error:.6f}")

        self.cluster_centers = np.load("adjusted_cluster_centers.npz")
        self.cluster_centers = {key: self.cluster_centers[key] for key in self.cluster_centers.files}

        self.label_mapping = {label: idx for idx, label in enumerate(self.cluster_centers.keys())}

    def _init_orthogonal(self, input_dim, output_dim):
        W = torch.empty(output_dim, input_dim)
        nn.init.orthogonal_(W)
        return W

    def forward(self, vad):
        """ Forward: VAD (3D) -> Embedding (768D) """
        embeddings = []
        for v in vad:
            rounded_v = tuple(map(lambda x: round(x.item(), 5), v))  # Round to 5 decimal places
            # print(rounded_v)
            label = self.vad_to_label.get(rounded_v, self.vad_to_label[find_closest_key(rounded_v, self.vad_to_label.keys())])
            W = self.W_blocks[label]  # Use the corresponding orthogonal matrix
            embeddings.append(torch.matmul(v, W.T))
        return torch.stack(embeddings)

    """

    def inverse(self, embedding):
        #Inverse: Embedding (768D) -> VAD (3D) using cluster centers and W_blocks 
        reconstructed_vads = []

        for emb in embedding:
            emb_np = emb.detach().cpu().numpy()

            # Find the closest cluster center (label)
            closest_center = min(self.cluster_centers, key=lambda lbl: np.linalg.norm(self.cluster_centers[lbl] - emb_np))

            # Retrieve the correct weight matrix
            W = self.W_blocks[self.label_mapping[closest_center]]  # Correct index lookup
            W_pinv = torch.linalg.pinv(W)  # Compute pseudo-inverse

            vad_reconstructed = torch.matmul(emb, W_pinv.T)
            vad_reconstructed = torch.clamp(vad_reconstructed, min=-1, max=1)  # Ensure valid range
            reconstructed_vads.append(vad_reconstructed)

        return torch.stack(reconstructed_vads)"""

    def inverse(self, embedding):
        """ Exact Inverse: Embedding (768D) -> VAD (3D) """
        reconstructed_vads = []
        for emb in embedding:
            emb_np = emb.detach().cpu().numpy()

            # Find the closest cluster center (label)
            closest_center = min(self.cluster_centers,
                                 key=lambda lbl: np.linalg.norm(self.cluster_centers[lbl] - emb_np))

            # Retrieve the correct weight matrix (which is orthogonal)
            W = self.W_blocks[self.label_mapping[closest_center]]  # Get the corresponding W

            # Exact inverse using transpose (since W is orthogonal)
            W_inv = W.T  # Instead of pinv

            vad_reconstructed = torch.matmul(emb, W_inv.T)
            vad_reconstructed = torch.clamp(vad_reconstructed, min=-1, max=1)  # Ensure valid range
            reconstructed_vads.append(vad_reconstructed)

        return torch.stack(reconstructed_vads)





### Step 1: Load and Save Label Tensor ###

# Load VAD dictionary
with open("sorted_vad_labels.pkl", "rb") as f:
    vad_dict = pickle.load(f)  # { (V,A,D): "label" }

# Convert dictionary to tensors
vad_values = np.array(list(vad_dict.keys()))  # Shape: (N, 3)
# print(vad_values)
labels = np.array(list(vad_dict.values()))  # Corresponding categorical labels
# print(labels)

# Convert categorical labels to numerical indices
unique_labels, label_indices = np.unique(labels, return_inverse=True)
print("LENGTH = " + str(len(unique_labels)))
label_mapping = {label: idx for label, idx in zip(unique_labels, range(len(unique_labels)))}

# vad_to_label = {tuple(map(float, np.round(v, 5))): label_mapping[label] for v, label in vad_dict.items()}
vad_to_label = {tuple(map(lambda x: round(float(x), 5), v)): label_mapping[label] for v, label in vad_dict.items()}


print("Keys in vad_to_label:", list(vad_to_label.keys())[:10])  # Print first 10 keys




# Save full mapping instead of just labels
save_path = "iemocap_cat_labels.pkl"
with open(save_path, "wb") as f:
    pickle.dump(vad_to_label, f)  # ✅ Save as a dictionary

### Step 2: Convert VAD Values to Tensor ###
vad_tensor = torch.tensor(vad_values, dtype=torch.float32)

### Step 3: Initialize Model and Project VAD -> 768D ###
model = ExactInvertibleVADProjection(num_classes=len(unique_labels), label_path=save_path)
vad_embeddings = model.forward(vad_tensor).detach().numpy()  # Shape: (N, 768)
"""
# Save projection matrices
projection_save_path = "vad_projection_matrices.pkl"
projection_data = {
    "W_blocks": [W.detach().cpu().numpy() for W in model.W_blocks],  # Convert to NumPy
    "unique_labels": unique_labels  # Store labels for reference
}

with open(projection_save_path, "wb") as f:
    pickle.dump(projection_data, f)

print(f"Saved projection matrices to {projection_save_path}")
"""

cluster_centers = {}
for label_idx in range(len(unique_labels)):
    cluster_centers[unique_labels[label_idx]] = np.mean(
        vad_embeddings[label_indices == label_idx], axis=0
    )

# Save cluster centers as a file
# np.savez("cluster_centers.npz", **cluster_centers)
# print("\n✅ Cluster centers saved to cluster_centers.npz!")


### Step 4: Invertibility Test ###
print("\nTesting Invertibility...\n")
random_vad = torch.rand((20, 3)) * 2 - 1  # Generate 20 random VAD points in range [-1, 1]

# Forward pass (VAD -> 768D)
embedded_vad = model.forward(random_vad)

# Inverse pass (768D -> VAD)
reconstructed_vad = model.inverse(embedded_vad)

# Compute reconstruction error
error = torch.norm(random_vad - reconstructed_vad, dim=1)  # L2 norm per sample

# Print original vs reconstructed values
for i in range(5):  # Print first 5 for brevity
    print(f"Original VAD  : {random_vad[i].tolist()}")
    print(f"Reconstructed : {reconstructed_vad[i].tolist()}")
    print(f"Error (L2 norm): {error[i].item():.6f}\n")

print(f"Average Reconstruction Error: {error.mean().item():.6f}")

for i, W in enumerate(model.W_blocks):
    W_np = W.detach().cpu().numpy()
    W_pinv = np.linalg.pinv(W_np)
    identity_approx = W_np @ W_pinv
    print(f"Reconstruction error (I - W @ W⁺) for W[{i}]:", np.linalg.norm(identity_approx - np.eye(W_np.shape[0])))

for emb in embedded_vad[:5]:  # Check first 5 embeddings
    emb_np = emb.detach().cpu().numpy()
    closest_center = min(model.cluster_centers, key=lambda lbl: np.linalg.norm(model.cluster_centers[lbl] - emb_np))
    print(f"Embedding: {emb_np[:5]}... Closest Cluster: {closest_center} Error: {np.linalg.norm(model.cluster_centers[closest_center] - emb_np):.6f}")


### Step 5: Visualization ###

# Assign a unique color to each label
color_palette = sns.color_palette("hsv", len(unique_labels))
colors = np.array([color_palette[i] for i in label_indices])  # Map labels to colors

# Reduce to 2D using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(vad_embeddings)

# Reduce to 2D using t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(vad_embeddings)

cluster_embeddings = np.array(list(cluster_centers.values()))  # Convert dict values to array
pca_cluster_centers = pca.transform(cluster_embeddings)
tsne_cluster_centers = TSNE(n_components=2, perplexity=8, random_state=42).fit_transform(
    np.array(list(cluster_centers.values()))
)


# Plot PCA result
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, alpha=0.7)
plt.scatter(pca_cluster_centers[:, 0], pca_cluster_centers[:, 1], c='black', marker='o', s=100, label="Cluster Centers")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Visualization of VAD Embeddings (Projected to 768D)")
plt.legend(handles=[plt.Line2D([0], [0], marker="o", color="w", markersize=10,
                                markerfacecolor=color_palette[i], label=label)
                    for i, label in enumerate(unique_labels)], title="Emotion Categories",
           bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()

# Plot t-SNE result
plt.figure(figsize=(8, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, alpha=0.7)
plt.xlabel("t-SNE Component 1")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization of VAD Embeddings (Projected to 768D)")
plt.legend(handles=[plt.Line2D([0], [0], marker="o", color="w", markersize=10,
                                markerfacecolor=color_palette[i], label=label)
                    for i, label in enumerate(unique_labels)], title="Emotion Categories",
           bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()
