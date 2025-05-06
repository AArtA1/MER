import os
import regex as re
import torch
import chardet
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def detect_encoding( file_path):
    """Detects encoding of a given file."""
    with open(file_path, "rb") as f:
        raw_data = f.read(5000)
        detected = chardet.detect(raw_data)
        return detected["encoding"] if detected["encoding"] else "utf-8"

# Dataset Class
class IEMOCAPDataset(Dataset):
    def __init__(self, root_dir, sessions):
        self.samples = []
        for session in sessions:
            eval_path = os.path.join(root_dir, session, "dialog", "EmoEvaluation")
            if not os.path.exists(eval_path):
                continue
            for file in os.listdir(eval_path):
                if file.endswith(".txt"):
                    file_path = os.path.join(eval_path, file)
                    encoding = detect_encoding(file_path)
                    with open(file_path, "r", encoding=encoding) as f:
                        lines = f.readlines()
                    for line in lines:
                        if "[" in line and "]" in line:
                            parts = line.split()
                            if len(parts) < 4:
                                continue
                            speakerId = next((p for p in parts if p.startswith("Ses")), None)
                            if not speakerId:
                                continue
                            filename = "_".join(speakerId.split("_")[:-1])
                            match = re.search(r'\[([-\d.,\s]+)\](?!.*\])', line)
                            if not match:
                                continue
                            vad_values = list(map(float, match.group(1).split(',')))
                            categorical_label = parts[4].strip("'\"")
                            self.samples.append((vad_values, categorical_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vad_values, categorical_label = self.samples[idx]
        return torch.tensor(vad_values, dtype=torch.float32), categorical_label


class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=3, num_classes=6, num_heads=3, num_layers=2, hidden_dim=64):
        super(EmotionClassifier, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        return self.fc(x)


# Training Function
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        total_loss = 0.0
        for vad_values, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(vad_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")


# Main Execution
if __name__ == "__main__":
    root_dir = "D:/IEMOCAP"
    sessions = ["Session1", "Session2", "Session3", "Session4", "Session5"]
    dataset = IEMOCAPDataset(root_dir, sessions)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = EmotionClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, criterion, optimizer)
    torch.save(model.state_dict(), "checkpoints/emotion_classifier.pth")
    print("Model training complete and saved!")

    # Example Usage
    #model.load_state_dict(torch.load("emotion_classifier.pth"))
    #model.eval()
    #sample_input = torch.tensor([[0.5, 0.2, 0.7]], dtype=torch.float32)
    #print("Predicted Emotion:", torch.argmax(model(sample_input), dim=1).item())
