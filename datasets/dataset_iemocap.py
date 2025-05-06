import os
import re
import chardet
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pickle
from .datasets import BaseDatasetHandler
from emotionbind.models.emotionbind_model import AttentionPooling, ExactInvertibleVADProjection
from emotionbind.utils.vad_utils import translate_VAD

class IEMOCAPDatasetHandler(BaseDatasetHandler, Dataset):
    def __init__(self, root_dir, video_dir, audio_dir, text_dir,
                 modalities=['text', 'audio', 'video'], split='train'):
        """
        Args:
            root_dir (str): Path to dataset root folder.
            modalities (list): Modalities to include (e.g., ["text", "audio", "video"]).
            split (str): Train/test split.
        """
        super().__init__(root_dir, split)
        # Feature directories
        self.root_dir = root_dir
        self.video_dir = os.path.normpath(video_dir)
        self.audio_dir = os.path.normpath(audio_dir)
        self.text_feature_dir = os.path.normpath(text_dir)
        self.text_file = self.find_text_feature_file(self.text_feature_dir)


        self.modalities = modalities
        self.split = split

        self.vad_projector = ExactInvertibleVADProjection(input_dim=3, output_dim=768)


        # Load text embeddings safely
        with open(self.text_file, "rb") as f:
            self.text_features = pickle.load(f)

        self.audio_pool = AttentionPooling(feature_dim=768, output_len=10)
        self.video_pool = AttentionPooling(feature_dim=768, output_len=10)

        # Load dataset
        self.data = self.load_data()
        #self.max_video_len, self.max_audio_len = self.get_max_lengths()

    def get_max_lengths(self):
        """Find max sequence lengths for video and audio across dataset."""
        max_video_len = 0
        max_audio_len = 0
        for sample in self.data:
            video_tensor = torch.load(sample["video"], pickle_module=pickle)
            audio_tensor = torch.load(sample["audio"], pickle_module=pickle)

            max_video_len = max(max_video_len, video_tensor.shape[0])
            max_audio_len = max(max_audio_len, audio_tensor.shape[0])

        return max_video_len, max_audio_len

    def find_text_feature_file(self, folder_path):
        """Searches for a .pkl text feature file in the given directory."""
        for file in os.listdir(folder_path):
            if file.endswith(".pkl"):
                return os.path.join(folder_path, file)
        raise FileNotFoundError(f"No text feature file found in {folder_path}")

    def detect_encoding(self, file_path):
        """Detects encoding of a given file."""
        with open(file_path, "rb") as f:
            raw_data = f.read(5000)
            detected = chardet.detect(raw_data)
            return detected["encoding"] if detected["encoding"] else "utf-8"

    def pad_sequence(self, tensor, max_len):
        """Pads sequence to max_len and returns attention mask."""
        seq_len, feat_dim = tensor.shape
        pad_len = max_len - seq_len

        # Create attention mask: 1 for real tokens, 0 for padding
        attention_mask = torch.ones(max_len, dtype=torch.float32)
        if pad_len > 0:
            attention_mask[seq_len:] = 0  # Mark padding positions

            # Pad sequence (pad last dimension)
            tensor = F.pad(tensor, (0, 0, 0, pad_len), value=0)

        return tensor, attention_mask

    def load_data(self):
        """
        Loads annotations and feature paths.

        Returns:
        - List of dictionaries: { 'video': path, 'audio': path, 'text': vector, 'label': VAD }
        """
        all_samples = []

        if self.split == 'train':
            sessions = ['Session1', 'Session2', 'Session3', 'Session4']
        elif self.split == 'test':
            sessions = ['Session5']
        else:
            raise ValueError(f"Unknown split '{self.split}'. Use 'train' or 'test'.")

        # encoding = detect_encoding(self.root_dir)
        categorical_label = "xxx"

        for session in sessions:
            eval_path = os.path.join(self.root_dir, session, "dialog", "EmoEvaluation")
            if not os.path.exists(eval_path):
                continue

            for file in os.listdir(eval_path):
                if file.endswith(".txt"):
                    file_path = os.path.join(eval_path, file)
                    encoding = self.detect_encoding(file_path)
                    with open(file_path, "r", encoding=encoding) as f:
                        lines = f.readlines()

                    for line in lines:
                        if "[" in line and "]" in line:  # Emotion labels are inside brackets
                            parts = line.split()
                            if len(parts) < 4:
                                continue

                            speakerId = next((p for p in parts if p.startswith("Ses")), None)
                            if not speakerId:
                                print(f"Skipping line due to missing speakerId: {line}")
                                continue  # Skip this line if no valid speaker ID is found
                            filename = "_".join(speakerId.split("_")[:-1])

                            # extract VAD values
                            try:
                                match = re.search(r'\[([-\d.,\s]+)\](?!.*\])', line)
                                if not match:
                                    print(f"Skipping line with missing VAD values: {line}")
                                    continue

                                vad_values_str = match.group(1)
                                vad_values = list(map(float, vad_values_str.split(',')))
                                vad_values = translate_VAD(vad_values, direction="to_norm")

                                # Extract categorical label
                                parts = line.split()
                                if len(parts) >= 4:
                                    categorical_label = parts[4].strip("'\"")

                            except ValueError:
                                print(f"Skipping line with invalid VAD values: {line}")
                                continue

                            video_feat = os.path.join(self.video_dir, f"{speakerId}.pt")
                            # print("Video feat: " + video_feat)
                            audio_feat = os.path.join(self.audio_dir, f"{speakerId}.pt")
                            # print("Audio feat: " + audio_feat)

                            text_session_data = self.text_features.get(session, {})
                            entries_list = text_session_data.get(filename + ".txt", None)
                            if not entries_list:
                                raise KeyError(f"Text features for {filename}.txt NOT found in session {session}!")
                            text_feat_entry = next((entry for entry in entries_list if entry["speaker"] == speakerId), None)
                            if not text_feat_entry:
                                raise KeyError(
                                    f"No matching speaker '{filename}' found in '{filename}.txt' under session '{session}'!")
                            text_feat = torch.tensor(text_feat_entry["feature_vector"], dtype=torch.float32)
                            # print("Text feat type: " + str(type(text_feat)))


                            video_exists = os.path.exists(video_feat) and os.path.getsize(video_feat) > 0
                            audio_exists = os.path.exists(audio_feat) and os.path.getsize(audio_feat) > 0
                            text_valid = text_feat is not None and text_feat.nelement() > 0

                            status_message = f"Processing sample: {speakerId} | " \
                                             f"Video: {'OK' if video_exists else 'MISSING'} | " \
                                             f"Audio: {'OK' if audio_exists else 'MISSING'} | " \
                                             f"Text: {'OK' if text_valid else 'INVALID'}"

                            print(status_message)


                            if os.path.exists(video_feat) and os.path.exists(audio_feat) and text_feat is not None:
                                # print("Adding the sample: ")
                                all_samples.append({
                                    "video": video_feat,
                                    "audio": audio_feat,
                                    "text": text_feat,
                                    "label": torch.tensor(vad_values, dtype=torch.float32),
                                    "categorical_label": categorical_label,
                                })

        return all_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Loads a sample given an index.

        Returns:
        - Tuple: (video_tensor, audio_tensor, text_tensor, label)
        """
        sample = self.data[idx]

        video_tensor = torch.load(sample["video"], pickle_module=pickle)
        audio_tensor = torch.load(sample["audio"], pickle_module=pickle)
        text_tensor = sample["text"].unsqueeze(0)

        if isinstance(video_tensor, np.ndarray):
            video_tensor = torch.tensor(video_tensor, dtype=torch.float32)

        if isinstance(audio_tensor, np.ndarray):
            audio_tensor = torch.tensor(audio_tensor, dtype=torch.float32)

        video_tensor = self.video_pool(video_tensor)
        audio_tensor = self.audio_pool(audio_tensor)

        if self.split == 'train':
            vad_label = self.vad_projector(sample["label"])
        else:
            vad_label = sample["label"]  # Keep labels as 3D for testing/evaluation

        return {
            "video": video_tensor,
            "audio": audio_tensor,
            "text": text_tensor,
            "label": vad_label,
            "categorical_label": sample["categorical_label"],
        }
