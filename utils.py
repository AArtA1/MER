from torch.utils.data import Dataset
import torch 
import numpy as np
from torch.utils.data import Dataset 
import cv2 
import os


def get_frames(path_to_video, start_frame, end_frame, transform = None):
    cap = cv2.VideoCapture(path_to_video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path_to_video}")

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if transform:
            frame = transform(frame)
        frames.append(frame)

    cap.release()

    return torch.tensor(np.stack(frames), dtype = torch.uint8) if len(frames) > 0 else torch.empty(0)


class CustomDataset(Dataset):
    def __init__(self, root_folder, data, transform):
        self.root_folder = root_folder
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        video_path = item["fpath"]
        label = item["class"]
        start_frame = item["start_frame"]
        end_frame = item["end_frame"]
        full_path = os.path.join(self.root_folder, video_path)

        frames = get_frames(full_path, start_frame, end_frame, self.transform)

        return frames, label