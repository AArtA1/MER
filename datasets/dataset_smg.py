if __name__ == "__main__":
    import sys
    sys.path.append("C:/Users/aslan/Documents/MER/emotionbind/")

import torch
from torch import nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from torchvision.transforms.functional import convert_image_dtype
from torch.utils.data import Dataset, DataLoader

import numpy as np 
from sklearn.metrics import (   accuracy_score,  
                                precision_recall_fscore_support,  
                                mean_squared_error, 
                                mean_absolute_error,  
                                r2_score 
                            )
import os 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from emotionbind.datasets.datasets import ExtendedDatasetHandler

from emotionbind.models.emotionbind_model import AttentionPooling, ExactInvertibleVADProjection
from emotionbind.utils.vad_utils import translate_VAD


class SMGDatasetHandler(ExtendedDatasetHandler, Dataset): 
    def __init__(self, root_dir, video_dir, modalities=['video'], split='train'): 
        """ 
        Initialize with the root folder of the dataset. 
 
        Args: 
        - root_dir (str): The main folder containing dataset files. 
        - split (str): Dataset split to load ('train', 'test', or 'valid'). 
        """ 
        super().__init__(root_dir, split) 
        self.csv_folder = os.path.join(root_dir, "smg_csv") 
        self.rgb_folders = {
            "SMG_RGB_Phase1": os.path.join(root_dir, "SMG_RGB_Phase1"), 
            "SMG_RGB_Phase2": os.path.join(root_dir, "SMG_RGB_Phase2"), 
        }

        self.modalities = modalities

        self.vis_embeddings_path = os.path.join(self.root_dir, 'video_embeddings')
        self.set_split(self.split)

        self.video_pool = AttentionPooling(feature_dim=768, output_len=10)

        self.vad_embeddings = nn.Embedding(17, 3) # 17 gestures in VAD space for SMG
        
        self.vad_projector = nn.Linear(3, 768)

        # self.vad_projector = ExactInvertibleVADProjection(input_dim=3, output_dim=768)

    def convert_labels_to_VAD_vector(self, label: int): 
        """ 
        Converts the dataset's emotion labels into a vector 
        in the 3D Valence-Arousal-Dominance cube. 
 
        Args: 
        - label (int): The original emotion label from the dataset. 
 
        Returns: 
        - tuple: A 3D vector (Valence, Arousal, Dominance). 
        """ 

        return self.vad_embeddings(torch.tensor(label - 1))

        # label_to_vad = { 
        #     "happy": (1.0, 0.8, 0.6), 
        #     "sad": (-0.8, -0.7, -0.5), 
        #     "angry": (-0.6, 0.9, 0.8), 
        #     "neutral": (0.0, 0.0, 0.0), 
        # } 
        # return torch.tensor(label_to_vad.get(label.lower(), (0.0, 0.0, 0.0))) 

    def __getitem__(self, idx):
        """
        Loads a sample given an index.

        Returns:
        - Tuple: (video_tensor, skelet_tensor, label)
        """
        sample = self.data.iloc[idx]

        video_path = os.path.join(self.vis_embeddings_path, self.split, f'sample_{idx}.pt')
        try:
            video_tensor = torch.load(video_path, weights_only=True)
        except:
            self.extract_video_feature(idx)

        if isinstance(video_tensor, np.ndarray):
            video_tensor = torch.tensor(video_tensor, dtype=torch.float32)

        # if isinstance(audio_tensor, np.ndarray):
        #     audio_tensor = torch.tensor(audio_tensor, dtype=torch.float32)

        # TODO:

        # video_tensor = self.video_pool(video_tensor.unsqueeze(0))
        # audio_tensor = self.audio_pool(audio_tensor)

        vad_label = self.convert_labels_to_VAD_vector(sample["class"])

        if self.split == 'train':
            vad_label = self.vad_projector(vad_label)

        return {
            "video": video_tensor.unsqueeze(0),
            #"audio": audio_tensor,
            #"text": text_tensor,
            "label": vad_label,
            "categorical_label": sample["class"],
        }
    
    def __len__(self):
        return len(self.data)

    def set_split(self, split):
        assert split in ['train', 'test', 'valid']
        self.split = split
        self.split_file = os.path.join(self.csv_folder, f"smg_{split}.csv")
        self.data = pd.read_csv(self.split_file)





if __name__ == "__main__":

    smg = SMGDatasetHandler('C:/Users/aslan/Documents/MER/SMG', '', split = 'train')

    smg.extract_video_features(4)

    #smg.extract_video_feature(2448)

    # #print(smg.load_data()[0][0]) 