# if __name__ == "__main__":
#     import sys
#     #sys.path.append("C:/Users/aslan/Documents/MER/emotionbind/")
#     sys.path.append("/home/aaaslanyan_2/MER/emotionbind")


import pickle
from typing import Any
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
from emotionbind.config import DEVICE, ROOT_DIR


class SMGDatasetHandler(ExtendedDatasetHandler, Dataset): 
    def __init__(self, root_dir, video_dir, skeleton_dir, modalities=['video'], split='train', is_mapping = True): 
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

        labeling_path = os.path.join(ROOT_DIR, 'emotionbind/labeling')

        mapping = pd.read_csv(os.path.join(labeling_path, 'NRC-VAD-Lexicon.txt'), sep = '\t', header=None, names=["word", "x", "y", "z"])

        mapping['word'] = mapping['word'].astype(str)

        self.mapping = mapping.set_index('word')[['x', 'y', 'z']].apply(tuple, axis=1).to_dict()

        self.is_mapping = is_mapping

        with open(os.path.join(labeling_path, 'mappings/smg_mapping.pkl'), 'rb') as handle:
            dataset_mapping = pickle.load(handle)
        
        self.dataset_mapping = list(dataset_mapping.items())

        self.modalities = modalities

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
        if self.is_mapping:
            if label == 17: # non-gesture label
                emotion_vad = (0.5, 0.5, 0.5)
            else:
                emotion = self.dataset_mapping[label-1][-1]
                emotion_vad = self.mapping[emotion]

            return torch.tensor(emotion_vad)

        else:
            return self.vad_embeddings(torch.tensor(label - 1))

    def __getitem__(self, idx):
        """
        Loads a sample given an index.

        Returns:
        - Tuple: (video_tensor, skelet_tensor, label)
        """
        sample = self.data.iloc[idx]

        # video
        video_path = os.path.join(self.vis_embeddings_path, self.split, f'sample_{idx}.pt')

        if not os.path.exists(video_path):
            self.extract_video_feature(idx)

        video_tensor = torch.load(video_path, weights_only=True, map_location = 'cpu')

        if isinstance(video_tensor, np.ndarray):
            video_tensor = torch.tensor(video_tensor, dtype=torch.float32)

        video_tensor = video_tensor.unsqueeze(0)

        # skeleton
        # skeleton_path = os.path.join(self.skeleton_embeddings_path, self.split, f'sample_{idx}.pt')
        
        # if not os.path.exists(skeleton_path):
        #     self.extract_skeleton_feature(idx)

        # skeleton_tensor = torch.load(skeleton_path, weights_only=True, map_location = 'cpu')

        skeleton_tensor = None

        if isinstance(skeleton_tensor, np.ndarray):
            skeleton_tensor = torch.tensor(skeleton_tensor, dtype=torch.float32)

        vad_label = self.convert_labels_to_VAD_vector(sample["class"])

        if self.split == 'train':
            vad_label = self.vad_projector(vad_label)

        return {
            "video": video_tensor,
            #"skeleton": skeleton_tensor,
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

    smg = SMGDatasetHandler('/home/aaaslanyan_2/MER/SMG', '', split = 'train', is_mapping = True)

    # smg.extract_video_features(4)

    print([smg[i] for i in range(5)])

    # print(*[smg.convert_labels_to_VAD_vector(i) for i in range (1, 18)])

    #smg.extract_video_feature(2448)

    # #print(smg.load_data()[0][0]) 