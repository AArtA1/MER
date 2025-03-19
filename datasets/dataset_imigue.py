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


class iMiGUEDatasetHandler(ExtendedDatasetHandler, Dataset): 
    def __init__(self, root_dir, video_dir, modalities=['video'], split='train'): 
        """ 
        Initialize with the root folder of the dataset. 
 
        Args: 
        - root_dir (str): The main folder containing dataset files. 
        - split (str): Dataset split to load ('train', 'test', or 'valid'). 
        """ 
        super().__init__(root_dir, split) 
        self.csv_folder = os.path.join(root_dir, "imigue_csv") 
        self.rgb_folders = {
            "iMiGUE_RGB_Phase1": os.path.join(root_dir, "iMiGUE_RGB_Phase1"), 
            "iMiGUE_RGB_Phase2": os.path.join(root_dir, "iMiGUE_RGB_Phase2"), 
        }
        
        self.modalities = modalities

        self.vis_embeddings_path = os.path.join(self.root_dir, 'video_embeddings')
        self.set_split(self.split)

        self.video_pool = AttentionPooling(feature_dim=768, output_len=10)

        self.vad_embeddings = nn.Embedding(33, 3) # 17 gestures in VAD space for SMG
        
        self.vad_projector = nn.Linear(3, 768)


    def convert_labels_to_VAD_vector(self, label): 
        """ 
        Converts the dataset's emotion labels into a vector 
        in the 3D Valence-Arousal-Dominance cube. 
 
        Args: 
        - label (str): The original emotion label from the dataset. 
 
        Returns: 
        - tuple: A 3D vector (Valence, Arousal, Dominance). 
        """ 
        label = 33 if label == 99 else label 
        
        return self.vad_embeddings(torch.tensor(label - 1))
        
        # return label_to_vad.get(label.lower(), (0.0, 0.0, 0.0)) 

    
    def __getitem__(self, idx):
        """
        Loads a sample given an index.

        Returns:
        - Tuple: (video_tensor, skelet_tensor, label)
        """
        sample = self.data.iloc[idx]

        video_path = os.path.join(self.vis_embeddings_path, self.split, f'sample_{idx}.pt')
        video_tensor = torch.load(video_path, weights_only=True)

        if isinstance(video_tensor, np.ndarray):
            video_tensor = torch.tensor(video_tensor, dtype=torch.float32)

        # if isinstance(audio_tensor, np.ndarray):
        #     audio_tensor = torch.tensor(audio_tensor, dtype=torch.float32)

        # TODO:

        # video_tensor = self.video_pool(video_tensor.unsqueeze(0))
        # audio_tensor = self.audio_pool(audio_tensor)

        vad_label = self.convert_labels_to_VAD_vector('happy')

        if self.split == 'train':
            vad_label = self.vad_projector(vad_label)

        return {
            "video": video_tensor,
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
        self.split_file = os.path.join(self.csv_folder, f"imigue_{split}.csv")     
        self.data = pd.read_csv(self.split_file)



if __name__ == "__main__":
    imigue_dataset = iMiGUEDatasetHandler('C:/Users/aslan/Documents/MER/iMiGUE', 'train')
        
    for split in ['train', 'valid', 'test']:
        imigue_dataset.set_split(split)
        imigue_dataset.extract_video_features(batch_size=4)




