import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from torchvision.transforms.functional import convert_image_dtype
from torch.utils.data import Dataset, DataLoader

import os
from tqdm import tqdm

from sklearn.metrics import accuracy_score, \
                            precision_recall_fscore_support, \
                            mean_squared_error, \
                            mean_absolute_error, \
                            r2_score


from abc import ABC, abstractmethod

from emotionbind.utils.video_utils import get_frames, sample_frames_uniform
from emotionbind.config import DEVICE
from emotionbind.utils.skeleton_utils import extract_pose_from_tensor, extract_skeleton_embeddings


class BaseDatasetHandler(Dataset):
    def __init__(self, root_dir, split):
        """
        Initialize with the root folder of the dataset.

        Args:
        - root_folder (str): The main folder containing dataset files.
        """
        self.root_dir = root_dir
        self.split = split
        self.data = None # self.load_data()


    def load_data(self):
        """
        Load data based on split (e.g., train/test).
        Implement in subclasses.

        """
        raise NotImplementedError("This method needs to be implemented in subclasses.")

    def convert_labels_to_VAD_vector(self, label):
        """
        Converts the dataset's emotion labels into a vector
        in the 3D Valence-Arousal-Dominance cube.

        Args:
        - label: The original emotion label from the dataset.

        Returns:
        - tuple: A 3D vector (Valence, Arousal, Dominance).
        """
        raise NotImplementedError("This method needs to be implemented in subclasses.")

    def convert_VAD_vector_to_dataset_labels(self, label):
        """
        Converts the 3D Valence-Arousal-Dominance vector to
        dataset's emotion labels format.

        Args:
        - label: The original emotion label from the dataset.

        Returns:
        - tuple: A 3D vector (Valence, Arousal, Dominance).
        """
        raise NotImplementedError("This method needs to be implemented in subclasses.")

    def evaluate(self, split='test'):
        """
        Perform evaluation on the dataset split (e.g., 'train', 'test').

        Args:
        - split (str): Dataset split to evaluate on.

        Returns:
        - dict: Computed metrics.
        """
        # Load dataset
        data_loader = self.dataset_handler.load_data(split)
        ground_truth, predictions = [], []

        for data, label in data_loader:
            pred = self.model.predict(data)
            ground_truth.append(label)
            predictions.append(pred)

        if isinstance(ground_truth[0], int):
            accuracy = accuracy_score(ground_truth, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predictions, average='weighted')

            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }

        elif isinstance(ground_truth[0], (list, np.ndarray)):  # VAD vectors
            ground_truth = np.array(ground_truth)
            predictions = np.array(predictions)
            mse = mean_squared_error(ground_truth, predictions)
            mae = mean_absolute_error(ground_truth, predictions)
            r2 = r2_score(ground_truth, predictions)

            metrics = {
                "mean_squared_error": mse,
                "mean_absolute_error": mae,
                "r2_score": r2,
            }

        else:
            raise ValueError("unknown label format")

        return metrics

 
class ExtendedDatasetHandler(ABC): 
    def __init__(self, root_dir, split): 
        """ 
        Initialize with the root folder of the dataset. 
 
        Args: 
        - root_dir (str): The main folder containing dataset files. 
        """ 
        self.root_dir = root_dir
        self.split = split
        self.split_file = None
        self.vis_embeddings_path = os.path.join(self.root_dir, 'video_embeddings')
        self.skeleton_embeddings_path = os.path.join(self.root_dir, 'skeleton_embeddings')

    @abstractmethod
    def convert_labels_to_VAD_vector(self, label): 
        """ 
        Converts the dataset's emotion labels into a vector 
        in the 3D Valence-Arousal-Dominance cube. 
 
        Args: 
        - label: The original emotion label from the dataset. 
 
        Returns: 
        - tuple: A 3D vector (Valence, Arousal, Dominance). 
        """ 
        raise NotImplementedError("This method needs to be implemented in subclasses.") 
 
    def evaluate(self, split='test'): 
        """ 
        Perform evaluation on the dataset split (e.g., 'train', 'test'). 
 
        Args: 
        - split (str): Dataset split to evaluate on. 
 
        Returns: 
        - dict: Computed metrics. 
        """ 
        # Load dataset 
        # data_loader = self.dataset_handler.load_data(split) 
        
        data_loader = self.load_data(split)
        ground_truth, predictions = [], [] 
 
        for data, label in data_loader: 
            pred = self.model.predict(data) 
            ground_truth.append(label) 
            predictions.append(pred) 
 
        if isinstance(ground_truth[0], int): 
            accuracy = accuracy_score(ground_truth, predictions) 
            precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predictions, average='weighted') 
 
            metrics = { 
                "accuracy": accuracy, 
                "precision": precision, 
                "recall": recall, 
                "f1_score": f1, 
            } 
 
        elif isinstance(ground_truth[0], (list, np.ndarray)):  # VAD vectors 
            ground_truth = np.array(ground_truth) 
            predictions = np.array(predictions) 
            mse = mean_squared_error(ground_truth, predictions) 
            mae = mean_absolute_error(ground_truth, predictions) 
            r2 = r2_score(ground_truth, predictions) 
 
            metrics = { 
                "mean_squared_error": mse, 
                "mean_absolute_error": mae, 
                "r2_score": r2, 
            }

        else: 
            raise ValueError("unknown label format") 

        return metrics 
    
    def extract_skeleton_features(self, batch_size=10):
        """
        Save tensors for the SMG dataset in batches.

        This function reads the configuration, groups the data by the 'fpath' column,
        sorts it by the 'start_frame' column, and then saves each video separately
        as a CSV file in the specified destination path.

        Args:
        - batch_size (int, optional): Number of videos to process in each batch.
        """
        data = self.data

        dst_path = os.path.join(self.skeleton_embeddings_path, self.split)

        os.makedirs(dst_path, exist_ok=True)

        def get_tensors(row):
            video_path = row["fpath"]
            start_frame = row["start_frame"]
            end_frame = row["end_frame"]
            full_path = os.path.join(self.root_dir, video_path)
            return get_frames(full_path, start_frame, end_frame)

        config_path = 'configs/skeleton/2s-agcn/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.py'
        config = Config.fromfile(config_path)
        # Setup a checkpoint file to load
        checkpoint = 'checkpoints/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221222-4c0ed77e.pth'
        # Initialize the recognizer
        model = init_recognizer(config, checkpoint, DEVICE='cpu')

        model.cls_head.fc = nn.Identity()

        # Используем предобученную модель на COCO
        inferencer = MMPoseInferencer(
            pose2d='human',  # использует HRNet или другую модель по умолчанию
        )

        batch = []
        with tqdm(total=len(data), desc="Processing", unit=" samples") as pbar:
            for index, row in data.iterrows():
                input_tensor = get_tensors(row)
                
                keypoints_seq, scores_seq = extract_pose_from_tensor(frames, inferencer)

                embeddings = extract_skeleton_embeddings(model, keypoints_seq, scores_seq)

                batch.append(embeddings)

                if len(batch) == batch_size:
                    input_batch = torch.stack(batch, dim=0).to(DEVICE)
                    encoded_batch = encoder(input_batch)

                    for i, tensor in enumerate(encoded_batch):
                        sample_index = index - batch_size + i + 1
                        sample_path = os.path.join(dst_path, f'sample_{sample_index}.pt')
                        torch.save(tensor, sample_path)

                    batch = []

                pbar.update(1)
                pbar.update(1)

        if batch:
            tensors_batch = torch.stack(batch, dim = 0).to(DEVICE) 
            # tensors_batch = transformation(tensors_batch)
            encoded_batch = encoder(tensors_batch)

            for i, tensor in enumerate(encoded_batch):
                sample_index = len(data) - len(batch) + i + 1
                sample_path = os.path.join(dst_path, f'sample_{sample_index}.pt')
                torch.save(tensor, sample_path)

    def extract_video_features(self, batch_size=10):
        """
        Save tensors for the SMG dataset in batches.

        This function reads the configuration, groups the data by the 'fpath' column,
        sorts it by the 'start_frame' column, and then saves each video separately
        as a CSV file in the specified destination path.

        Args:
        - batch_size (int, optional): Number of videos to process in each batch.
        """
        data = self.data

        dst_path = os.path.join(self.vis_embeddings_path, self.split)

        os.makedirs(dst_path, exist_ok=True)

        def get_tensors(row):
            video_path = row["fpath"]
            start_frame = row["start_frame"]
            end_frame = row["end_frame"]
            full_path = os.path.join(self.root_dir, video_path)
            return get_frames(full_path, start_frame, end_frame)

        encoder = mvit_v2_s(weights='DEFAULT')

        encoder.head = nn.Identity()

        encoder.to(DEVICE)

        transformation = MViT_V2_S_Weights.KINETICS400_V1.transforms()

        batch = []
        with tqdm(total=len(data), desc="Processing", unit=" samples") as pbar:
            for index, row in data.iterrows():
                input_tensor = get_tensors(row).permute(0, 3, 1, 2)
                input_tensor = convert_image_dtype(input_tensor)
                input_tensor = sample_frames_uniform(input_tensor, 16)
                batch.append(transformation(input_tensor))

                if len(batch) == batch_size:
                    input_batch = torch.stack(batch, dim=0).to(DEVICE)
                    # input_batch = transformation(input_batch)
                    encoded_batch = encoder(input_batch)

                    for i, tensor in enumerate(encoded_batch):
                        sample_index = index - batch_size + i + 1
                        sample_path = os.path.join(dst_path, f'sample_{sample_index}.pt')
                        torch.save(tensor, sample_path)

                    batch = []

                pbar.update(1)

        if batch:
            tensors_batch = torch.stack(batch, dim = 0).to(DEVICE) 
            # tensors_batch = transformation(tensors_batch)
            encoded_batch = encoder(tensors_batch)

            for i, tensor in enumerate(encoded_batch):
                sample_index = len(data) - len(batch) + i + 1
                sample_path = os.path.join(dst_path, f'sample_{sample_index}.pt')
                torch.save(tensor, sample_path)

    def extract_video_feature(self, idx):
        """
        Save tensors for the SMG and IMiGUE datasets per sample.

        This function reads the configuration, groups the data by the 'fpath' column,
        sorts it by the 'start_frame' column, and then saves each video separately
        as a CSV file in the specified destination path.

        Args:
        - idx (int): idx of sample.
        """
        data = self.data

        dst_path = os.path.join(self.vis_embeddings_path, self.split)

        os.makedirs(dst_path, exist_ok=True)

        def get_tensors(row):
            video_path = row["fpath"]
            start_frame = row["start_frame"]
            end_frame = row["end_frame"]
            full_path = os.path.join(self.root_dir, video_path)
            return get_frames(full_path, start_frame, end_frame)

        encoder = mvit_v2_s(weights='DEFAULT')

        encoder.head = nn.Identity()

        encoder.to(DEVICE)

        transformation = MViT_V2_S_Weights.KINETICS400_V1.transforms()

        input_tensor = get_tensors(data.loc[idx]).permute(0, 3, 1, 2)
        input_tensor = convert_image_dtype(input_tensor)
        input_tensor = sample_frames_uniform(input_tensor, 16)

        input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = transformation(input_tensor).to(DEVICE)

        encoded_tensor = encoder(input_tensor)

        sample_path = os.path.join(dst_path, f'sample_{idx}.pt')
        
        encoded_tensor = encoded_tensor.squeeze(0)

        torch.save(encoded_tensor, sample_path)
    
    @abstractmethod
    def set_split(self, split):
        raise NotImplementedError("This method needs to be implemented in subclasses.") 