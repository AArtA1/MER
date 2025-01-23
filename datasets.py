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
from utils import CustomDataset, get_frames

from abc import ABC, abstractmethod
 
class BaseDatasetHandler(ABC): 
    def __init__(self, root_folder, split): 
        """ 
        Initialize with the root folder of the dataset. 
 
        Args: 
        - root_folder (str): The main folder containing dataset files. 
        """ 
        self.root_folder = root_folder
        self.split = split
        self.split_file = None
    
    @abstractmethod
    def load_data(self, split='train'): 
        """ 
        Load data based on split (e.g., train/test). 
        Implement in subclasses. 
 
        Args: 
        - split (str): Dataset split to load ('train' or 'test'). 
        """ 
        raise NotImplementedError("This method needs to be implemented in subclasses.") 

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
    
    def _load_config(self, split=None):
        if split is not None: 
            self._set_split(split) 
        
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file {self.split_file} does not exist.")

        return pd.read_csv(self.split_file)
    
    @abstractmethod
    def _set_split(self, split):
        raise NotImplementedError("This method needs to be implemented in subclasses.") 


class SMGDatasetHandler(BaseDatasetHandler): 
    def __init__(self, root_folder, split, type = 'SMG'): 
        """ 
        Initialize with the root folder of the dataset. 
 
        Args: 
        - root_folder (str): The main folder containing dataset files. 
        - split (str): Dataset split to load ('train', 'test', or 'valid'). 
        """ 
        super().__init__(root_folder, split) 
        self.csv_folder = os.path.join(root_folder, "smg_csv") 
        self.rgb_folders = {
            "SMG_RGB_Phase1": os.path.join(root_folder, "SMG_RGB_Phase1"), 
            "SMG_RGB_Phase2": os.path.join(root_folder, "SMG_RGB_Phase2"), 
        }
        self.split_file = os.path.join(self.csv_folder, f"smg_{split}.csv")
 
    def convert_labels_to_VAD_vector(self, label): 
        """ 
        Converts the dataset's emotion labels into a vector 
        in the 3D Valence-Arousal-Dominance cube. 
 
        Args: 
        - label (str): The original emotion label from the dataset. 
 
        Returns: 
        - tuple: A 3D vector (Valence, Arousal, Dominance). 
        """ 
        label_to_vad = { 
            "happy": (1.0, 0.8, 0.6), 
            "sad": (-0.8, -0.7, -0.5), 
            "angry": (-0.6, 0.9, 0.8), 
            "neutral": (0.0, 0.0, 0.0), 
        } 
        return label_to_vad.get(label.lower(), (0.0, 0.0, 0.0)) 
 
    def load_data(self, split=None, transform=None): 
        """ 
        Create a PyTorch dataset to load video parts from start_frame to end_frame. 
 
        Args: 
        - split (str): Dataset split ('train', 'test', or 'valid'). 
        - transform: Optional transformations for the video frames. 
 
        Returns: 
        - Dataset: A PyTorch Dataset instance. 
        """ 
        data = self._load_config(split) 
 
        return CustomDataset(self.root_folder, data, transform) 
    
    def save_tensors(self, dst_path, split = None): 
        """
        Save tensors for the SMG dataset.

        This function reads the configuration, groups the data by the 'fpath' column,
        sorts it by the 'start_frame' column, and then saves each video separately
        as a CSV file in the specified destination path.

        Args:
        - dst_path (str): The destination path where the CSV files will be saved.
        - split (str, optional): Dataset split to load ('train', 'test', or 'valid').
        """   
        data = self._load_config(split)

        grouped_data = data.groupby('fpath')#.apply(lambda x: x.sort_values('start_frame'))

        group_dfs = {group_name: group_data for group_name, group_data in grouped_data}

        for name, group in group_dfs.items():
            # def get_tensors(row):
            #         video_path = row["fpath"]
            #         start_frame = row["start_frame"]
            #         end_frame = row["end_frame"]
            #         full_path = os.path.join(self.root_folder, video_path)

            #         return get_frames(full_path, start_frame, end_frame)
            
            # group = group.iloc[:3]

            # group['features'] = group.apply(get_tensors, axis = 1)            
            
            file_name = name.split('/')[-2]

            group.sort_values(by = 'start_frame')
            group.to_csv(os.path.join(dst_path, self.split, file_name + '.csv'), index = False)

        return grouped_data    
    
    def _set_split(self, split):
        assert split in ['train', 'test', 'valid']
        self.split = split
        self.split_file = os.path.join(self.csv_folder, f"smg_{split}.csv")       


class iMiGUEDatasetHandler(BaseDatasetHandler): 
    def __init__(self, root_folder, split, type = 'SMG'): 
        """ 
        Initialize with the root folder of the dataset. 
 
        Args: 
        - root_folder (str): The main folder containing dataset files. 
        - split (str): Dataset split to load ('train', 'test', or 'valid'). 
        """ 
        super().__init__(root_folder, split) 
        self.csv_folder = os.path.join(root_folder, "imigue_csv") 
        self.rgb_folders = {
            "iMiGUE_RGB_Phase1": os.path.join(root_folder, "iMiGUE_RGB_Phase1"), 
            "iMiGUE_RGB_Phase2": os.path.join(root_folder, "iMiGUE_RGB_Phase2"), 
        }
        self.split_file = os.path.join(self.csv_folder, f"imigue_{split}.csv")
 
    def convert_labels_to_VAD_vector(self, label): 
        """ 
        Converts the dataset's emotion labels into a vector 
        in the 3D Valence-Arousal-Dominance cube. 
 
        Args: 
        - label (str): The original emotion label from the dataset. 
 
        Returns: 
        - tuple: A 3D vector (Valence, Arousal, Dominance). 
        """ 
        label_to_vad = { 
            "happy": (1.0, 0.8, 0.6), 
            "sad": (-0.8, -0.7, -0.5), 
            "angry": (-0.6, 0.9, 0.8), 
            "neutral": (0.0, 0.0, 0.0), 
        } 
        return label_to_vad.get(label.lower(), (0.0, 0.0, 0.0)) 
 
    def load_data(self, split=None, transform=None): 
        """ 
        Create a PyTorch dataset to load video parts from start_frame to end_frame. 
 
        Args: 
        - split (str): Dataset split ('train', 'test', or 'valid'). 
        - transform: Optional transformations for the video frames. 
 
        Returns: 
        - Dataset: A PyTorch Dataset instance. 
        """ 

        data = self._load_config(split) 
 
        return CustomDataset(self.root_folder, data, transform) 
    
    def save_tensors(self, dst_path, split = None):    
        """
        Save tensors for the SMG dataset.

        This function reads the configuration, groups the data by the 'fpath' column,
        sorts it by the 'start_frame' column, and then saves each video separately
        as a CSV file in the specified destination path.

        Args:
        - dst_path (str): The destination path where the CSV files will be saved.
        - split (str, optional): Dataset split to load ('train', 'test', or 'valid').
        """
        data = self._load_config(split)

        grouped_data = data.groupby('fpath')#.apply(lambda x: x.sort_values('start_frame'))

        group_dfs = {group_name: group_data for group_name, group_data in grouped_data}

        for name, group in group_dfs.items():
            # def get_tensors(row):
            #         video_path = row["fpath"]
            #         start_frame = row["start_frame"]
            #         end_frame = row["end_frame"]
            #         full_path = os.path.join(self.root_folder, video_path)

            #         return get_frames(full_path, start_frame, end_frame)
            
            # group = group.iloc[:3]

            # group['features'] = group.apply(get_tensors, axis = 1)            
            
            file_name = name.split('/')[-2]

            group.sort_values(by = 'start_frame')
            group.to_csv(os.path.join(dst_path, self.split, file_name + '.csv'), index = False)
    
    def _set_split(self, split):
        assert split in ['train', 'test', 'valid']
        self.split = split
        self.split_file = os.path.join(self.csv_folder, f"imigue_{split}.csv")     
 
 
 
if __name__ == "__main__": 
    
    # smg_dataset = SMGDatasetHandler('C:/Users/aslan/Documents/MER/SMG', 'train') 
 
    # smg = SMGDatasetHandler('C:/Users/aslan/Documents/MER/SMG', 'train')

    # #print(smg.load_data()[0][0]) 

    # smg.save_tensors('smg_sorted')

    # print(smg.save_tensors('smg_sorted'))

    imigue = iMiGUEDatasetHandler('C:/Users/aslan/Documents/MER/iMiGUE', 'train')

    dataset = imigue.load_data('valid')

    print(dataset[0][0].max())