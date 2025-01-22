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
import torch 
from torch.utils.data import Dataset 
import cv2 
 
 
class BaseDatasetHandler: 
    def __init__(self, root_folder, split): 
        """ 
        Initialize with the root folder of the dataset. 
 
        Args: 
        - root_folder (str): The main folder containing dataset files. 
        """ 
        self.root_folder = root_folder
        self.split = split
 
    def load_data(self, split='train'): 
        """ 
        Load data based on split (e.g., train/test). 
        Implement in subclasses. 
 
        Args: 
        - split (str): Dataset split to load ('train' or 'test'). 
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



class SMGCustomDataset(Dataset): 
    def __init__(self, root_folder, data, transform): 
        """ 
        Initialize the dataset. 

        Args: 
        - data (list): List of dictionaries with video metadata. 
        - transform: Transformations to apply on the frames. 
        """ 
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

        # Capture video and extract frames 
        cap = cv2.VideoCapture(os.path.join(self.root_folder, video_path)) 
        if not cap.isOpened(): 
            raise RuntimeError(f"Failed to open video: {video_path}") 

        frames = [] 
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) 
        for frame_idx in range(start_frame, end_frame + 1): 
            ret, frame = cap.read() 
            if not ret: 
                break 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB 
            if self.transform: 
                frame = self.transform(frame) 
            frames.append(frame) 

        cap.release() 

        # Stack frames along a new dimension for model input 
        # batch x width x height x depth
        frames = torch.Tensor(np.stack(frames)) if len(frames) > 0 else torch.empty(0) 

        return frames, label 


class SMGDatasetHandler(BaseDatasetHandler): 
    def __init__(self, root_folder, split, type = 'SMG'): 
        """ 
        Initialize with the root folder of the dataset. 
 
        Args: 
        - root_folder (str): The main folder containing dataset files. 
        - split (str): Dataset split to load ('train', 'test', or 'valid'). 
        """ 
        super().__init__(root_folder, split) 
        assert type in ['SMG', 'iMiGUE']
        self.type = type
        if self.type == 'SMG':
            self.csv_folder = os.path.join(root_folder, "smg_csv") 
            self.rgb_folders = {
                "SMG_RGB_Phase1": os.path.join(root_folder, "SMG_RGB_Phase1"), 
                "SMG_RGB_Phase2": os.path.join(root_folder, "SMG_RGB_Phase2"), 
            }
            self.split_file = os.path.join(self.csv_folder, f"smg_{split}.csv")
        
        elif self.type == 'iMiGUE':
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
 
        return SMGCustomDataset(self.root_folder, data, transform) 
    
    def save_tensors(self, dst_path, split = None):    
        
        
    
    def _set_split(self, split):
        if self.type == 'SMG':
            self.split_file = os.path.join(self.csv_folder, f"smg_{split}.csv")
        elif self.type == 'iMiGUE':
            self.split_file = os.path.join(self.csv_folder, f"imigue_{split}.csv")

    def _load_config(self, split=None):
        if split is None: 
            split = self.split
        else:
            self._set_split(self, split) 

        # TODO: add changing split

        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file {self.split_file} does not exist.")

        return pd.read_csv(self.split_file)

        # # Read the CSV file corresponding to the split 
        # data_df = pd.read_csv(self.split_file) 
        # data = [] 
        # for _, row in data_df.iterrows(): 
        #     video_path = row["fullpath"] 
        #     if not os.path.exists(video_path): 
        #         raise FileNotFoundError(f"Video file {video_path} does not exist.") 
        #     data.append({ 
        #         "video_path": video_path, 
        #         "label": row["label_class"], 
        #         "start_frame": row["start_frame"], 
        #         "end_frame": row["end_frame"], 
        #     }) 
 
        # return data 
    

def save_tensors(dataset):
    

 
    # def evaluate(self, model, split=None): 
    #     """ 
    #     Perform evaluation on the dataset split. 
 
    #     Args: 
    #     - model: The model to use for predictions. 
    #     - split (str): Dataset split to evaluate on ('train', 'test', 'valid').
    #     Returns: 
    #     - dict: Computed metrics. 
    #     """ 
    #     if split is None: 
    #         split = self.split 
 
    #     data_loader = self.load_data(split) 
    #     ground_truth, predictions = [], [] 
 
    #     for item in data_loader: 
    #         video_path = item["video_path"] 
    #         label = item["label"] 
    #         start_frame = item["start_frame"] 
    #         end_frame = item["end_frame"] 
 
    #         pred = model.predict(video_path, start_frame, end_frame) 
    #         ground_truth.append(self.convert_labels_to_VAD_vector(label)) 
    #         predictions.append(pred) 
 
    #     return super().evaluate_metrics(ground_truth, predictions) 
     
 
 
 
if __name__ == "__main__": 
    
    # smg_dataset = SMGDatasetHandler('C:/Users/aslan/Documents/MER/SMG', 'train') 
 
    smg = SMGDatasetHandler('C:/Users/aslan/Documents/MER/SMG', 'train')

    print(smg.load_data()[0][0].shape)