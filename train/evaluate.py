import os
import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
from emotionbind.datasets.dataset_iemocap import IEMOCAPDatasetHandler
from emotionbind.models.emotionbind_model import EmotionBindModel
from emotionbind.utils.vad_utils import find_closest_label, load_vad_dict
from emotionbind.train.train import get_dataset_handler, concordance_correlation_coefficient

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "./checkpoints"

DATASET_MODALITIES = {
    "iemocap": ["video", "audio", "text"],
    "smg" : ["video", "skeleton"],
    "imigue" : ["video", "skeleton"]
}

def PCC(a: torch.tensor, b: torch.tensor):
    am = torch.mean(a, dim=0)
    bm = torch.mean(b, dim=0)
    num = torch.sum((a - am) * (b - bm), dim=0)
    den = torch.sqrt(sum((a - am) ** 2) * sum((b - bm) ** 2)) + 1e-5
    return num/den

def CCC(a: torch.tensor, b: torch.tensor):
    rho = 2 * PCC(a,b) * a.std(dim=0, unbiased=False) * b.std(dim=0, unbiased=False)
    rho /= (a.var(dim=0, unbiased=False) + b.var(dim=0, unbiased=False) + torch.pow(a.mean(dim=0) - b.mean(dim=0), 2) + 1e-5)
    return rho

def pearson_correlation_coefficient(y_true, y_pred):
    return np.array([np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1] for i in range(y_true.shape[1])])

def evaluate_model(test_loader, checkpoint_path, checkpoint_name):
    model = EmotionBindModel(dataset_name=args.dataset_name)
    model.to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Evaluating model from {checkpoint_path}")

    results_dict = {
        "vad_predictions": [],
        "vad_ground_truth": [],
        "categorical_ground_truth": []
    }

    ground_truth, predictions = [], []
    categorical_ground_truth, categorical_predictions = [], []

    # vad_dict = load_vad_dict('../utils/sorted_vad_labels.pkl')

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Model"):
            modalities = DATASET_MODALITIES.get(args.dataset_name, [])
            inputs = {}
            for key in ("video", "skeleton"):
                if key in batch:
                    tgt = "vision" if key == "video" else key
                    tgt = "pose" if tgt == "skeleton" else tgt
                    inputs[tgt] = batch[key].to(DEVICE)

            vad_labels = batch["label"].to(DEVICE)
            categorical_labels = batch["categorical_label"]

            _, vad_predictions = model(inputs)

            #categorical_pred = [str(find_closest_label(v.cpu().numpy(), vad_dict, 1.0)) for v in vad_predictions]

            print("Vad labels: " + str(vad_labels))
            print("Predictions: " + str(vad_predictions))

            #predictions.append(vad_predictions.cpu().numpy())
            #ground_truth.append(vad_labels.cpu().numpy())

            predictions.append(vad_predictions.cpu())
            ground_truth.append(vad_labels.cpu())

            categorical_ground_truth.extend(categorical_labels)
            #categorical_predictions.extend(categorical_pred)

    #ground_truth = np.vstack(ground_truth)
    #predictions = np.vstack(predictions)

    results_dict["vad_predictions"] = predictions
    results_dict["vad_ground_truth"] = ground_truth
    results_dict["categorical_ground_truth"] = categorical_ground_truth

    filename = f"evaluation_results_{checkpoint_name}.pkl"

    with open(filename, "wb") as f:
        pickle.dump(results_dict, f)

    ground_truth_tensor = torch.cat(ground_truth, dim=0).to(dtype=torch.float32, device=DEVICE)
    predictions_tensor = torch.cat(predictions, dim=0).to(dtype=torch.float32, device=DEVICE)

    mse = torch.nn.functional.mse_loss(predictions_tensor, ground_truth_tensor).item()
    mae = torch.nn.functional.l1_loss(predictions_tensor, ground_truth_tensor).item()
    ss_total = ((ground_truth_tensor - ground_truth_tensor.mean()) ** 2).sum()
    ss_residual = ((ground_truth_tensor - predictions_tensor) ** 2).sum()
    r2 = (1 - ss_residual / ss_total).item()

    print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")

    #mse = mean_squared_error(ground_truth, predictions)
    #mae = mean_absolute_error(ground_truth, predictions)
    #r2 = r2_score(ground_truth, predictions)

    #ccc = concordance_correlation_coefficient(ground_truth, predictions)
    #pcc = pearson_correlation_coefficient(ground_truth, predictions)

    #ccc2 = CCC(ground_truth_tensor, predictions_tensor)
    #pcc2 = PCC(ground_truth_tensor, predictions_tensor)

    #print(f"CCC (Arousal): {ccc[0]:.6f}, CCC (Valence): {ccc[1]:.6f}, CCC (Dominance): {ccc[2]:.6f}")
    #print(f"PCC (Arousal): {pcc[0]:.6f}, PCC (Valence): {pcc[1]:.6f}, PCC (Dominance): {pcc[2]:.6f}")

    #print(f"CCC (Arousal): {ccc2[0]:.6f}, CCC (Valence): {ccc2[1]:.6f}, CCC (Dominance): {ccc2[2]:.6f}")
    #print(f"PCC (Arousal): {pcc2[0]:.6f}, PCC (Valence): {pcc2[1]:.6f}, PCC (Dominance): {pcc2[2]:.6f}")

    #f1 = f1_score(categorical_ground_truth, categorical_predictions, average="weighted")
    #print(f"F1-score: {f1:.6f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dataset_dir", type=str, default="D:/SMG", help="Path to training dataset directory")
    parser.add_argument("--text_feature_dir", type=str, default="D:/SMG/text_features",
                        help="Path to text feature directory")
    parser.add_argument("--video_feature_dir", type=str, default="D:/SMG/mvit_v2_scene_features",
                        help="Path to video feature directory")
    parser.add_argument("--audio_feature_dir", type=str, default="D:/SMG/wav2vec2_audio_features",
                        help="Path to audio feature directory")
    parser.add_argument("--faces_feature_dir", type=str, default="D:/SMG/face_features",
                        help="Path to faces video feature directory")
    parser.add_argument("--skeleton_feature_dir", type=str, default="D:/SMG/skeleton_features",
                        help="Path to ecg video feature directory")
    parser.add_argument("--dataset_name", type=str, choices=['smg', 'imigue', 'iemocap'], help="Name of the dataset")
    parser.add_argument("--num_processes", type=int, default=1, help="Number of parallel training processes")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--multi_mode", type=bool, default=False, help="Whether to enable multi-dataset training")
    args = parser.parse_args()

    feature_dirs = {}
    if args.video_feature_dir:
        feature_dirs["video"] = args.video_feature_dir
    if args.audio_feature_dir:
        feature_dirs["audio"] = args.audio_feature_dir
    if args.text_feature_dir:
        feature_dirs["text"] = args.text_feature_dir
    if args.faces_feature_dir:
        feature_dirs["faces"] = args.faces_feature_dir
    if args.skeleton_feature_dir:
        feature_dirs["skeleton"] = args.skeleton_feature_dir

    if not os.path.exists(args.root_dataset_dir):
        raise FileNotFoundError(f"The dataset directory does not exist: {args.root_dataset_dir}")

    split = "test"
    test_dataset = get_dataset_handler(args.dataset_name, args.root_dataset_dir, feature_dirs, split)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if not os.path.exists(CHECKPOINT_DIR) or not os.listdir(CHECKPOINT_DIR):
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")

    checkpoint = ("best_checkpoint_proc2.pth")
    checkpoint_name = os.path.splitext(checkpoint)[0]

    evaluate_model(test_loader, os.path.join(CHECKPOINT_DIR, checkpoint), checkpoint_name)