import re
import os
import numpy as np
import torch
import pickle
from transformers import RobertaTokenizer, RobertaModel

# Initialize the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')

def parse_transcription_file(file_path):
    """
    Parse the transcription file and extract timestamps and speech segments.

    Args:
    - file_path (str): Path to the transcription text file.

    Returns:
    - List[dict]: List of dictionaries containing speaker, start_time, end_time, and text.
    """
    utterances = []

    # Regular expression pattern to extract speaker, timestamps, and text
    pattern = r'(\S+)\s+\[([\d.]+)-([\d.]+)\]:\s+(.*)'

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(pattern, line.strip())
            if match:
                speaker = match.group(1)
                start_time = float(match.group(2))
                end_time = float(match.group(3))
                text = match.group(4).strip()
                # Create a dictionary and add it to the list
                utterances.append({
                    'speaker': speaker,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text,
                    'feature_vector': None  # Placeholder for feature vector
                })

    return utterances

def extract_features(utterances, model, tokenizer):
    """
    Extract features for each utterance while keeping all metadata.

    Args:
    - utterances (list of dict): List of dictionaries containing speaker, timestamps, and text.
    - model (transformer model): The pre-trained model for feature extraction.
    - tokenizer (transformer tokenizer): The tokenizer associated with the model.

    Returns:
    - List of dict: Original dictionaries enriched with feature_vector.
    """
    for utterance in utterances:
        # Tokenize the text from the dictionary
        encoded_input = tokenizer(
            utterance['text'],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Extract features with no gradient computation
        with torch.no_grad():
            output = model(**encoded_input)

        # Extract the [CLS] token feature
        cls_embedding = output.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        # Add the feature vector to the dictionary
        utterance['feature_vector'] = cls_embedding.tolist()  # Convert numpy array to list

    return utterances

def process_all_files(root_folder):
    all_features = {}

    # Loop through all session folders
    for session_folder in os.listdir(root_folder):
        session_folder_path = os.path.join(root_folder, session_folder)

        # Only process directories that are session folders
        if os.path.isdir(session_folder_path) and session_folder.startswith('Session'):
            all_features[session_folder] = {}  # Initialize dict for the session
            transcriptions_folder = os.path.join(session_folder_path, 'dialog', 'transcriptions')

            if os.path.isdir(transcriptions_folder):
                for txt_file in os.listdir(transcriptions_folder):
                    # Exclude Mac system files (those starting with '._')
                    if txt_file.endswith('.txt') and not txt_file.startswith('._'):
                        file_path = os.path.join(transcriptions_folder, txt_file)

                        # Step 1: Parse the transcription file and extract speaker, timestamps & text
                        utterances = parse_transcription_file(file_path)

                        # Step 2: Extract features and save
                        features = extract_features(utterances, roberta_model, tokenizer)

                        # Append results to the session > file dictionary
                        all_features[session_folder][txt_file] = features

    return all_features

def save_features_to_pickle(hierarchical_features, output_file):
    """
    Save the hierarchical structure of features to a pickle file.

    Args:
    - hierarchical_features (dict): Nested dict with sessions, filenames, and utterances.
    - output_file (str): Path to save the pickle file.
    """
    with open(output_file, 'wb') as f:
        pickle.dump(hierarchical_features, f)
    print(f"Features saved to {output_file}")

# Example usage
root_folder = r"D:\IEMOCAP"
all_features_hierarchy = process_all_files(root_folder)
save_features_to_pickle(all_features_hierarchy, 'iemocap_text_features_roberta.pkl')
