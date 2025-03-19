import numpy as np
import pickle
from scipy.spatial.distance import cdist

def translate_VAD(vad_values, direction="to_norm"):
    """
    Translate a list of VAD values between 1-to-9 scale and -1-to-1 scale.

    Parameters:
    vad_values (list): A list of three numbers representing Valence, Arousal, and Dominance.
    direction (str): Direction of translation. "to_normalized" converts 1-to-9 to -1-to-1.
                     "to_original" converts -1-to-1 to 1-to-9.

    Returns:
    list: Translated VAD values.
    """
    if len(vad_values) != 3:
        raise ValueError("Input must be a list of three numbers representing Valence, Arousal, and Dominance.")

    if direction == "to_norm":
        return [(value - 5) / 4 for value in vad_values]
    elif direction == "to_1-9":
        return [round(value * 4 + 5, 2) for value in vad_values]
    else:
        raise ValueError('Invalid direction. Use "to_normalized" or "to_original".')


def find_closest_label(vad_query, vad_dict, boundary=0.1):
    """Find the closest categorical label based on VAD distance, with boundary filtering."""
    vad_values_list = np.array(list(vad_dict.keys()))
    label_list = np.array(list(vad_dict.values()))

    # Calculate distance to all VAD points
    distances = cdist([vad_query], vad_values_list, metric='euclidean')[0]

    # Find points within the boundary
    filtered_indices = np.where(distances <= boundary)[0]

    if len(filtered_indices) == 0:
        return None  # No points within the boundary

    closest_index = filtered_indices[np.argmin(distances[filtered_indices])]
    closest_label = label_list[closest_index]

    return closest_label


def load_vad_dict(filename):
    """Load VAD dictionary from a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
"""
def find_closest_key(target_key, key_list):
   
    target_array = np.array(target_key)
    key_arrays = np.array(key_list)

    distances = np.linalg.norm(key_arrays - target_array, axis=1)
    closest_idx = np.argmin(distances)

    return tuple(key_list[closest_idx])"""


def find_closest_key(target_key, key_list):
    """Find the closest key using Euclidean distance."""
    key_list = list(key_list)  # Convert dict_keys to a list first!

    target_array = np.array(target_key, dtype=np.float32).reshape(-1)  # Ensure it’s a 1D array
    key_arrays = np.array(key_list, dtype=np.float32)  # Now this conversion is safe

    if key_arrays.ndim == 1:  # If only one key exists, reshape it
        key_arrays = key_arrays.reshape(1, -1)

    distances = np.linalg.norm(key_arrays - target_array, axis=1)
    closest_idx = np.argmin(distances)

    return tuple(key_list[closest_idx])  # Return the closest key as a tuple

    return tuple(key_list[closest_idx])  # Return the closest key as a tuple


def get_label(vad_tuple, cat_label_dict):
    """Get label from dictionary, searching for closest if needed."""
    if vad_tuple in cat_label_dict:
        return cat_label_dict[vad_tuple]

    # If key is not found, find the closest one
    closest_key = find_closest_key(vad_tuple, list(cat_label_dict.keys()))
    return cat_label_dict[closest_key]