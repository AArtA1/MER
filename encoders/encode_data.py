from video_encoder import VideoEncoder
from audio_encoder import AudioEncoder
from text_encoder import TextEncoder
from biosensor_encoder import ECGEncoder, EEGEncoder
from pose_encoder import PoseEncoder
def get_encoders(output_dim):
    encoders = {
        "video": VideoEncoder("pytorch/vision:mvit_v2_s", output_dim),
        "audio": AudioEncoder("encoder_models/...", output_dim),
        "text": TextEncoder("encoder_models/...", output_dim),
        "ecg": ECGEncoder("encoder_models/...", output_dim),
        "eeg": EEGEncoder("encoder_models/...", output_dim),
        "pose": PoseEncoder("encoder_models/...", output_dim),
    }
    return encoders