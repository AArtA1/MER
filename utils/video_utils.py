import ffmpeg
import numpy as np
import torch 
from torch.utils.data import Dataset 
import cv2 


def extract_frames_by_interval(video_path, interval_seconds=1, start_time=0):
    """
    Extract one frame every X seconds from a video as numpy arrays.

    Args:
    - video_path (str): Path to the video file.
    - interval_seconds (int): Number of seconds between each frame.
    - start_time (int): Start extracting frames from this time (in seconds).

    Returns:
    - list: Frames as numpy arrays (H, W, 3).
    """
    probe = ffmpeg.probe(video_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    duration = float(video_info['duration'])

    frames = []
    current_time = start_time
    while current_time < duration:
        out, _ = (
            ffmpeg
            .input(video_path, ss=current_time)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24", vframes=1)
            .run(capture_stdout=True, capture_stderr=True)
        )
        frame = np.frombuffer(out, np.uint8).reshape((height, width, 3))
        frames.append(frame)
        current_time += interval_seconds

    return frames


def get_frames(path_to_video, start_frame, end_frame, transform = None):
    cap = cv2.VideoCapture(path_to_video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path_to_video}")

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if transform:
            frame = transform(frame)
        frames.append(frame)

    cap.release()

    return torch.tensor(np.stack(frames), dtype = torch.uint8) if len(frames) > 0 else torch.empty(0)


def sample_frames_uniform(video_tensor: torch.Tensor, num_frames: int = 16) -> torch.Tensor:
    """
    Uniformly sample `num_frames` frames from a 4D video tensor of shape (T, C, H, W).

    Args:
        video_tensor (torch.Tensor): Video data of shape (T, C, H, W), where T is
            the total number of frames.
        num_frames (int): The desired number of frames in the output.

    Returns:
        torch.Tensor: A 4D tensor of sampled frames with shape (num_frames, C, H, W).
    """
    T = video_tensor.shape[0]
    if T == 0:
        raise ValueError("Empty video: cannot sample from zero frames.")
    
    # Create num_frames evenly spaced points between 0 and T-1
    indices = torch.linspace(0, T - 1, steps=num_frames)
    indices = torch.round(indices).long()  # Convert to integer indices

    # Clamp indices in case rounding goes slightly out of bounds
    indices = torch.clamp(indices, 0, T - 1)

    # Use indexing to select frames
    sampled_frames = video_tensor[indices]

    return sampled_frames

