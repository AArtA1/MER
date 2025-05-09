import os.path as osp
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from mmengine.utils import track_iter_progress

from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample


def extract_pose_from_tensor(
        frames_tensor: torch.Tensor,      # (T, H, W, 3)  либо (T, 3, H, W)
        inferencer,
        device: torch.device = torch.device("cpu"),
        rgb_last: bool = True,            # True, если формат (H, W, 3). False → (3, H, W)
        transform=None                    # опц. аугментации/препроцессинг, если нужны
):
    """
    Возвращает:
        keypoints_seq : np.ndarray  shape (T, 17, 2)
        scores_seq    : np.ndarray  shape (T, 17)
    """
    # приводим тип данных и устройство
    frames_tensor = frames_tensor.to(device, non_blocking=True)

    keypoints_list, scores_list = [], []

    T = frames_tensor.shape[0]
    for t in range(T):
        frame_t = frames_tensor[t]

        # Приводим к формате (H, W, 3) uint8 numpy, т.к. большинство
        # инференсеров ожидают именно его.
        if not rgb_last:  # если (3, H, W) → (H, W, 3)
            frame_t = frame_t.permute(1, 2, 0).contiguous()

        frame_np = frame_t.cpu().numpy()

        if transform is not None:
            frame_np = transform(frame_np)

        result_gen = inferencer(frame_np)   # ← ваш треккинг-инференсер
        result = next(result_gen)

        if result["predictions"]:
            pred = result["predictions"][0][0]
            # print(pred)
            keypoints = np.array(pred["keypoints"]).astype(np.float32)       # (17, 2)
            scores = np.array(pred["keypoint_scores"]).astype(np.float32)    # (17,)
        else:
            keypoints = np.zeros((17, 2), dtype=np.float32)
            scores   = np.zeros((17,),    dtype=np.float32)

        keypoints_list.append(keypoints)
        scores_list.append(scores)

    keypoints_seq = np.stack(keypoints_list)      # (T, 17, 2)
    scores_seq   = np.stack(scores_list)          # (T, 17)

    return keypoints_seq, scores_seq


def extract_skeleton_embeddings(model: nn.Module,
                                keypoints_seq: np.array,
                                scores_seq: np.array,
                               test_pipeline = None):

    keypoints = np.stack(keypoints_seq, axis=0)  # (T, 17, 2)
    scores = np.stack(scores_seq, axis=0)        # (T, 17)
    total_frames = keypoints.shape[0]

    # === ВАЖНО: входной словарь для MMACTION ===
    skeleton = {
        'keypoint': keypoints[np.newaxis, ...].astype(np.float32),       # (1, T, 17, 2)
        'keypoint_score': scores[np.newaxis, ...].astype(np.float32),    # (1, T, 17)
        'total_frames': total_frames
    }

    
    if test_pipeline is None:
        cfg = model.cfg
        init_default_scope(cfg.get('default_scope', 'mmaction'))
        test_pipeline_cfg = cfg.test_pipeline
        test_pipeline = Compose(test_pipeline_cfg)

    data = skeleton

    data = test_pipeline(data)
    data = pseudo_collate([data])

    data = model.data_preprocessor(data, False)

    return model.extract_feat(**data, stage = 'head')[0]