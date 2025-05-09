# Multimodal Emotion Recognition

---

## 1. Quick Start (recommended)

> **Requirements:** Conda ≥ 23.1, CUDA 11.8 / 12.1, Git.

### 1.1 Create and activate a Conda environment

```bash
# create the environment
conda create -y -n emotionbind python=3.10
conda activate emotionbind
```


### 1.2 Install PyTorch 2.4 + CUDA

Pick the wheel matching your GPU / CUDA toolkit (examples below):

```bash
# CUDA 12.1
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
# pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CPU-only
# pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```


### 1.3 Install project dependencies

```bash
pip install -r requirements.txt

# for skeletons extracting
source bash.sh
```


## 2. Datasets

Supported datasets: SMG, IMIGUE
Below is the SMG example; adjust paths for other datasets.

### 2.1 Expected folder layout (SMG)

SMG/

├── video_features/        # video


└── skeleton_features/       # pose

### 2.2 Sources

All sources are available by the link below:

[label]: 

### 3. Training

## 3.1 Basic command

```bash
export PYTHONPATH=$(pwd)

python train.py \
  --dataset_name SMG \
  --root_dataset_dir /space/emotion_data/SMG/ \
  --text_feature_dir /space/emotion_data/SMG/text_features \
  --video_feature_dir /space/emotion_data/SMG/mvit_v2_scene_features \
  --audio_feature_dir /space/emotion_data/SMG/wav2vec2_audio_features \
  --num_processes 4 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --epochs 10
```

Checkpoints and TensorBoard logs are saved to ./logs by default.

### 3.2 Monitoring with TensorBoard

```bash
# install if missing
pip install tensorboard

# launch the server (default port 6006)
tensorboard --logdir logs
```

Open http://localhost:6006 in your browser to inspect losses and metrics.


## 4. Training Configuration

Flag	Default	Description
--batch_size	8	Batch size
--learning_rate	1e-4	Learning rate
--epochs	10	Number of epochs
--num_processes	2	CPU workers for preprocessing
--device	auto	CUDA if available, otherwise CPU

Run python train.py --help for the complete list.


## 5. Requirements (concise)
	•	Python ≥ 3.8 (tested on 3.10)
	•	PyTorch 2.4.0
	•	tqdm, argparse, pickle
	•	Everything listed in requirements.txt


