# install MMEngine, MMCV and MMDetection using MIM
%pip install -U openmim
!mim install mmengine
!mim install "mmcv==2.1.0"
!mim install "mmpose"
!mim install "mmdet"

# Install mmaction2
!rm -rf mmaction2
!git clone https://github.com/open-mmlab/mmaction2.git -b main
%cd mmaction2

!pip install -e .