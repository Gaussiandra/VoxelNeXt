# Based on https://github.com/VISION-SJTU/PillarNet-LTS/blob/2cfa24537cd8974a579a39f57426957bb9d8b134/docker/Dockerfile
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

ENV DEBIAN_FRONTEND="noninteractive"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install basics
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 \
    libsm6 libxrender-dev libxext6 nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/cuda-12.1/bin:$PATH"
ENV CUDA_PATH="/usr/local/cuda-12.1"
ENV CUDA_HOME="/usr/local/cuda-12.1"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"

# Python packages for VoxelNeXt setup
RUN pip install --no-cache-dir spconv-cu120==2.3.6 scikit-image==0.19.3 numba==0.59.1

# VoxelNeXt setup (it takes a long time)
RUN git clone https://github.com/Gaussiandra/VoxelNeXt.git -b ml3d_viz
WORKDIR /workspace/VoxelNeXt
RUN python setup.py develop

# Other python packages
RUN pip install --no-cache-dir gdown matplotlib opencv_python nuscenes_devkit gpustat
RUN pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# Download checkpoint from google drive
RUN gdown 1IV7e7G9X-61KXSjMGtQo579pzDNbhwvf -O cbgs_voxel0075_voxelnext.pth

# Install ROS noetic, based on https://wiki.ros.org/noetic/Installation/Ubuntu
RUN apt update \
    && apt install -y curl lsb-release \
    && sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && apt update \
    && apt install -y ros-noetic-desktop-full tmux
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

RUN apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential \
    && rosdep init \
    && rosdep update
    