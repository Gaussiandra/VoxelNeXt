xhost +local:

docker run -it \
    -v /media/itsarin/HDD-2TB3/nuScenes/:/workspace/nuScenes \
    -v /media/itsarin/HDD-2TB3/level5_bags/:/workspace/level5_bags \
    --name voxelnext \
    --runtime=nvidia \
    --gpus all \
    --device=/dev/nvidia-uvm \
    --device=/dev/nvidia-uvm-tools \
    --device=/dev/nvidia-modeset \
    --device=/dev/nvidiactl \
    --device=/dev/nvidia0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    voxelnext-torch2.1.2
