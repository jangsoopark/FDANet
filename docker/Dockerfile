
# docker build . -t change-detection
# Base container: docker pull pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt update

RUN pip install seaborn && \
    pip install numpy && \
    pip install scipy&& \
    pip install tqdm && \
    pip install jupyter && \
    pip install matplotlib && \
    pip install scikit-image && \
    pip install scikit-learn && \
    pip install opencv_python && \
    pip install future && \
    pip install h5py && \ 
    pip install tifffile && \
    pip install rasterio


RUN apt update && \
    apt install -y wget vim emacs nano libgl1-mesa-glx


RUN mkdir -p /workspace

ARG work_dir=/workspace

WORKDIR ${work_dir}
