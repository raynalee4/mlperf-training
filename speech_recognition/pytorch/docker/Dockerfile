FROM nvidia/cuda:9.0-cudnn7-runtime-centos7
#FROM nvidia/cuda:9.0-cudnn7-devel-ubi7

WORKDIR /mlperf

ENV CUDA_HOME /usr/local/cuda
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV PATH $PATH:$CUDA_HOME/bin
ENV CUDA_TOOLKIT_ROOT_DIR $CUDA_HOME
ENV LD_LIBRARY_PATH "$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
ENV LIBRARY_PATH $CUDA_HOME/lib64:$LIBRARY_PATH
ENV LD_LIBRARY_PATH $CUDA_HOME/lib64:$LD_LIBRARY_PATH
#ENV CFLAGS "-l$CUDA_HOME/include $CFLAGS"
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

RUN yum install -y centos-release-scl
RUN yum install devtoolset-7 rh-python36 -y

# Generic python installations
# PyTorch Audio for DeepSpeech: https://github.com/SeanNaren/deepspeech.pytorch/releases
# Development environment installations
RUN yum update -y && yum install -y \
  ca-certificates \
  build-essential \
  gcc-c++ \
  sox \
  libsox-dev \
  libsox-fmt-all \
  git \
  cmake \
  tree \
  htop \
  bmon \
  iotop \
  tmux \
  vim


# Unlike apt-get, upgrading pip does not change which package gets installed,
# (since it checks pypi everytime regardless) so it's okay to cache pip.
# Install pytorch
# http://pytorch.org/
RUN source scl_source enable rh-python36 devtoolset-7 && \
    pip install --upgrade pip && \
    pip install h5py \
                hickle \
                matplotlib \
                tqdm==4.19.9 \
                torch==1.0.1 \
                torchvision==0.2.0 \
                audtorcht ad==0.2.0 \
                cffi \
                python-Levenshtein \
                librosa \
                wget \
                tensorboardX \
                warpctc-pytorch10-cuda90==0.1.3

RUN yum update && yum install -y cmake sudo

# install ctcdecode
RUN git clone --recursive https://github.com/parlance/ctcdecode.git
RUN cd ctcdecode && \ 
    source scl_source enable rh-python36 devtoolset-7 && \
    pip install .

ENV SHELL /bin/bash
