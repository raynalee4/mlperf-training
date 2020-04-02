#FROM docker-registry.default.svc:5000/mlperf/rnn-translator@sha256:21140c744b3b2e3f36db92b2dcd970b539b9887b7fa46600cc3cfefec3ec2fd5
FROM nvidia/cuda:10.0-cudnn7-runtime-centos7
#FROM nvidia/cuda:9.0-cudnn7-devel-ubi7

WORKDIR /mlperf

#WORKDIR /mlperf/rnn_translator/pytorch
#RUN curl -O https://raw.githubusercontent.com/raynalee4/mlperf-training/rnn-translator/rnn_translator/pytorch/run.sh
#RUN cat run.sh

ENV LANGUAGE en_US:en
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8
RUN ln -s /usr/local/cuda-10.0/lib64/libcudnn.so.7 /usr/local/cuda-10.0/lib64/libcudnn.so

ENV CUDA_HOME /usr/local/cuda 
ENV PATH /usr/local/cuda-10.0/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH /usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

RUN yum update -y && \
    yum install -y centos-release-scl && \
    yum install devtoolset-7 rh-python36 -y && \
    yum install -y \
    ca-certificates \
    build-essential \
    git \
    curl \
    wget \
    libjpeg-dev \
    libpng-dev \
    epel-release \
    python3 \
    python3-devel

ADD . /mlperf
RUN mkdir /results && \
    mkdir /results/gmnt

RUN source scl_source enable rh-python36 devtoolset-7 && \
    pip install --upgrade pip && \
    pip install -r rnn_translator/requirements.txt && \
    pip install -r rnn_translator/pytorch/requirements.txt
	#git clone https://www.github.com/nvidia/apex && \
	#cd apex && \
	#python setup.py install

RUN chmod +x rnn_translator/download_dataset.sh rnn_translator/verify_dataset.sh