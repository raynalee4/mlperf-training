FROM nvidia/cuda:8.0-cudnn7-runtime-centos7
#FROM nvidia/cuda:10.2-cudnn7-devel-ubi7

WORKDIR /mlperf

RUN yum update -y && yum install -y \
    ca-certificates \
    build-essential \
    epel-release \
    git \
    python3 \
    python3-devel

RUN pip3 -V

ADD . /mlperf

ENV PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
RUN ln -s /usr/local/cuda-8.0/lib64/libcudnn.so.7 /usr/local/cuda-8.0/lib64/libcudnn.so

RUN pip3 install -r sentiment_analysis/requirements.txt

RUN python3 sentiment_analysis/download.py
RUN python3 sentiment_analysis/verify.py