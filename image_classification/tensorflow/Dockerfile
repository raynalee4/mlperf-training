FROM nvidia/cuda:10.2-cudnn7-runtime-centos7
#FROM nvidia/cuda:10.2-cudnn7-devel-ubi7


WORKDIR /mlperf

RUN yum update -y

RUN yum update -y && yum install -y \
    ca-certificates \
    build-essential \
    git \
    python \
    python-pip \
    python-devel \
    python3-devel


ENV HOME /mlperf
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN yum install -y python-setuptools

RUN yum install -y python-pip python3-pip virtualenv htop gcc-c++
RUN pip3 install --upgrade numpy scipy sklearn tensorflow-gpu==1.14.0

# Mount data into the docker
ADD . /mlperf

RUN pip3 install -r ./image_classification/tensorflow/requirements.txt

ENTRYPOINT ["./docker_entrypoint.sh"]
