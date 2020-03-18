FROM nvidia/cuda:10.0-cudnn7-runtime-centos7
#FROM nvidia/cuda:10.0-cudnn7-devel-ubi7


WORKDIR /mlperf

RUN yum update -y

RUN yum update -y && yum install -y \
	ca-certificates \
	build-essential \
	git \
	unzip \
	curl

RUN yum install -y centos-release-scl
RUN yum install devtoolset-7 rh-python36 -y

RUN git clone https://github.com/raynalee4/mlperf-training.git && \
	cd mlperf-training && \
	git checkout recommendation

WORKDIR /mlperf/mlperf-training/recommendation/pytorch
RUN cat requirements.txt

COPY requirements.txt /mlperf/mlperf-training/recommendation/pytorch

RUN source scl_source enable rh-python36 devtoolset-7 && \
	pip install --upgrade pip && \
	pip install -r requirements.txt && \
	pip install torch==1.0.1.post2 \ 
				torchvision==0.2.2

RUN source scl_source enable rh-python36 devtoolset-7 && \
	cd negative_sampling_cpp && \
	python setup.py install

WORKDIR /mlperf/mlperf-training/recommendation
RUN source scl_source enable rh-python36 devtoolset-7 && \
	mkdir /data_dir && \
	cp download_dataset.sh verify_dataset.sh /data_dir && \
	cd /data_dir && \
	./download_dataset.sh && \
	./verify_dataset.sh && \
	unzip ml-20m.zip

WORKDIR /mlkperf/data_generation/fractal_graph_expansions
RUN source scl_source enable rh-python36 devtoolset-7 && \
	pip install -r requirements.txt && \
	DATA_DIR=/my_data_dir ./data_gen.sh