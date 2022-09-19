FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

SHELL ["/bin/bash", "-c"]

RUN pip3 install -U pip

RUN pip3 install transformers

COPY . /app/simmc2

WORKDIR /app/simmc2

