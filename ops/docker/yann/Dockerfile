ARG TORCH_VERSION="1.5.1"
ARG CUDA_VERSION="cuda10.1-cudnn7"
ARG TYPE="runtime


FROM pytorch/pytorch:${TORCH_VERSION}-${CUDA_VERSION}-${TYPE}

ARG VERSION
RUN if [ -n "$VERSION" ] ; then pip install yann ; else pip install yann==${VERSION} ; fi