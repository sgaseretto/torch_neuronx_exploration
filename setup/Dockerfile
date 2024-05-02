ARG NEURON_VER=2.18.2
ARG PYTORCH_VER=2.1.2
# pytorch 1.13.1
#FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:1.13.1-neuronx-py310-sdk2.14.1-ubuntu20.04
# pytorch 2.1.2
# FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:2.1.2-neuronx-py310-sdk2.18.2-ubuntu20.04
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:${PYTORCH_VER}-neuronx-py310-sdk${NEURON_VER}-ubuntu20.04
# Configure tensorboard for trn1
RUN python3 -m pip install --upgrade tensorboard==2.15.1
RUN python3 -m pip config set global.extra-index-url "https://pip.repos.neuron.amazonaws.com"
RUN python3 -m pip install tensorboard-plugin-neuronx neuronx_distributed torchvision

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

# RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components 1
# # Set PATH so micromamba is found
# ENV PATH="/usr/local/bin:${PATH}"
# COPY conda.yaml conda.yaml
# COPY requirements.txt requirements.txt
# # Use micromamba to install packages from the conda.yml file into the base environment
# RUN micromamba install -y -n base -f conda.yaml && \
# micromamba clean --all --yes
# ENV PATH="/root/micromamba/bin:${PATH}"
# RUN mkdir -p /opt/ml
WORKDIR /opt/app/traiunium