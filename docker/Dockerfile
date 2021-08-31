FROM ubuntu:20.04 

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y git curl gnupg

# Install PyCoral

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update && apt-get install -y gasket-dkms libedgetpu1-std python3-pycoral

RUN apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip setuptools wheel
RUN python3 -m pip install numpy
RUN python3 -m pip install opencv-python-headless
RUN python3 -m pip install tqdm pyyaml json

# Mountpoint for yolo testing
RUN mkdir /yolo
