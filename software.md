[TOC]

## Software Setup

Here are some instructions/suggestions on how to install various packages you need for running inference and development work on your device. I assume familiarity with git and the command line generally.

### System organisation

Personally, I like to keep all my repositories in a code folder, e.g. `/home/josh/code` but you can do as you will.

### Useful or necessary packages

You will absolutely need git. `sudo apt install git`

I suggest installing **htop**, a nice graphical resource monitor as well as **tmux** which is a bit like screen, but nicer to use. Tmux lets you spawn persistent terminal sessions that won't die if you disconnect or close the connection to the system - great for long-running tasks.

`sudo apt install htop tmux`

### Virtual Environments (venvs)

Different runtime environments have widely differing library requirements. As such, it's strongly recommended that you use virtual environments for everything. If you've not used `venv` before then don't worry, it's as simple as:

```
python3 -m venv <name>
```

which will create a folder in the current directory containing a full and clean Python install. You can then use:

```
source <name>/bin/activate
```

and your terminal should show a little `(<name>)` before the prompt indicating that you're now in the environment. You can double check what you're using by typing `which python` - note that using a virtual environment is nice because you can drop the `python3` or `python3.8` or whatever.

(You can install Python 3.8 on the Nano using apt: `sudo apt install python3.8-dev python3.8-venv`, but I wouldn't recommend it because everything is set up already for 3.6)

Go ahead and makeyour environment:

```
python3.6 -m venv py36
```

Activate the environment and run:

```
pip install --upgrade pip wheel setuptools
```

because whatever versions of those you have will probably be hopelessly outdated.

It's up to you what you call your venvs, for example you may need a more specific environment for some library, for example Tensorflow. Either way it's cleaner than installing everything into your system python.

### Requirements for different inference runtimes

Remember that the Nano Jetson and the Coral are primarily **inference devices**. They're not designed for training and exporting models. Therefore we're going to focus on the libraries required to **perform detection**. You should also have access to a desktop machine to do some of the export work. Linux is preferable, but you can also use Windows and install Ubuntu from the app store, which is really useful for certain things like the EdgeTPU compiler.

**EdgeTPU** PyCoral does the heavy lifting here, since the actual inference work is done on-device. So you don't actually need Tensorflow to work with it, only the tensorflow-lite runtime. As you can see below, it's as simple as an `apt-get` command. On the Coral Board, it's already installed for you.

**TensorRT** The easiest way to use TensorRT is to have it sit on top of PyTorch. Then, when you load a model that's been converted to TensorRT, you just use all the torch-like commands (except your model will be a `TRTModule`). NVIDIA IOT provides a nice package to convert existing PyTorch models, although it does have a tendency to crash and not tell you why.

**Torch** Obviously if you want to run a model in PyTorch with CUDA enabled, then you need PyTorch... The simplest way is to use NVIDIA's pre-packaged libraries. On the Coral you'll probably have to build from scratch.

### Docker

If you want to use the Dockerfile provided, then you need to install Docker...!

#### Installing Docker
You can install Docker easily enough on either system, following the instructions on Docker's [doc pages](https://docs.docker.com/engine/install/ubuntu/). The guide is roughly copied here for reference, but you should really follow the official wording in case you get stuck.

You can use the quick-install script:

```
curl -fsSL https://get.docker.com -o get-docker.sh
sh ./get-docker.sh
```

Or if that doesn't work, install the necessary pre-requisites:

```
sudo apt-get update

sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
```

Add the Docker GPG key:

```
 curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
```

And use the arm64 repository:

```
echo \
  "deb [arch=arm64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

Then install Docker itself

```
 sudo apt-get update

 sudo apt-get install docker-ce docker-ce-cli containerd.io
```
#### Building the test container

Simple:

```
docker build . -t edgetpu
```

Then run as:

```
docker --rm -it -v /path/to/this/repo:/yolo --privileged edgetpu bash
```

The options are `--rm` to delete containers once you exit, `-it` for interactive mode and `-v` to mount this repository as a volume inside the container. `--privileged` is required so you can access hardware.

This should put you in a bash prompt and then you can `cd` to `/yolo` and run the test code as you wish.

## Library Dependencies

### EdgeTPU: Coral library installation and testing

Instructions from Google are [here](https://coral.ai/docs/edgetpu/tflite-python/#overview)

You can install `pycoral` which should also install `tensorflow-lite`:

```bash
sudo apt-get install python3-pycoral
```

#### Virtual env install

Inside your virtual environment, find the URL of the latest `aarch64` release from: https://github.com/google-coral/pycoral/releases and install using pip:

```
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp36-cp36m-linux_aarch64.whl
```

and then in Python, check you can see your device:

```
import pycoral.utils.edgetpu
pycoral.utils.edgetpu.list_edge_tpus()

> [{'type': 'pci', 'path': '/dev/apex_0'}]
```

You'll see something different if you use the dev board or a USB dongle.

#### Testing

Grab the example repo:

```bash
mkdir coral && cd coral

git clone https://github.com/google-coral/pycoral.git

cd pycoral

bash examples/install_requirements.sh classify_image.py
```

and verify that the demo works:

```
josh@josh-jetson:~/code/coral/pycoral$ python3 examples/classify_image.py --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite --labels test_data/inat_bird_labels.txt --input test_data/parrot.jpg
----INFERENCE TIME----
Note: The first inference on Edge TPU is slow because it includes loading the model into Edge TPU memory.
14.1ms
3.3ms
2.6ms
2.5ms
2.5ms
-------RESULTS--------
Ara macao (Scarlet Macaw): 0.75781
```

Congratulations, you've now made an unholy marriage between Google and Nvidia.

### OpenCV

Thankfully you can install this easily with `pip install opencv-python` as there are now aarch64 wheels on PyPI.

### Torch

This is the batteries-included option. Make sure you've activated your `py36` environment.

Follow the instructions here: https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-9-0-now-available/72048

For example, if you want PyTorch 1.9.0:

```
wget https://nvidia.box.com/shared/static/h1z9sw4bb1ybi0rm3tu8qdj8hs05ljbm.whl -O torch-1.9.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 
pip install Cython
pip install numpy torch-1.9.0-cp36-cp36m-linux_aarch64.whl
```

And you should be good, double check by launching an interpreter and checking you can allocate a CUDA array, if using the Nano.

```
(py36) josh@josh-jetson:~/code/vision$ python
Python 3.6.9 (default, Jan 26 2021, 15:33:00)
[GCC 8.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> a = torch.ones((1,1)).cuda()
>>> a
tensor([[1.]], device='cuda:0')
```

#### Building from source

NVIDIA have a guide [here]() or you can use [this one]():

```
sudo apt-get update && sudo apt-get upgrade

# Get install dependencies
sudo apt-get install ninja-build git cmake  
sudo apt-get install libopenmpi-dev libomp-dev ccache
sudo apt-get install libopenblas-dev libblas-dev libeigen3-dev libjpeg-dev

# Remember, this is inside our venv
pip install setuptools wheel
pip install mock pillow  
pip install scikit-build  

# Clone Pytorch
git clone -b v1.8.1 --depth 1 --recursive https://github.com/pytorch/pytorch.git  
cd pytorch

# Install requirements 
# future, numpy, pyyaml, requests  
# setuptools, six, typing_extensions, dataclasses 
pip install -r requirements.txt

# Create a symlink to cublas
sudo ln -s /usr/lib/aarch64-linux-gnu/libcublas.so /usr/local/cuda/lib64/libcublas.so

# Set NINJA parameters

export BUILD_CAFFE2_OPS=OFF  
export USE_FBGEMM=OFF
export USE_FAKELOWP=OFF
export BUILD_TEST=OFF
export USE_MKLDNN=OFF
export USE_NNPACK=OFF
export USE_XNNPACK=OFF
export USE_QNNPACK=OFF
export USE_PYTORCH_QNNPACK=OFF
export USE_CUDA=ON
export USE_CUDNN=ON
export TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2"
export USE_NCCL=OFF  
export USE_SYSTEM_NCCL=OFF
export USE_OPENCV=OFF
export MAX_JOBS=2

# Set path to ccache

export PATH=/usr/lib/ccache:$PATH

# Set clang compiler

export CC=clang
export CXX=clang++

# Start the build

python setup.py bdist_wheel
```

This is a good time to go for a long walk, as this'll take a while. The CPU portion is fast, but the CUDA bit will tend to be slow. The result will be a wheel (`whl`) file in the `dist` folder which you can install with `pip`.

### TensorRT

First, install PyTorch using one of the above methods.

The install instructions are largely pilfered from [here](https://docs.donkeycar.com/guide/robot_sbc/tensorrt_jetson_nano/). Note that TensorRT is actually already installed, but we want nice bindings to it, which is what we're doing here.

Add the following lines to your `~/.bashrc` file:

```
# Add this to your .bashrc file
export CUDA_HOME=/usr/local/cuda
# Adds the CUDA compiler to the PATH
export PATH=$CUDA_HOME/bin:$PATH
# Add the libraries
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

And then `source ~/.bashrc` to take effect. Check that you can run the CUDA compiler:

```
josh@josh-jetson:~$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Wed_Oct_23_21:14:42_PDT_2019
Cuda compilation tools, release 10.2, V10.2.89
```

Install `pycuda` in your `py36` venv:

```
pip install pycuda
```

Check that you can import `tensorrt`:

```
> python
Python 3.6.9 (default, Jan 26 2021, 15:33:00)
[GCC 8.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorrt
>>> 
```

### Tensorflow

Just use Python 3.6. Again, NVIDIA to the rescue. Find out your JetPack version:

```
cat /etc/nv_tegra_release

# R32 (release), REVISION: 4.4, GCID: 23942405, BOARD: t210ref, EABI: aarch64, DATE: Fri Oct 16 19:44:43 UTC 2020
```

In this case `4.4` .

Make a new virtual environment, because the official instructions require specific versions of things:

```
python3.6 -m venv py36-tf
source py36-tf/bin/activate

sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
pip install --upgrade pip
pip install testresources setuptools wheel cython
pip install numpy==1.16.1
pip install future==0.17.1 mock==3.0.5 h5py==2.9.0 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
pip install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow==2.3.1+nv20.12
```

Are you bored of waiting for 10 minutes for Numpy to build? Of course you are.