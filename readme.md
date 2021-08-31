[TOC]

## Ultralytics Competition Notes

This repository is an entry into the Ultralytics export challenge for the EdgeTPU. It provides the following solution:

* A minimal repository which has extremely few dependencies:
  * `pycoral` , `opencv` for image handling (you could drop this using e.g Pillow) and `numpy`
  * Other "light" dependencies include `tqdm` for pretty logging, and `yaml` for parsing names files.
  * **No dependency on Torch**, _which means no building Torch_ - from clone to inference is extremely fast.
  * Code has been selectively taken from the original Ultralytics repository and converted to use Numpy, for example non-max suppression. There is essentially no speed penalty for this on a CPU-only device.
* I chose _not_ to fork ultralytics/yolov5 because the competition scoring was weighted by deployment simplicity. Installing Torch and various dependencies on non-desktop hardware can be a significant challenge - and there is no need for it when using the tflite-runtime.
  * This does mean that currently there is no simple benchmark code in this repo, but it should be easy to support.
* Packages are easily installable on embedded platforms such as the Google Coral Dev board and the Jetson Nano
* This repository uses the Jetson Nano as an example, but the code should be transferrable given the few dependencies required
  * Non-tested setup instructions are given for the Coral, but these are largely based on Google's guidelines.
* tflite export is taken from https://github.com/zldrobit/yolov5:
  * These models have the detection layer built-in. This provides a significant speed boost, but does mean that larger models are unable to compile.
* Speed is good: you can expect 24 fps using the EdgeTPU on a Jetson Nano for a 224 px input.
  * You can easily swap in a different model/input size, but larger/smaller models are going to vary in runtime and accuracy.
  * The workaround for exporting a 416 px model is to use an older runtime version where the transpose operation is not supported. This significantly slows model performance because then the `Detect` stage must be run as a CPU operation. See [bogdannedelcu](https://github.com/bogdannedelcu/yolov5-export-to-coraldevmini)'s solution for an example of this.
    * Note this approach doesn't work any more because the compiler supports the Transpose option. I tried exporting with different model runtimes in an attempt to force the compiler to switch to CPU execution before these layers, but it didn't seem to help.
* Extensive documentation is provided for hardware setup and library testing. This is more for the Jetson than anything else, as library setup on the Coral Dev Board should be minimal.
* A Dockerfile is provided for a repeatable setup and test environment

## Introduction

In this repository we'll explore how to run a state-of-the-art object detection mode, [Yolov5], on the Google Coral EdgeTPU. 

**TL;DR (see the Dockerfile):**

```
sudo apt-get update && sudo apt-get -y upgrade
sudo apt-get install -y git curl gnupg

# Install PyCoral (you don't need to do this on a Coral Board)
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
sudo apt-get update
sudo apt-get install -y gasket-dkms libedgetpu1-std python3-pycoral

# Get Python dependencies
sudo apt-get install -y python3 python3-pip
pip3 install --upgrade pip setuptools wheel
python3 -m pip install numpy
python3 -m pip install opencv-python-headless
python3 -m pip install tqdm pyyaml

# Clone this repository
git clone https://github.com/jveitchmichaelis/edgetpu-yolo
cd edgetpu-yolo

# Run the test script
python test_edgetpu.py -m yolov5s-int8-224_edgetpu.tflite --bench_image --bench_speed
```

Wasn't that easy? You can swap out different models and try other images if you like. You should see an inference speed of around 25 fps with a 224x224 px input model.

Note if you're using a PCIe accelerator, you will need to install an appropriate kernel driver. See the hardware notes for more information.

## Dev/Further instructions

1. Hardware setup (hardware.md)
   * Briefly covers setup for the Coral Dev Board(s)
   * Covers electrical and mechanical setup for the Jetson Nano, EdgeTPU driver installation, etc.
2. On-device software setup (software.md)
   * Setting up virtual environments and Docker
   * Installing `pycoral` and related libraries
   * Notes on installing PyTorch, OpenCV etc from source [for development and testing work]
3. Model generation and export (export.md)
   * Exporting a TFLite model from PyTorch
   * Notes on the `edgetpu_compiler`

## Running Inference

As the introduction says, all you need to do is install the dependencies and then run:

```
python test_edgetpu.py -m yolov5s-int8-224_edgetpu.tflite --bench_image --bench_speed
```

This should give you first a speed benchmark (on 100 images - edit the file if you want to run more) and then on the Zidane test image (you should get two detections for the 224 model).

I've also included an (untested) option to run from a video stream.

The provided code is pretty much the minimal you need to get going with the TPU. It provides a simple class for loading the model and running inference. There are also a few utilities copied from Yolov5 for image annotation, but it's very basic at this stage.

You can also use the `EdgeTPUModel` class in your own software quite easily:

```
from edgetpumodel EdgeTPUModel
from utils import get_image_tensor

model = EdgeTPUModel("model_name", "names.yaml")
input_shape = model.get_input_shape()

full_image, net_image, pad = get_image_tensor("/path/to/image", input_shape[0])
pred = model.predict(net_image)
model.process_predictions(pred[0], full_image, pad)
```

It's not yet ready for production(!) but you should find it easy to adapt.

## Benchmarks/Performance

Here is the result of running three different models. All benchmarks were performed using an M.2 accelerator on a Jetson Nano 4GB.

* 96x96 input, runs fully on the TPU ~60-70fps
* 192x192 input, runs mostly on the TPU ~30-35fps
* 224x224 input, runs mostly on the TPU ~25-30 fps
* \>= 256 px currently fails to compile due to large tensors. It's probable that the backbone alone would compile fine and then detection can run on CPU, but this is typically extremely slow - an order of magnitude slower. Better, I think, to explore options for Yolov5 models with smaller width/depth parameters.

```
(py36) josh@josh-jetson:~/code/edgetpu_yolo$ python test_edgetpu.py -m yolov5s-int8-96_edgetpu.tflite --bench_speed
INFO:EdgeTPUModel:Loaded 80 classes
INFO:__main__:Performing test run
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 58.28it/s]
INFO:__main__:Inference time (EdgeTPU): 13.40 +- 1.68 ms
INFO:__main__:NMS time (CPU): 0.43 +- 0.39 ms
INFO:__main__:Mean FPS: 72.30

(py36) josh@josh-jetson:~/code/edgetpu_yolo$ python test_edgetpu.py -m yolov5s-int8-192_edgetpu.tflite --bench_speed
INFO:EdgeTPUModel:Loaded 80 classes
INFO:__main__:Performing test run
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:03<00:00, 30.85it/s]
INFO:__main__:Inference time (EdgeTPU): 26.43 +- 4.09 ms
INFO:__main__:NMS time (CPU): 0.77 +- 0.35 ms
INFO:__main__:Mean FPS: 36.77

(py36) josh@josh-jetson:~/code/edgetpu_yolo$ python test_edgetpu.py -m yolov5s-int8-224_edgetpu.tflite --bench_speed
INFO:EdgeTPUModel:Loaded 80 classes
INFO:__main__:Performing test run
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:03<00:00, 25.15it/s]
INFO:__main__:Inference time (EdgeTPU): 33.31 +- 3.69 ms
INFO:__main__:NMS time (CPU): 0.76 +- 0.12 ms
INFO:__main__:Mean FPS: 29.35
```

I have not performed an explicit accuracy benchmark on these models, but I will say that 96x96 is probably unusable unless it was a model that was properly quantisation-aware trained and was for a very limited task. 224px gives good results on standard images, e.g. zidane, but it won't find the tie. This is quite normal for edge-based models with small inputs.

As far as I'm aware, the original TFLite models can run on the desktop and can be analysed as usual that way.

You could attempt to tile the model on larger images which may give reasonable results.