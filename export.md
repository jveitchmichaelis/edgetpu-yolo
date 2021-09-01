

## Setting up Yolov5

Hoo boy, that took a while. Now we're in a position to start exporting and converting models!

### On your desktop:

Get the repository and follow Ultralytics' install instructions:

```
git clone  https://github.com/ultralytics/yolov5
cd yolov5
pip3 install -r requirements.txt
```
Note tf export has now been merged into the main repository, so you should be good.

## Exporting models

Generally we're going to assume that you're _not_ doing this from your edge device. You will need a desktop computer with various packages installed. This is actually a requirement for the EdgeTPU compiler anyway.

### PyTorch

Standard Yolov5 is an exported PyTorch `.pt` file. You can download the 'official' release checkpoints from Ultralytics, or you can train your own Yolov5s. It's strongly recommended that you use the `s` variant, as the other models are too large to run on-device.

### Tflite

A tflite model is necessary to export to the EdgeTPU. Tensorflow includes a tflite converter which accepts either a saved Tensorflow graph or a Keras model. I would guess most people will be comfortable using Keras models. Fortunately, the hard work has been done for us, so all you need to do is run:

```
python .\models\tf.py --img-size 224 --tf-raw-resize --source D:\data\coco\images\train2014 --tfl-int8 --ncalib 1000
```

By default this will pick up the yolov5s.pt weights.

Note `--img-size` should be a multiple of 16 generally, 224 is a good speed/accuracy trade-off. You can try going higher, but at some point the models will no longer compiler for Edge. `--source` should be a path to some images that the model use as a representative dataset. They don't need to be anything special, just random images (so we tend to use COCO). These are used to determine how to scale the weights in the model. We also need `--tfl-int8` so that only `INT8` supported operations are generated.

What this code does internally is create a version of Yolov5 in Keras. For each PyTorch layer, there is an equivalent hand-coded Keras layer (well, most are off-the-shelf except for the Detection layer at the end). Then the TFLite converter API is used to export the model. At this point, you should have a `.tflite` model. It is not yet suitable for the EdgeTPU, but you can benchmark it at least.

Why do we set an image size? The EdgeTPU requires fixed size inputs and so we have to bake this into the model. This is pretty common for edge models (e.g. with the Neural Compute Stick). The input size is however arbitrary, provided the model fits into RAM. If you have e.g. a couple of convolution layers, the TPU will happily support input sizes of over 1024x1024. It will also support arbitrary numbers of input channels.

The resulting model requires input and output scaling, as described [here](https://www.tensorflow.org/lite/performance/quantization_spec). From what I've seen with these models, the conversion is basically a normalisation to [0,255) and the output is the inverse. With other models (which accept say input values outside this range) you might see much more unusual scaling/zero point values.

### EdgeTPU

This assumes you've exported a tflite model as above, using 8-bit input/output quantisation.

You will need an x86 system running Linux. You can also use Google's web-based compiler (upload your tflite model):

https://colab.research.google.com/github/google-coral/tutorials/blob/master/compile_for_edgetpu.ipynb

If you're running Windows, this will work perfectly well inside WSL (i.e. install Ubuntu from the app store). To install the compiler, follow the steps in the notebook on your own machine. It's straightforward:

```bash
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install edgetpu-compiler	
```

and run it to check:

```bash
edgetpu_compiler -v
Edge TPU Compiler version 16.0.384591198
```

then export your model:

```
edgetpu_compiler -sa yolov5s-224-int8.tflite -d -t 600
```

And you should see something like:

```
edgetpu_compiler -sa yolov5s-int8-224.tflite
Edge TPU Compiler version 16.0.384591198
Started a compilation timeout timer of 180 seconds.

Model compiled successfully in 3173 ms.

Input model: yolov5s-int8-224.tflite
Input size: 7.37MiB
Output model: yolov5s-int8-224_edgetpu.tflite
Output size: 8.34MiB
On-chip memory used for caching model parameters: 6.08MiB
On-chip memory remaining for caching model parameters: 0.00B
Off-chip memory used for streaming uncached model parameters: 1.70MiB
Number of Edge TPU subgraphs: 2
Total number of operations: 294
Operation log: yolov5s-int8-224_edgetpu.log

Model successfully compiled but not all operations are supported by the Edge TPU. A percentage of the model will instead run on the CPU, which is slower. If possible, consider updating your model to use only operations supported by the Edge TPU. For details, visit g.co/coral/model-reqs.
Number of operations that will run on Edge TPU: 289
Number of operations that will run on CPU: 5

Operator                       Count      Status

RESIZE_NEAREST_NEIGHBOR        2          Mapped to Edge TPU
CONCATENATION                  1          Operation is otherwise supported, but not mapped due to some unspecified limitation
CONCATENATION                  17         Mapped to Edge TPU
STRIDED_SLICE                  13         Mapped to Edge TPU
TRANSPOSE                      1          Operation is otherwise supported, but not mapped due to some unspecified limitation
TRANSPOSE                      2          Mapped to Edge TPU
MAX_POOL_2D                    3          Mapped to Edge TPU
LOGISTIC                       62         Mapped to Edge TPU
QUANTIZE                       2          Operation is otherwise supported, but not mapped due to some unspecified limitation
QUANTIZE                       24         Mapped to Edge TPU
SUB                            3          Mapped to Edge TPU
PAD                            6          Mapped to Edge TPU
RESHAPE                        1          Operation is otherwise supported, but not mapped due to some unspecified limitation
RESHAPE                        5          Mapped to Edge TPU
ADD                            10         Mapped to Edge TPU
MUL                            80         Mapped to Edge TPU
CONV_2D                        62         Mapped to Edge TPU
Compilation child process completed within timeout period.
Compilation succeeded!
```

The output file will be called `yolov5s-int8-224_edgetpu.tflite` (or you can use the `-o` option to fix it).

That's it! You can now run this model on your EdgeTPU, but you'll need some additional logic to process the output.
