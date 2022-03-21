## Google Coral Development Board setup

Follow Google's documentation for [Getting Started with the Dev Board](https://coral.ai/docs/dev-board/get-started/)  or [Dev Board Mini](https://coral.ai/docs/dev-board-mini). This should get your system up-and-running and it also discusses installing PyCoral and testing that the EdgeTPU works.

Google suggests the following procedure to make sure your board is fully up to date (not a bad idea):

```
sudo apt-get update

sudo apt-get dist-upgrade

sudo reboot now
```

You can now move on to [step 2](), using whatever login method you prefer - monitor/keyboard, via SSH or mdt, etc.

## Jetson Nano System setup

This tutorial assumes that you have the following:

* An NVIDIA Jetson Nano developer kit
* A Google Coral EdgeTPU accelerator (either a USB or M2 device)

### Flashing, etc

Here we're going to be using a Jetpack 4.4 image, but the latest version from NVIDIA should work, you can download [here](https://developer.nvidia.com/embedded/jetpack). I suggest using an SD card image as it's generally simpler than using the SDK manager, especially if you don't have a Ubuntu machine to hand.

You should have a zip file called something like jetson-nano-4gb-jp441-sd-card-image.zip

Pick a big SD card - nowadays 256GB is pretty cheap and will give you tons of space. I wouldn't suggest anything under 32GB - the 4GB Nano image is 14.4GB extracted, and you'll want space for swap, packages, etc. I've never had issues with cards wearing out, but if this is an issue then you can add an external USB drive to store data that you frequently access. Buy the fastest card you can afford, e.g. a Sandisk Extreme/Ultra.

Use you preferred method to flash the card, I would recommend [Etcher](https://www.balena.io/etcher/) for beginners. If you use the portable version, you may need to run as Administrator.

For more information, see [here](http://docs.nvidia.com/jetson/jetpack/install-jetpack/index.html).

### Networking/headless access

The development board does not support WiFi without a dongle, so you need to connect via Ethernet. SSH via the primary user (whatever you created on first setup) is supported out of the box - it's not like the Raspberry Pi which needs specific enabling.

You can also connect via serial, or use a monitor/keyboard as normal.

If you're connecting to your home network, it's a good idea to give your Nano a sensible hostname and to give it a static IP address on your router.

You'll need a monitor and keyboard for the first setup, where you'll have to agree to NVIDIA's licenses and so on. Once you've done that, you don't need it again. In this process you'll set up the default user and password.

Plug the Nano into your network and verify that you can SSH in.

### Adding swap space

I run from a 256 GB SD card, which leaves plenty of space (with room for swap). Once you're online, add some swap space:

```
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

then add the following line to your `fstab`:

```
/swapfile swap swap defaults 0 0
```

### Initial system update

As a general rule, as soon as you've got the system up and running (e.g. SSH works) you should run:

```
sudo apt update
sudo apt upgrade -y
```

to get the latest version of the normal system packages. In addition you can install some useful utilities:

```
sudo apt install htop tmux tree git 
```

### Installing the Google Coral/EdgeTPU

These notes describe the setup and testing of the Google Coral Edge TPU in the Jetson Nano development board. Specifically the Coral model [G650-04527-01 (M.2 A/E key)](https://www.mouser.co.uk/new/google/coral-m2-accelerator-ae/) was used. This model mates with the M.2 E slot on the Nano dev board.

#### Mechanical and Electrical

Take the usual precautions for static discharge, if it's an issue where you live (such as the South Pole). The Nano dev board includes an M.2 expansion slot underneath the Jetson Nano module. Begin by removing the two screws on the module and press the sides of the DIMM header to release the module. It will flip upwards and can then be pulled out.

The M.2 retaining screw on the board I used was extremely tight and required a fairly large screwdriver to loosen. Caution, as it seems quite easy to strip the head using smaller precision bits. Otherwise it should be straightforward to insert the TPU board into the slot and to fix it with the screw.

When installed (prior to screwing down the Jetson module), the board should look something like below:

![Google Coral installation into Jetson Nano dev board](../main/images/jetson_nano_coral.png)

Note that there is no thermal management here and there is no room for a heatsink on the TPU itself. Forced air cooling might be worth using. It's quite possible that in a hot environment, under load, the EdgeTPU will throttle. The dev board is not designed for production anyway so just bear it in mind.

### Power and thermals

It's probably a good idea to power the development board using the barrel jack as peripherals (like the EdgeTPU) can draw a reasonable amount of current under load. If you do this, you need to jumper **J48** to enable the barrel jack and to disable the USB power supply. Then a 5V (up to 4A) power supply can be used.

The barrel adapter is **centre-positive**.

The Jetson Nano supports a 5V 40x40mm PWM fan - e.g. a [Noctua](https://noctua.at/en/products/fan/nf-a4x10-5v-pwm). It can be attached to the heatsink using M3 screws. 12V fans may also work, but they will run slow. The fan connects via a standard ATX header on the dev board the same header that you find on desktop motherboards), so any off the shelf fan that fits should work. This is also recommended because while the module has a big ol' heatsink, it still gets quite warm under load.

I put my Nano in a nice 3D printed [enclosure](https://www.thingiverse.com/thing:3518410), which came out reasonably well on the Ultimaker S5 we have here at the South Pole. Some of the port walls are a little thin, but they work. Here's the final product - directed down seems to work best[0]:

![Jetson nano in an enclosure](../main/images/jetson_nano_case.png)

### Jetson Nano kernel configuration

The Nano does require a couple of kernel command line adjustments to make the EdgeTPU work. In `/boot/extlinux/extlinux.conf` find the line starting with `APPEND ${cbootargs} ` and add to the end:

```
pcie_aspm=off
```

then reboot.

It's not clear (to me) why this fixes the issue, but it seems that Active State Power Management (ASPM) should be disabled. I also saw a lot of repeated error messages related to AER which you can disable with `pci=noaer`, but you probably want this enabled. I wasn't able to get the device to communicate with the driver without doing this, my development board would also reboot sporadically.

### EdgeTPU Driver installation

First double check that the device appears in `lspci`, it may not be called anything at this point (before driver installation it may initially appear as a "Non-VGA device") but it should be present in the list:

```
yolo_env) josh@josh-jetson:~/code/yolov5$ lspci
00:01.0 PCI bridge: NVIDIA Corporation Device 0fae (rev a1)
00:02.0 PCI bridge: NVIDIA Corporation Device 0faf (rev a1)
01:00.0 System peripheral: Device 1ac1:089a 
02:00.0 Ethernet controller: Realtek Semiconductor Co., Ltd. RTL8111/8168/8411 PCI Express Gigabit Ethernet Controller (rev 15)
```

Follow the instructions on [Google's site](https://coral.ai/docs/m2/get-started/). The only modification here is to also add the user to `plugdev` which may or may not be necessary.

```bash
# Add Nvidia repository
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update

# Install pcie driver
sudo apt-get install gasket-dkms libedgetpu1-std

# Add user to apex/plugdev
sudo sh -c "echo 'SUBSYSTEM==\\"apex\\", MODE=\\"0660\\", GROUP=\\"apex\\"' >> /etc/udev/rules.d/65-apex.rules"

sudo groupadd apex

sudo adduser $USER apex
sudo usermod -aG plugdev $USER
```

Then reboot. Verify that the accelerator comes up:

```bash
lspci -nn | grep 089a
03:00.0 System peripheral: Device 1ac1:089a
```

and that the `apex` device exists:

```
ls /dev/apex\_0
```

[0] https://www.patricksteinert.de/technology/internet-of-things/nvidia-jetson-nano-fan-direction/
