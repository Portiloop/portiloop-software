# Portiloop software

This software works with the [Coral implementation](https://github.com/Portiloop/portiloop-hardware) of the `Portiloop` EEG closed-loop stimulation device.

It enables controlling the `Portiloop` from a Graphical User Interface (GUI).

## Quick links
- [Installation on the Portiloop](#installation-portiloop-v3)
- [GUI usage](#usage)
- [Developer guide](#developer-guide)

## Installation (Portiloop V2.3):

You have just got your hands on the hardware for the Portiloop V2.3 (A Google Coral Dev Board Mini and a Portiloop board). Here are the steps you need to follow to get started.

### Hadware prerequisites
- A male-to-female jupmer wire (required to reflash if you brick the Coral Dev Board Mini)
- Headphones with a jack that has TWO rings (not three)


### Flashing the Google Coral
Find the instructions to update your Coral Dev Board Mini to the latest OS version [here](https://coral.ai/docs/dev-board-mini/reflash/).

_(We recommend the force-fastboot method, as it works without `mdt`)_

:warning: : The portiloop software will not work if you ignore this step.
The software is developed for version 5.0 Eagle (Dec 2020), downloadable [here](https://dl.google.com/coral/mendel/excelsior/excelsior-eagle-20201210233645.zip)

:warning: If at any point you brick your Portiloop (i.e., if the power light gets stuck on orange and never turns green, most likely because you failed to turn the Coral off gracefully), the only solution is to reflash the Coral with the force fastboot method, using a female-to-male jumper wire as described [here](https://gweb-coral-full.uc.r.appspot.com/docs/dev-board-mini/reflash/#force-boot-into-fastboot-mode). We also recommend that you wipe out the content of the home folder by executing `bash flash.sh -H` instead of executing `bash flash.sh` at the end of these instructions.

### Accessing the Google Coral

These first steps will help you set up an SSH connection to the device.

- Power up the board through the USB-C power port.
- Connect another USB cable to the OTG-port on the board and to your _Linux_ host machine. Then connect to the board through serial (or via `mdt`) by executing `screen /dev/ttyACM0` (or `mdt shell`)

_If you see a message telling you that screen is busy, you can use `sudo lsof /dev/ttyACM0` and then retry the screen step._

- Login to the board using the default username and password: mendel
- Once you are logged in, you can now connect to your desired wifi network using `nmtui`.
- Enable SSH access with password by executing the following command:

`sudo sed -i 's/PasswordAuthentication no/PasswordAuthentication yes/g' /etc/ssh/sshd_config`

_(if you get an error message, you can instead execute `sudo nano /etc/ssh/sshd_config`, find the `PasswordAuthentication no` line, replace 'no' by 'yes', and save by pressing `CTRL + o`, then `ENTER`, then `CTRL + x`)_

- Note your randomly-generated hostname (displayed as `mendel@your-hostname`) or define a custom hostname (`sudo hostnamectl set-hostname your-hostname`)
- Shutdown the device (`sudo shutdown now` or press the `Power` button for 3 seconds) and pull out the OTG-port cable before the Coral board reboots (which otherwise happens automatically after a few seconds as long as this cable is plugged in).

Next time you turn the Coral board on, you should be able to ssh into it using the hostname (or the IP address of the device):
- `ssh mendel@your-hostname.local`

If some issues arise, make sure your PC is connected to the same network as the Coral Dev Board Mini.

### Software installation

- Plug the `USB-C Power` cable in and turn the portiloop on by pressing the `Power` button for 3 seconds.
- SSH into the device
- Clone this repository in the home folder: `cd ~ && git clone https://github.com/Portiloop/portiloop-software.git`
- Go into the cloned repository: `cd ~/portiloop-software`,
- Run `make` and follow the instructions when prompted
  - Don't forget to reboot the device afterward
- Note that `make` may fail at several points during installation. Whenever it does, just call `make` again.

That's it! Your Jupyter server should now be up and running, listening on IP address `192.168.4.1` and port `8080`, and automatically starting whenever the system boots up. You can now access it by typing `192.168.4.1:8080` in your browser. This should lead you to a login page where you'll be prompted for your password. If any issue arises, try with a different web browser.
Similarly, the `Simple UI` can be accessed by typing `192.168.4.1:8081` in your browser. 

## Usage:

### SD card

To work on the Portiloop, your SD card must have a partition. You can check whether your SD card has a partition by accessing the Coral via SSH, and executing `lsblk`. This command should output something like:
```
NAME         MAJ:MIN RM  SIZE RO TYPE MOUNTPOINT
mmcblk0      179:0    0  7.3G  0 disk 
├─mmcblk0p1  179:1    0    4M  0 part 
├─mmcblk0p2  179:2    0  128M  0 part /boot
├─mmcblk0p3  179:3    0    2G  0 part /home
└─mmcblk0p4  179:4    0  5.2G  0 part /
mmcblk0boot0 179:32   0    4M  1 disk 
mmcblk0boot1 179:64   0    4M  1 disk 
mmcblk2      179:96   0  119G  0 disk 
└─mmcblk2p1  179:97   0  119G  0 part
```
i.e., the SD card should appear as `mmcblk2` and have a partition appearing as `mmcblk2p1`.

When using an SD card to record your EEG signal in CSV format, plug the SD card into your `Portiloop` before powering the `Portiloop` on.
Your CSV will then be recorded in the SD card under the `workspace` folder.
Otherwise, your CSV will be recorded in internal memory under `/home/mendel/workspace`

_:warning: Recording large CSV files in internal memory will quickly make your `Portiloop` unusable: the internal memory is quite small and recording CSVs in internal memory should never be done, except for quick testing.
In case you inadvertently fill up your `Portiloop` internal memory, it will refuse to boot and you will have to reflash and reinstall the entire system._

### Headphones

To use headphones on the `Portiloop`, plug the headphones in before powering the device.
Then, when you turn the device on, it should play a sound in the headphones when the light turns green.

### Power up

To power the `Portiloop`, plug your USB-C battery into `USB-C Power` and press the `Power` button for 3 seconds.
Then, wait for the light next to `USB-C Power` to turn green (the green color indicates that boot was successful).

### Power down

To power the `Portiloop` down, press the `Power` button for 3 seconds again and wait for a couple more seconds for the light close to `USB-C Power` to turn off.
You can then unplug the `USB-C Power` cable.

When everything goes smoothly, the light close to `USB-C Power` turns off.
However, it may happen that this light refuses to turn off due to some internal issue.
In that case, just wait for a couple more seconds before unplugging.

### Indicator LEDs

The Portiloop system has 3 indicator LEDs:
- Coral board LED between the two USB-C connectors:
  - orange: boot in progress
  - green: boot complete
  - red: the Coral Dev Board Mini is in fastboot mode, ready to flash
  - turned off when the Portiloop is off
- Portiloop power LED
  - (disabled by default, can be enabled programatically)
- Portiloop indicator LED
  - (disabled by default, can be enabled programatically)

### Connect

Connect your computer (or smartphone) to the WiFi access point of the `Portiloop` that you want to use.

The `Portiloop` has two web-based Graphical User Interfaces that you can access via any web browser:
- A user-friendly `Simple UI`
  - _accessible via `192.168.4.1:8081`_
- An advanced UI in the form of a `jupyter` notebook
  - _accessible via `192.168.4.1:8080`_

### Simple UI
To access the `Simple UI`, open your favorite browser and enter the following address: `192.168.4.1:8081`.

This UI is pretty self-explanatory.
It has several options, including:
- Detecting patterns of interest in real-time (e.g., Sleep Spindles)
- Performing closed-loop stimulation based on this detection
- Recording raw EEG along with the above detections in a CSV file

You can select the relevant pipeline and parameters for your experiments in the `Advanced` tab.

:information_source: The `Acquisition only` pipeline does not perform any detection or stimulation.

### Jupyter UI

The `Portiloop` advanced UI is a web-based interface running as a `jupyter` server.
To access this UI, open your favorite browser and enter the following address: `192.168.4.1:8080`.

You should now be connected to the `jupyter` server.

_If the jupyter notebook is not yet created:_
- Hit `New` and select `Python 3`.

This creates a `jupyter` notebook, in which you can simply paste and execute the following:

```python
from portiloop.capture import JupyterUI

cap = JupyterUI()
```

#### Channels:

The `Channels` panel enables you to configure each electrode:
- `simple`: the electrode is used to measure signal
- `bias`: the electrode is used to output the measured bias ("ground") signal
- `test`: the electrode is used to output a test signal
- `temp`: the electrode is used to output a signal corresponding to the ADS temperature (conversion required)

#### General controls:

- `Freq` is the desired sampling rate
- `Time` is the maximum duration of the experiment (you can also stop the experiment manually)
- `Recording` is the name of the `.csv` output file if you wish to record the signal locally
- Tick `Filter` to enable the online filtering pipeline
- Tick `Detect` to enable the online detection pipeline
- Tick `Stimulate` to enable the online stimulation pipeline
- Tick `Record CSV` to record the signal in the file designated in `Recording`
- Tick `Stream LSL` to broadcast the signal on the local network via [LSL](https://labstreaminglayer.readthedocs.io/info/intro.html)
- Tick `Display` to display the signal in the GUI
- `Threshold` enables customizing the optional detection threshold from the GUI (e.g., for classifiers)
- The `Clock` widget lets you select the sampling method:
  - `Coral` sets the `ADS1299` sampling rate to twice your target sampling rate, and uses the Coral Real-Time clock to stick to your target sampling rate
  - `ADS` sets the `ADS1299` sampling rate to the closest compatible to your target sampling rate and uses the ADS interrupts

#### Custom Filtering

The `Filtering` section lets you customize the filtering pipeline from the GUI.

- The `FIR filter` switch lets you select between the default low-pass FIR filter (used in the Portiloop [paper](https://arxiv.org/abs/2107.13473)), or customize this filter according to your needs (`FIR order` and `FIR cutoff`)
- `Polyak mean`, `Polyak std` and `Epsilon` let you customize the online standardization pipeline, which also acts as a high-pass filter (only available in the `Sleep Spindles` pipeline)


## Developer guide:

The core Portiloop software architecture is defined in `portiloop.src.core`.
It defines the three interfaces that developers of custom pipelines must implement:
- `processing.py` defines the `Processor` interface, in charge of all signal processing.
- `detection.py` defines the `Detector` interface, in charge of all detection algorithms/models.
- `stimulation.py` defines the `Stimulator` interface, in charge of all response to the detector output.

The Portiloop software is a Python library.
You may implement the aforementioned interfaces, put these in a `pipeline` dictionary.
The `portiloop.custom.custom_pipelines` module defines the `PIPELINES` dictionary, which you can modify to define your custom Portiloop pipeline: just add an entry following the existing template, i.e.:

```python
from portiloop.src.custom.custom_pipelines import PIPELINES

PIPELINES["Your_Pipeline_Name"] = { # This name will appear in the Portiloop GUIs.
    "processor": Your_Processor_Class, # class of your Processor implementation (not instance).
    "detector": Your_Detector_Class, # class of your Detector implementation (not instance).
    "stimulator": Your_Stimulator_Class, # class of your Stimulator implementation (not instance).
    "config_modifiers": {}
},
```
You can then feed this dictionary as argument to the Portiloop GUIs.

## Contribute:

Developers of new Portiloop pipelines that are to be merged with the repo must work entirely under the `portiloop.custom` package, where abstract interfaces are implemented for tasks such as Sleep Spindle or Sleep Slow Oscillations detection and stimulation.
Abide only to these interfaces and do not make additional assumptions, otherwise your custom pipeline will break the Portiloop GUIs, which rely exclusively on these interfaces.

:warning: Non-core internal developers should not modify the `portiloop.core` package, nor the `jupyter_gui` and `simple_gui` packages.
If you feel you need to do any of these things, please ask for assistance, as you are probably doing something wrong.