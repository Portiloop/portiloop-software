# Portiloop software

This software works with the [Coral implementation](https://github.com/Portiloop/portiloop-hardware) of the `Portiloop` EEG closed-loop stimulation device.

It enables controlling the `Portiloop` from a simple Graphical User Interface (GUI).

## Quick links
- [Installation on the Portiloop](#installation)
- [GUI usage](#usage)

## Usage:

The `Portiloop` GUI is a web-based interface running as a `jupyter` server.

- Connect to the `Portiloop` WiFi network.
- Open your favorite web browser
- Enter the following address: `192.168.0.1:9000`

You should now be connected to the `jupyter` server.

_If the jupyter notebook is not yet created:_
- Hit `New` and select `Python 3`.

This creates a `jupyter` notebook, in which you can simply paste and execute te following:

```python
from portiloop.capture import Capture

cap = Capture()
```

_When the jupyter notebook is created:_

You can open the notebook and simply execute the cell.

The GUI now looks like this:

![gui](figures/gui.png)

### Channels:

The `Channels` pannel enables you to configure each electrode:
- `disabled`: the electrode is not used
- `simple`: the electrode is simply used to measure signal (not recommended)
- `with bias`: the electrode is used to measure signal and to compute a bias ("ground") signal
- `bias out`: the electrode is used to output the bias ("ground") signal

### General controls:

- `Freq` is the desired sampling rate
- `Time` is the maximum duration of the experiment (you can also stop the experiment manually)
- `Recording` is the name of the `.edf` output file if you wish to record the signal locally
- Tick `Filter` to enable the online filtering pipeline
- Tick `Detect` to enable the online detection pipeline
- Tick `Stimulate` to enable the online stimulation pipeline
- Tick `Record EDF` to record the signal in the file designated in `Recording`
- Tick `Stream LSL` to broadcast the signal on the local network via [LSL](https://labstreaminglayer.readthedocs.io/info/intro.html)
- Tick `Display` to display the signal in the GUI
- `Threshold` enables customizing the optional detection threshold from the GUI (e.g., for classifiers)
- The `Clock` widget lets you select the sampling method:
  - `Coral` sets the `ADS1299` sampling rate to twice your target sampling rate, and uses the Coral Real-Time clock to stick to your target sampling rate
  - `ADS` sets the `ADS1299` sampling rate to the closest compatible to your target sampling rate and uses the ADS interrupts

### Custom Filtering

The `Filtering` section lets you customize the filtering pipeline from the GUI.

- The `FIR filter` switch lets you select between the default low-pass FIR filter (used in the Portiloop [paper](https://arxiv.org/abs/2107.13473)), or customize this filter according to your needs (`FIR order` and `FIR cutoff`)
- `Polyak mean`, `Polyak std` and `Epsilon` let you customize the online standardization pipeline, which also acts as a high-pass filter

### Capture

The `Capture` switch lets you start and stop the experiment at any point in time

_Note: once the experiment is started, all widgets are deactivated until you stop the experiment._

## Installation:

Follow these instruction if the software is not readily installed on your `Portiloop` device.

### Install the library:

_(Requires python 3)_

#### Install the following libraries from apt to avoid issues:
- `sudo apt install python3-numpy`
- `sudo apt install python3-scipy`
- `sudo apt install python3-pycoral`
- Clone this repository on the `Coral` board
- `cd` to he root of the repository where the `setup.py` file is located
- Execute `pip3 install -e .`

### Setup the Coral board as a wifi access point

You can find instructions [here](https://www.linux.com/training-tutorials/create-secure-linux-based-wireless-access-point/) to set Linux as a WiFi access point.

### Setup a jupyter server:

- On your `Portiloop` device, execute `pip3 install notebook`
- Generate a `jupyter` password and copy the result:
```python
from notebook.auth import passwd
passwd()
```
- Execute `jupyter notebook --generate-config`
- `cd` to the `.jupyter` folder and edit `jupyter_notebook_config.py`
- Find the relevant lines, and uncomment them while setting the following values:
  - `c.NotebookApp.ip = '*'`
  - `c.NotebookApp.open_browser = False`
  - `c.NotebookApp.password = u'your_generated_password_here'`
  - `c.NotebookApp.port = 9000`

### Setup a service for your jupyter server to start automatically:

- `cd /etc/systemd/system`
- create an empty file named `notebook.service` and open it.
- paste the following and save:
```bash
[Unit]
Description=Autostarts jupyter server

[Service]
User=mendel
WorkingDirectory=~
ExecStart=jupyter notebook
Restart=always

[Install]
WantedBy=multi-user.target
```
- Execute `sudo systemctl daemon-reload`
- Execute `sudo systemctl start notebook.service`
- Check that your service is up and running: `sudo systemctl status notebook.service`
