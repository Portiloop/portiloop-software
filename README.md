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

- The `Channels` pannel enables you to configure each electrode:
  - `disabled`: the electrode is not used
  - `simple`: the electrode is simply used to measure signal (not recommended)
  - `bias in`: the electrode is used to measure signal and to compute a bias ("ground") signal
  - `bias out`: the electrode is used to output the bias ("ground") signal
- Use the `Freq` widget to enter your desired sampling rate
- Use the `Time` widget to enter a maximum duration for the experiment (you can also stop the experiment manually)
- Use the `Recording` widget to enter the name of a `.edf` output file if you wish to record the signal
- If you tick the `Record` checkbox, the signal will be recorded in this file
- If you tick the `Display` checkbox, the signal will be displayed for the duration of the whole experiment
- The `Clock` widget lets you select the sampling method:
  - `Coral` sets the `ADS1299` sampling rate to twice your target, and uses the Coral RT clock to sample at your target
  - `ADS` sets the `ADS1299` sampling rate to the closest compatible to your target and uses the ADS interrupts to sample
- Finally, the `Capture` widget lets you start and stop the experiment at any point in time

_Note: once the experiment is started, all widgets are deactivated until you stop the experiment._

## Installation:

Follow these instruction if the software is not readily installed on your `Portiloop` device.

### Install the library:

_(Requires python 3)_

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
