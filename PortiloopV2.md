
# Portiloop V2

You've just got your hands on the hardware for the Portiloop V2 (A Google Coral Mini and a PiEEG board). Here are the steps you need to follow to get started using the EEG capture, the Spindle detection software, and the TPU processing.

## Accessing the Google Coral

These first steps will help you set up an SSH connection to the device.
* Power up the board through the USB power port.
* Connect another USB cable to the OTG-port on the board and to your *linux* host machine. Follow the following steps to connect to the board through serial: 
    * `ls /dev/ttyMC*`
    * `screen /dev/ttyMC0` 
    If you see a message telling you that screen is busy, you can use `sudo lsof /dev/ttyMC0` and then retry the screen step.
    * Login to the board using default username and password: mendel
* Once you are logged in, you can now connect to you desired wifi network using nmtui.
* If you want to access the board through ssh (which is recommended for any sort of development):
    * On the serial console, open the `/etc/ssh/sshd_config` file. 
    * Scroll down to the `PasswordAuthenticated` line and change the 'no' to a 'yes'.
Once all of that is done, you should be able to ssh into your device, using either the ip address or the hostname. If some issues arise, make sure you are conneted to the same network.

## Dependencies

To install all dependencies to the original portiloop code, clone it and run the installation.sh script. 

## Jupyter notebook

