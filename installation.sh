#!/bin/bash

# For this script to work, the repo must be located in the (mendel) home folder

echo "--- PORTILOOP V2 INSTALLATION SCRIPT ---"

echo "The script will now update your system."

cd ~

echo "Updating apt..."
sudo apt-get update

echo "Upgrading pip3..."
python3 -m pip install --upgrade pip

echo "Installing dependencies..."
sudo apt-get install -y python3-matplotlib python3-scipy python3-dev libasound2-dev jupyter-notebook jupyter

echo "Installing latest pycoral and tflite-runtime..."
wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl
wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp37-cp37m-linux_aarch64.whl
pip3 uninstall tflite-runtime
pip3 uninstall pycoral
pip3 install tflite_runtime-2.5.0.post1-cp37-cp37m-linux_aarch64.whl
pip3 install pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl

echo "Installing the Portiloop software [This may take a while]"
cd ~/portiloop-software
pip3 install -e .

echo "Activating the widgets for the jupyter notebook..."
jupyter nbextension enable --py widgetsnbextension

echo "Creating workspace directory..."
cd ~
mkdir workspace
mkdir workspace/edf_recording

echo "Setting up access point..."
sudo apt-get install hostapd dnsmasq
cd ~/portiloop-software/portillop/setup_files
sudo cp create_ap0.sh /usr/local/bin/create_ap0.sh
sudo chmod +x /usr/local/bin/create_ap0.sh
sudo bash /usr/local/bin/create_ap0.sh
nmcli device set ap0 managed no
sudo cp unmanaged.conf /etc/NetworkManager/conf.d/unmanaged.conf
sudo cp create_ap.service /etc/systemd/system/create_ap.service

# TODO copy sysctl.conf here

echo "Enter access point wifi SSID:"
read portiloop_SSID
echo $portiloop_SSID
echo "Enter access point wifi password:"
read portiloop_password
echo $portiloop_password

sudo touch /etc/hostapd/hostapd.conf
sudo truncate -s 0 /etc/hostapd/hostapd.conf
sudo sh -c 'echo "interface=ap0" >> /etc/hostapd/hostapd.conf'
sudo sh -c 'echo "driver=nl80211" >> /etc/hostapd/hostapd.conf'
sudo sh -c 'echo "ssid=$portiloop_SSID" >> /etc/hostapd/hostapd.conf'
sudo sh -c 'echo "hw_mode=g" >> /etc/hostapd/hostapd.conf'
sudo sh -c 'echo "channel=6" >> /etc/hostapd/hostapd.conf'
sudo sh -c 'echo "wpa=2" >> /etc/hostapd/hostapd.conf'
sudo sh -c 'echo "wpa_passphrase=$portiloop_password" >> /etc/hostapd/hostapd.conf'
sudo sh -c 'echo "wpa_key_mgmt=WPA-PSK" >> /etc/hostapd/hostapd.conf'
sudo sh -c 'echo "wpa_pairwise=TKIP CCMP" >> /etc/hostapd/hostapd.conf'
sudo sh -c 'echo "rsn_pairwise=CCMP" >> /etc/hostapd/hostapd.conf'
sudo sh -c 'echo "auth_algs=1" >> /etc/hostapd/hostapd.conf'
sudo sh -c 'echo "macaddr_acl=0" >> /etc/hostapd/hostapd.conf'
