#!/bin/bash

# For this script to work, the repo must be located in the (mendel) home folder

echo "--- PORTILOOP V2 INSTALLATION SCRIPT ---"

echo "Enter the desired access point wifi SSID:"
read portiloop_SSID
echo "Enter the desired access point wifi password:"
read portiloop_password

echo "The script will now update your system."

cd ~

sudo apt remove -y reportbug python3-reportbug
echo "Updating apt..."
sudo apt-get update

echo "Upgrading pip3..."
# sudo /usr/bin/python3 -m pip install --upgrade pip
pip3 install --upgrade pip

echo "Installing dependencies..."
sudo apt-get install -y python3-matplotlib python3-scipy python3-dev libasound2-dev jupyter-notebook jupyter
sudo apt-get install -y jupyter-nbextension-jupyter-js-widgets

echo "Installing latest pycoral and tflite-runtime..."
# pip3 uninstall tflite-runtime -y
# pip3 uninstall pycoral -y
wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl
wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp37-cp37m-linux_aarch64.whl
pip3 install tflite_runtime-2.5.0.post1-cp37-cp37m-linux_aarch64.whl --user
pip3 install pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl --user
rm tflite_runtime-2.5.0.post1-cp37-cp37m-linux_aarch64.whl
rm pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl

echo "Installing the Portiloop software [This may take a while]"
cd ~/portiloop-software
pip3 install -e .  --user

echo "Activating the widgets for the jupyter notebook..."
jupyter nbextension enable --py widgetsnbextension

echo "Creating workspace directory..."
cd ~
mkdir workspace
mkdir workspace/edf_recording

echo "Setting up access point..."
sudo apt-get install hostapd dnsmasq
cd ~/portiloop-software/portiloop/setup_files
sudo cp create_ap0.sh /usr/local/bin/create_ap0.sh
sudo chmod +x /usr/local/bin/create_ap0.sh
sudo bash /usr/local/bin/create_ap0.sh
nmcli device set ap0 managed no
sudo cp unmanaged.conf /etc/NetworkManager/conf.d/unmanaged.conf
sudo cp create_ap.service /etc/systemd/system/create_ap.service
sudo cp sysctl.conf /etc/sysctl.conf

sudo touch /etc/hostapd/hostapd.conf
sudo truncate -s 0 /etc/hostapd/hostapd.conf
sudo sh -c 'echo "interface=ap0" >> /etc/hostapd/hostapd.conf'
sudo sh -c 'echo "driver=nl80211" >> /etc/hostapd/hostapd.conf'
sudo -E sh -c "echo \"ssid=${portiloop_SSID}\" >> /etc/hostapd/hostapd.conf"
sudo sh -c 'echo "hw_mode=g" >> /etc/hostapd/hostapd.conf'
sudo sh -c 'echo "channel=6" >> /etc/hostapd/hostapd.conf'
sudo sh -c 'echo "wpa=2" >> /etc/hostapd/hostapd.conf'
sudo -E sh -c "echo \"wpa_passphrase=${portiloop_password}\" >> /etc/hostapd/hostapd.conf"
sudo sh -c 'echo "wpa_key_mgmt=WPA-PSK" >> /etc/hostapd/hostapd.conf'
sudo sh -c 'echo "wpa_pairwise=TKIP CCMP" >> /etc/hostapd/hostapd.conf'
sudo sh -c 'echo "rsn_pairwise=CCMP" >> /etc/hostapd/hostapd.conf'
sudo sh -c 'echo "auth_algs=1" >> /etc/hostapd/hostapd.conf'
sudo sh -c 'echo "macaddr_acl=0" >> /etc/hostapd/hostapd.conf'

sudo cp hostapd /etc/default/hostapd
sudo systemctl unmask hostapd
sudo cp dnsmasq.conf /etc/dnsmasq.conf
sudo cp setup_tables.sh /usr/local/bin/setup_tables.sh
sudo cp setup_tables.service /etc/systemd/system/setup_tables.service
sudo chmod +x /usr/local/bin/setup_tables.sh
sudo cp jupyter.service /etc/systemd/system/jupyter.service

echo "Reloading systemctl daemon..."
sudo systemctl daemon-reload
echo "Enabling AP service..."
sudo systemctl enable create_ap.service
echo "Enabling hostapd service..."
sudo systemctl enable hostapd.service
echo "Enabling dnsmask service..."
sudo systemctl enable dnsmasq.service
echo "Enabling setup_tables service..."
sudo systemctl enable setup_tables.service
echo "Enabling jupyter service..."
sudo systemctl enable jupyter.service

# jupyter notebook --generate-config
echo "Launching jupyter notebook password manager..."
jupyter notebook password

echo "All done!"