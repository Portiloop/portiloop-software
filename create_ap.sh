#!/bin/bash

set -e

echo "--- PORTILOOP V2 ACCESS POINT SETUP ---"

echo "Enter the desired access point wifi SSID:"
read portiloop_SSID
echo "Enter the desired access point wifi password:"
read portiloop_password
echo "Enter the desired access point wifi channel:"
read portiloop_channel

cd ~
echo "Preparing apt..."
export LC_ALL="en_US.UTF-8"
sudo apt remove -y reportbug python3-reportbug
gpg --keyserver keyserver.ubuntu.com --recv-keys B53DC80D13EDEF05
gpg --export --armor B53DC80D13EDEF05 | sudo apt-key add -
sudo apt-get --allow-releaseinfo-change-suite update

echo "Creating access point interface..."
sudo apt-get install hostapd dnsmasq

cd ~/portiloop-software/portiloop/setup_files
sudo cp create_ap0.sh /usr/local/bin/create_ap0.sh
sudo chmod +x /usr/local/bin/create_ap0.sh
sudo cp unmanaged.conf /etc/NetworkManager/conf.d/unmanaged.conf
sudo cp create_ap.service /etc/systemd/system/create_ap.service
sudo cp sysctl.conf /etc/sysctl.conf

sudo touch /etc/hostapd/hostapd.conf
sudo truncate -s 0 /etc/hostapd/hostapd.conf
sudo sh -c 'echo "interface=ap0" >> /etc/hostapd/hostapd.conf'
sudo sh -c 'echo "driver=nl80211" >> /etc/hostapd/hostapd.conf'
sudo -E sh -c "echo \"ssid=${portiloop_SSID}\" >> /etc/hostapd/hostapd.conf"
sudo sh -c 'echo "hw_mode=g" >> /etc/hostapd/hostapd.conf'
sudo -E sh -c "echo \"channel=${portiloop_channel}\" >> /etc/hostapd/hostapd.conf"
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

cd ~/portiloop-software
echo "Interface created, please reboot the device and move on to the installation script."
