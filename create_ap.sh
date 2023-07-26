 #!/bin/bash

echo "Creating access point interface...
"
sudo apt-get update
sudo apt-get install hostapd dnsmasq

cd ~/portiloop-software/portiloop/setup_files
sudo cp create_ap0.sh /usr/local/bin/create_ap0.sh
sudo chmod +x /usr/local/bin/create_ap0.sh
# sudo bash /usr/local/bin/create_ap0.sh
# nmcli device set ap0 managed no
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

cd ~/portiloop-software
echo "Interface created, please reboot the device and move on to the installation script."
