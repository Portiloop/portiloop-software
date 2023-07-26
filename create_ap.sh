 #!/bin/bash

 echo "Creating access point interface..."
 sudo apt-get install hostapd dnsmasq
 cd ~/portiloop-software/portiloop/setup_files
 sudo cp create_ap0.sh /usr/local/bin/create_ap0.sh
 sudo chmod +x /usr/local/bin/create_ap0.sh
 sudo bash /usr/local/bin/create_ap0.sh
 nmcli device set ap0 managed no
 cd ~/portiloop-software
echo "Interface created, please reboot the device and move on to the installation script."
