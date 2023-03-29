# Portiloop V2

You've just got your hands on the hardware for the Portiloop V2 (A Google Coral Mini and a PiEEG board). Here are the steps you need to follow to get started using the EEG capture, the Spindle detection software, and the TPU processing.

## Accessing the Google Coral

These first steps will help you set up an SSH connection to the device.

- Power up the board through the USB power port.
- Connect another USB cable to the OTG-port on the board and to your _linux_ host machine. Follow the following steps to connect to the board through serial:
  - `ls /dev/ttyMC*`
  - `screen /dev/ttyMC0`
    If you see a message telling you that screen is busy, you can use `sudo lsof /dev/ttyMC0` and then retry the screen step.
  - Login to the board using default username and password: mendel
- Once you are logged in, you can now connect to you desired wifi network using nmtui.
- If you want to access the board through ssh (which is recommended for any sort of development):
  _ On the serial console, open the `/etc/ssh/sshd_config` file.
  _ Scroll down to the `PasswordAuthenticated` line and change the 'no' to a 'yes'.
  Once all of that is done, you should be able to ssh into your device, using either the ip address or the hostname. If some issues arise, make sure you are connected to the same network.

## Dependencies

To install all dependencies, clone it and run the installation.sh script. This script takes care of all the installations for you so it may take a while (~25 minutes).

## Setting up the Access Point

### 1. Download dependencies for access point

To set up an access point, you will need to install a few dependencies. To install them, you can use the following command:

```bash
sudo apt-get update && sudo apt-get install hostapd dnsmasq
```

This will update your system's package list and install the `hostapd` and `dnsmasq` packages.

### 2. Set up the interface ap0

Next, you will need to set up a systemd service to configure and enable the `ap0` interface.

First, we can create a script to create the interface using `sudo nano /usr/local/bin/create_ap0.sh`. The script should contain the following content:

```bash
!/bin/bash

# Delete the existing p2p0 interface
sudo iw dev p2p0 del

# Reload the Network Manager utility
sudo systemctl restart NetworkManager

# Create a new ap0 interface in AP mode
sudo iw phy phy1 interface add ap0 type __ap

# Disable power management for the ap0 interface
sudo iw dev ap0 set power_save off

# Reload the Network Manager utility again
sudo systemctl restart NetworkManager

# Get an IPV4 address for the server
sudo ifconfig ap0 192.168.4.1 up
```

To avoid configuration issues, we need to tell NetworkManager to ignore this interface. Create a file called `/etc/NetworkManager/conf.d/unmanaged.conf`. In this file, write the following:

```ini
[keyfile]
unmanaged-devices=interface-name:ap0
```

To make sure this starts works everytime we turn the Portiloop on, we need to create a new service. First, you can create a new service file at `/etc/systemd/system/create_ap.service` with the following content:

```ini
[Unit]
Description=Create The Access Point for the coral
Before=hostapd.service dnsmasq.service

[Service]
Type=simple
ExecStart=/usr/local/bin/create_ap0.sh

[Install]
WantedBy=multi-user.target
```

This service file specifies that it should run the `create_ap0.sh` script once on boot before the hostapd and dnsmasq services start.

### 3. Configure Hostapd

Hostapd is the software that will create the wireless access point. First, you will need to open the in `/etc/sysctl.conf` file and change the line for ip_forwarding to `net.ipv4.ip_forward=1`.

Next, you will need to create a configuration file at `/etc/hostapd/hostapd.conf` with the following content:

```ini
interface=ap0
driver=nl80211
ssid=YOUR-SSID-HERE
hw_mode=g
channel=6
wpa=2
wpa_passphrase=YOUR-PASSWORD-HERE
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP CCMP
rsn_pairwise=CCMP
auth_algs=1
macaddr_acl=0
```

This configuration file specifies the `ap0` interface, the SSID and password for the access point, and the encryption type to use. Make sure to replace `YOUR-SSID-HERE` and `YOUR-PASSWORD-HERE` with your own values. You now need to specify to hostapd which configuration file to do. Open the hostapd configuration file using `sudo nano /etc/default/hostapd`. Uncomment the DAEMON_CONF line and set it to the path of the configuration file you just created:
`DAEMON_CONF="/etc/hostapd/hostapd.conf"`.

### 4. Configure dnsmasq

Dnsmasq is the software that will provide DHCP and DNS services for the access point. Start by opening the dnsmasq configuration file with `sudo nano /etc/dnsmasq.conf`. Add the following content at the top of the file:

```ini

# Configuration for Access Point
interface=ap0
dhcp-range=192.168.4.2,192.168.4.20,255.255.255.0,24h
dhcp-option=3,192.168.4.2
dhcp-option=6,192.168.4.1
server=8.8.8.8
```

This configuration file specifies the `ap0` interface, the range of IP addresses to assign to clients, and the DNS server to use. Note that the IP address of the `dhcp-option=6,...` should be the same as the IP address set in step 2.

### 5. Start Systemd services

To make sure that everything happens on startup, we need to enable all services. Execute the following commands:

```bash
sudo systemctl enable create_ap.service
sudo systemctl enable hostapd.service
sudo systemctl enable dnsmasq.service
```

## Jupyter notebook

To access the portiloop easily, we recommend setting up a jupyter notebook server which will be available from any browser. To set up the Jupyter server using a systemd service, follow these steps:

1. On the command line, type `jupyter notebook password`. This will show a prompt where you can enter the desired password.
2. Create a new systemd service file using the command `sudo nano /etc/systemd/system/jupyter.service`.
3. Add the following lines to the service file:

```ini
[Unit]
Description=Jupyter Notebook Server
After=create_ap.service
After=hostapd.service
After=dnsmasq.service

[Service]
Type=simple
ExecStart=/bin/bash -c "/usr/bin/jupyter notebook --no-browser --ip 192.168.4.1 --port 8080 --notebook-dir=/home/mendel"
User=mendel
Group=mendel
Restart=on-failure
RestartSec=60s

[Install]
WantedBy=multi-user.target
```

4. Save and close the file.
5. Reload the systemd daemon to load the new service file: `sudo systemctl daemon-reload`.
6. Start the Jupyter service: `sudo systemctl start jupyter`.
7. Check the status of the service: `sudo systemctl status jupyter`. If everything is set up correctly, you should see a message indicating that the service is active and running.
8. To make sure the service starts automatically on boot, enable it: `sudo systemctl enable jupyter`.

That's it! Your Jupyter server should now be up and running, listening on IP address 192.168.4.1 and port 8080, and automatically starting whenever the system boots up. You can now access it by typing 192.168.4.1:8080 in your browser. This should lead you to a login page where you'll be prompted for your password. If any issue arise, try with a different web browser.
