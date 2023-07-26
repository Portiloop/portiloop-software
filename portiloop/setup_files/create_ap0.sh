#!/bin/bash

# Get the name of the interface on phy1
# phy1_interface=$(sudo iw dev | awk '/phy#1/ {getline; print $2}')
phy1_interface=$(iw dev | awk '/phy#1/ {getline; print $2}')

# Check if the interface name is p2p0
if [[ $phy1_interface == "ap0" ]]; then
    echo "ap0 already set up, not running script..."
else
    echo $phy1_interface
    # Delete the existing p2p0 interface
    /sbin/iw dev $phy1_interface del

    # Reload the Network Manager utility
    systemctl restart NetworkManager

    # Create a new ap0 interface in AP mode
    /sbin/iw phy phy1 interface add ap0 type __ap

    # Disable power management for the ap0 interface
    /sbin/iw dev ap0 set power_save off

    # Reload the Network Manager utility again
    systemctl restart NetworkManager

    # Get an IPV4 address for the server
    ifconfig ap0 192.168.4.1 up
fi
