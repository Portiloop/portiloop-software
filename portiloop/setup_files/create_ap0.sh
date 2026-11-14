#!/bin/bash

set -e

if iw list | grep -q "Wiphy phy1"; then
    # Dual-PHY board (e.g., Coral Dev Board Mini): use phy1 for ap0
    phy1_interface=$(iw dev | awk '/phy#1/ {getline; print $2}')
    if [[ "$phy1_interface" == "ap0" ]]; then
        echo "ap0 already set up, not running script..."
    else
        echo "$phy1_interface"
        if [[ -n "$phy1_interface" ]]; then
            /sbin/iw dev "$phy1_interface" del
        fi
        systemctl restart NetworkManager
        /sbin/iw phy phy1 interface add ap0 type __ap
        /sbin/iw dev ap0 set power_save off || true
        systemctl restart NetworkManager
        ip addr replace 192.168.4.1/24 dev ap0
        ip link set ap0 up
    fi
else
    # Single-PHY board (e.g., RDK X5): add ap0 alongside wlan0 on phy0
    if ip link show ap0 >/dev/null 2>&1; then
        echo "ap0 already exists, ensuring IP and unmanaged state..."
    else
        /sbin/iw phy phy0 interface add ap0 type __ap
    fi
    /sbin/iw dev ap0 set power_save off || true
    nmcli device set ap0 managed no || true
    ip addr replace 192.168.4.1/24 dev ap0
    ip link set ap0 up
fi

nmcli device set ap0 managed no || true
