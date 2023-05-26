#!/bin/bash

echo "Telling kernel to turn on ipv4 ip_forwarding"
echo 1 > /proc/sys/net/ipv4/ip_forward
echo "Done. Setting up iptables rules to allow FORWARDING"

DOWNSTREAM=ap0 # ap0 is client network (running hostapd)
UPSTREAM=wlan0 # upstream network (internet)

# Allow IP Masquerading (NAT) of packets from clients (downstream) to upstream network (internet)
iptables -t nat -A POSTROUTING -o $UPSTREAM -j MASQUERADE

# Forward packets from downstream clients to the upstream internet
iptables -A FORWARD -i $DOWNSTREAM -o $UPSTREAM -j ACCEPT

# Forward packers from the internet to clients IF THE CONNECTION IS ALREADY OPEN!
iptables -A FORWARD -i $UPSTREAM  -o $DOWNSTREAM -m state --state RELATED,ESTABLISHED -j ACCEPT

# Setup the external DNS server
iptables -t nat -A PREROUTING -i $DOWNSTREAM -p udp --dport 53 -j DNAT --to-destination 8.8.8.8:53

echo "Done setting up iptables rules. Forwarding enabled"