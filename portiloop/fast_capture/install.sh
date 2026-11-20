#!/bin/bash

# Installs the fast acquisition (2 to 16 kSPS, see README.md) on this Portiloop: builds the capture program and the
# FFR player, and installs the spimap kernel module (loaded at boot) and its udev rules. Run it on the Portiloop, as
# mendel:
#   bash ~/portiloop-software/portiloop/fast_capture/install.sh
# Then restart the Simple GUI (sudo systemctl restart simplegui) to get the "Acquisition only (fast)", "FFR (fast)" and
# "Click ABR (fast)" pipelines.

KERNEL=4.19.125-mtk  # spimap.ko was built for this kernel

info() { echo -e "\033[1;34m==>\033[0m $*"; }
fail() { echo -e "\033[1;31mERROR:\033[0m $*"; exit 1; }

[[ $EUID -eq 0 ]] && fail "Run this script as mendel, without sudo."
cd "$(dirname "$(readlink -f "$0")")" || fail "Could not open the fast_capture folder."

[[ $(uname -r) == "$KERNEL" ]] || fail "spimap.ko was built for $KERNEL, this Portiloop runs $(uname -r): rebuild it (README.md)."
# same vermagic as the kernel's own modules (CONFIG_MODVERSIONS is off and modules are not signed)
expected=$(/sbin/modinfo -F vermagic spidev 2> /dev/null)
actual=$(/sbin/modinfo -F vermagic spimap/spimap.ko)
[[ -n $expected && $actual == "$expected" ]] || fail "spimap.ko ($actual) does not match the kernel modules ($expected)."

info "Building the capture program..."
gcc -O2 -Wall -pthread -o adscapture adscapture.c -lm || fail "Could not build adscapture."
info "Building the FFR player..."
gcc -O2 -Wall -o ffrplay ffrplay.c -lasound -lm || fail "Could not build ffrplay (libasound2-dev is needed)."

info "Installing the spimap module and its udev rules (your password may be asked)..."
sudo install -D -m 644 spimap/spimap.ko "/lib/modules/$KERNEL/extra/spimap.ko" || fail "Could not install spimap.ko."
sudo /sbin/depmod -a "$KERNEL" || fail "depmod failed."
echo spimap | sudo tee /etc/modules-load.d/portiloop-spimap.conf > /dev/null || fail "Could not install the module loading."
sudo install -m 644 spimap.rules /etc/udev/rules.d/99-portiloop-spimap.rules || fail "Could not install the udev rules."
sudo udevadm control --reload
sudo /sbin/modprobe spimap || fail "Could not load the spimap module."
sudo udevadm trigger --action=add --subsystem-match=misc --sysname-match=spimap
sudo udevadm trigger --action=add --subsystem-match=platform --sysname-match=1100c000.spi
sudo udevadm settle

ls -l /dev/spimap
echo "SPI controller power/control: $(cat /sys/devices/platform/soc/1100c000.spi/power/control)"
[[ -r /dev/spimap && -w /dev/spimap ]] || fail "/dev/spimap is not accessible to $(whoami)."
[[ $(cat /sys/devices/platform/soc/1100c000.spi/power/control) == on ]] || fail "power/control is not 'on'."
info "Done. Restart the Simple GUI to use the \"Acquisition only (fast)\", \"FFR (fast)\" and \"Click ABR (fast)\" pipelines: sudo systemctl restart simplegui"
