# Build scr
sudo mkimage -A arm -T script -O linux -d boot.txt /boot/boot.scr

# Edit device tree overlay
sudo vim /boot/portiloop.dts && sudo dtc -I dts -O dtb -o /boot/portiloop.dtbo /boot/portiloop.dts
