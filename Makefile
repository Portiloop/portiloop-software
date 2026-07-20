all: miniforge

# all can be either "vanilla" or "miniforge"

# === Apt setup and AP creation ===

step_pre0.temp:
	echo "--- APT UPGRADING STEP ---"
	sudo apt upgrade -y
# 	echo "Disabling weston..."
# 	sudo systemctl disable weston
	touch step_pre0.temp

step_pre1.temp: step_pre0.temp
	echo "--- BOOT PARTITION FLASHING STEP ---"
# 	cd ~/portiloop-software/portiloop/setup_files
# 	echo "Downloading protected boot partition from GitHub..."
# 	wget https://github.com/Portiloop/portiloop-software/releases/download/v0.1.3/boot_ext4.img
# 	wget https://github.com/Portiloop/portiloop-software/releases/download/v0.1.3/fstab
	touch step_pre1.temp

step_pre2.temp: step_pre1.temp
	echo "Flashing boot partition..."
# 	sudo umount /boot
# 	cd ~/portiloop-software/portiloop/setup_files
# 	sudo dd if=boot_ext4.img of=/dev/mmcblk0p2 bs=4M status=progress
# 	sudo sync
# 	echo "Replacing fstab..."
# 	sudo cp /etc/fstab /etc/fstab.old
# 	sudo cp fstab /etc/fstab
# 	rm boot_ext4.img fstab
	touch step_pre2.temp

step_pre3.temp: step_pre2.temp
	echo "--- PORTILOOP V3 PRE-INSTALLATION STEP ---"

	cd ~/portiloop-software/portiloop/setup_files && sudo cp security.list /etc/apt/sources.list.d/security.list
	gpg --keyserver keyserver.ubuntu.com --recv-keys B53DC80D13EDEF05
	gpg --export --armor B53DC80D13EDEF05 | sudo apt-key add -
	gpg --keyserver keyserver.ubuntu.com --recv-keys C0BA5CE6DC6315A3
	gpg --export --armor C0BA5CE6DC6315A3 | sudo apt-key add -
	sudo apt-get --allow-releaseinfo-change update
	sudo apt-get update
	touch step_pre3.temp

step0.temp: step_pre3.temp
	echo "Creating acces point..."
	cd ~/portiloop-software && bash create_ap.sh
	touch step0.temp

# === miniforge pipeline ===

step1.temp: step0.temp
	echo "--- PORTILOOP V3 INSTALLATION (Miniforge version) ---"
	sudo apt-get install -y python3-matplotlib python3-scipy python3-dev libasound2-dev
	touch step1.temp

step2.temp: step1.temp
	echo "Installing Miniforge..."
	wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
	bash Miniforge3-Linux-aarch64.sh -b
	rm Miniforge3-Linux-aarch64.sh
	echo "Moving ~/miniforge3 directory to /opt/miniforge3..."
	sudo mv ~/miniforge3 /opt/.
	echo "Creating simlink..."
	ln -s /opt/miniforge3 ~/miniforge3
	touch step2.temp

step3.temp: step2.temp
	echo "Creating portiloop virtual environment..."
	~/miniforge3/bin/conda create -n portiloop python=3.7 -y
	touch step3.temp

step4.temp: step3.temp
	echo "Installing latest pycoral and tflite-runtime..."
	wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl
	wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp37-cp37m-linux_aarch64.whl
	~/miniforge3/envs/portiloop/bin/pip3 install tflite_runtime-2.5.0.post1-cp37-cp37m-linux_aarch64.whl
	~/miniforge3/envs/portiloop/bin/pip3 install pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl
	rm tflite_runtime-2.5.0.post1-cp37-cp37m-linux_aarch64.whl
	rm pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl
	touch step4.temp

step5.temp: step4.temp
	echo "Installing the Portiloop software [This may take a while]"
	cd ~/portiloop-software && sudo apt-get install git-lfs && git lfs pull && ~/miniforge3/envs/portiloop/bin/pip3 install notebook && ~/miniforge3/envs/portiloop/bin/pip3 install -e .
	echo "Activating the widgets for the jupyter notebook..."
	~/miniforge3/envs/portiloop/bin/jupyter nbextension enable --py widgetsnbextension
	echo "Creating workspace directory..."
	cd ~ && mkdir workspace && mkdir workspace/recordings
	echo "Copying files..."
	cd ~/portiloop-software/portiloop/setup_files && sudo cp asound.conf /etc/asound.conf
	cd ~/portiloop-software/portiloop/setup_files && sudo cp create_login_folder.service /etc/systemd/system/create_login_folder.service
	cd ~/portiloop-software/portiloop/setup_files && sudo cp miniforge_jupyter.service /etc/systemd/system/jupyter.service
	cd ~/portiloop-software/portiloop/setup_files && sudo cp simplegui.service /etc/systemd/system/simplegui.service
# 	cd ~/portiloop-software/portiloop/setup_files && sudo cp fix_headphone_jack.sh /usr/local/bin/fix_headphone_jack.sh
# 	sudo chmod +x /usr/local/bin/fix_headphone_jack.sh
# 	cd ~/portiloop-software/portiloop/setup_files && sudo cp fix_headphone_jack.service /etc/systemd/system/fix_headphone_jack.service
	touch step5.temp

step6.temp: step5.temp
	echo "Reloading systemctl daemon..."
	sudo systemctl daemon-reload
# 	echo "Enabling headphone jack fix service..."
# 	sudo systemctl enable fix_headphone_jack.service
	echo "Enabling manual login service..."
	sudo systemctl enable create_login_folder.service
	echo "Enabling jupyter service..."
	sudo systemctl enable jupyter.service
	echo "Enabling simple GUI service..."
	sudo systemctl enable simplegui.service
	touch step6.temp

step7.temp: step6.temp
	echo "Playing test sound to update ALSA:"
	echo "NOTE: THIS STEP MAY FAIL, JUST EXECUTE make AGAIN IF YOU GET AN ERROR."
	cd ~/portiloop-software/portiloop/sounds && aplay -Dplug:softvol stimulus.wav
	touch step7.temp

step8.temp: step7.temp
	echo "Editing FSTAB"
	echo "/dev/mmcblk2p1 /media/sd_card auto nofail,rw,user,exec,umask=000 0 2" | sudo tee -a /etc/fstab
	touch step8.temp

miniforge: step8.temp
	echo "Launching jupyter notebook password manager..."
	~/miniforge3/envs/portiloop/bin/jupyter notebook password
	rm *.temp
	echo "All done! Please reboot the device."


clean:
	rm *.temp
