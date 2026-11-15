REPO_DIR := $(CURDIR)
SETUP_DIR := $(REPO_DIR)/portiloop/setup_files
VENV_DIR := $(HOME)/portiloop-venv
USER_NAME := $(shell id -un)
USER_UID := $(shell id -u)
IS_RDKX5 := $(shell grep -qiE "rdk x5|d-robotics" /sys/firmware/devicetree/base/model 2>/dev/null && echo 1 || echo 0)

ifeq ($(IS_RDKX5),1)
all: rdkx5
else
all: miniforge
endif

# all can be either "rdkx5" or "miniforge"

# === RDK X5 installation pipeline ===

rdkx5_deps.temp:
	@echo "--- RDK X5 SYSTEM DEPENDENCIES ---"
	sudo apt-get update
	sudo apt-get install -y python3-venv python3-dev python3-matplotlib python3-scipy libasound2-dev git-lfs exfatprogs iw
	touch rdkx5_deps.temp

rdkx5_ap.temp: rdkx5_deps.temp
	@echo "--- RDK X5 ACCESS POINT SETUP ---"
	cd $(REPO_DIR) && bash create_ap.sh
	touch rdkx5_ap.temp

rdkx5_venv.temp: rdkx5_ap.temp
	@echo "--- RDK X5 PYTHON VENV & PORTILOOP INSTALL ---"
	python3 -m venv --system-site-packages $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip wheel
	$(VENV_DIR)/bin/pip install notebook ipywidgets
	cd $(REPO_DIR) && (git lfs pull || true) && $(VENV_DIR)/bin/pip install -e .
	cd $(REPO_DIR)/portiloop/fast_capture && bash install.sh
	mkdir -p $(HOME)/workspace/recordings
	touch rdkx5_venv.temp

rdkx5_services.temp: rdkx5_venv.temp
	@echo "--- RDK X5 SERVICES & ALSA SETUP ---"
	sudo cp $(SETUP_DIR)/asound.conf /etc/asound.conf
	sed "s/^mendel /$(USER_NAME) /" $(SETUP_DIR)/limits.conf | sudo tee /etc/security/limits.d/portiloop.conf > /dev/null
	sed -e "s/mendel/$(USER_NAME)/g" -e "s/1000/$(USER_UID)/g" $(SETUP_DIR)/create_login_folder.service | sudo tee /etc/systemd/system/create_login_folder.service > /dev/null
	sed -e "s|/home/mendel/miniforge3/bin/conda run -n portiloop jupyter|$(VENV_DIR)/bin/jupyter|g" \
	    -e "s|/home/mendel|$(HOME)|g" \
	    -e "s/=mendel/=$(USER_NAME)/g" \
	    $(SETUP_DIR)/miniforge_jupyter.service | sudo tee /etc/systemd/system/jupyter.service > /dev/null
	sed -e "s|~/miniforge3/envs/portiloop/bin/python|$(VENV_DIR)/bin/python|g" \
	    -e "s|/home/mendel/portiloop-software|$(REPO_DIR)|g" \
	    -e "s|/home/mendel|$(HOME)|g" \
	    -e "s/=mendel/=$(USER_NAME)/g" \
	    $(SETUP_DIR)/simplegui.service | sudo tee /etc/systemd/system/simplegui.service > /dev/null
	sudo systemctl daemon-reload
	sudo systemctl enable create_login_folder.service jupyter.service simplegui.service
	touch rdkx5_services.temp

rdkx5: rdkx5_services.temp
	@echo "Playing test sound to update ALSA..."
	aplay -Dplug:softvol $(REPO_DIR)/portiloop/sounds/stimulus.wav || true
	@if [ "$(SKIP_JUPYTER_PW)" != "1" ]; then \
		echo "Launching jupyter notebook password manager..."; \
		$(VENV_DIR)/bin/jupyter notebook password; \
	fi
	rm -f *.temp
	@echo "All done! Please reboot the device."

# === Coral / Miniforge pipeline ===

step_pre0.temp:
	echo "--- APT UPGRADING STEP ---"
	sudo apt upgrade -y
# 	echo "Disabling weston..."  # Don't: this causes sound issues
# 	sudo systemctl disable weston
	touch step_pre0.temp

step_pre1.temp: step_pre0.temp
	echo "--- BOOT PARTITION FLASHING STEP ---"
	cd ~/portiloop-software/portiloop/setup_files
	echo "Downloading protected boot partition from GitHub..."
	wget https://github.com/Portiloop/portiloop-software/releases/download/v0.1.3/boot_ext4.img
	wget https://github.com/Portiloop/portiloop-software/releases/download/v0.1.3/fstab
	touch step_pre1.temp

step_pre2.temp: step_pre1.temp
	echo "Flashing boot partition..."
	sudo umount /boot
	cd ~/portiloop-software/portiloop/setup_files
	sudo dd if=boot_ext4.img of=/dev/mmcblk0p2 bs=4M status=progress
	sudo sync
	echo "Replacing fstab..."
	sudo cp /etc/fstab /etc/fstab.old
	sudo cp fstab /etc/fstab
	rm boot_ext4.img fstab
	touch step_pre2.temp

step_pre3.temp: step_pre2.temp
	echo "--- PORTILOOP V3 PRE-INSTALLATION STEP ---"

	cd ~/portiloop-software/portiloop/setup_files && sudo cp security.list /etc/apt/sources.list.d/security.list
	gpg --keyserver keyserver.ubuntu.com --recv-keys B53DC80D13EDEF05
	gpg --export --armor B53DC80D13EDEF05 | sudo apt-key add -
	gpg --keyserver keyserver.ubuntu.com --recv-keys C0BA5CE6DC6315A3
	gpg --export --armor C0BA5CE6DC6315A3 | sudo apt-key add -
	sudo apt-get --allow-releaseinfo-change update
	sudo apt update
	sudo apt install -y exfat-fuse exfat-utils
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
	cd ~/portiloop-software/portiloop/setup_files && sudo cp limits.conf /etc/security/limits.d/portiloop.conf
	cd ~/portiloop-software/portiloop/setup_files && sudo cp sd_card.rules /etc/udev/rules.d/99-portiloop-sd-card.rules
	touch step5.temp

step6.temp: step5.temp
	echo "Reloading systemctl daemon..."
	sudo systemctl daemon-reload
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
