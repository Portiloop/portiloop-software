all: miniforge

# all can be either "vanilla" or "miniforge"

# === miniforge pipeline ===

step1.temp:
	echo "--- PORTILOOP V2 INSTALLATION SCRIPT (Miniforge version) ---"
	sudo apt-get update
	sudo apt-get install -y python3-matplotlib python3-scipy python3-dev libasound2-dev
	touch step1.temp

step2.temp: step1.temp
	echo "Installing Miniforge..."
	wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
	bash Miniforge3-Linux-aarch64.sh -b
	rm Miniforge3-Linux-aarch64.sh
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
	cd ~ && mkdir workspace && mkdir workspace/edf_recording
	echo "Copying files..."
	cd ~/portiloop-software/portiloop/setup_files && sudo cp miniforge_jupyter.service /etc/systemd/system/jupyter.service
	touch step5.temp

step6.temp: step5.temp
	echo "Reloading systemctl daemon..."
	sudo systemctl daemon-reload
	echo "Enabling jupyter service..."
	sudo systemctl enable jupyter.service
	touch step6.temp

miniforge: step6.temp
	echo "Launching jupyter notebook password manager..."
	~/miniforge3/envs/portiloop/bin/jupyter notebook password
	cd ~/portiloop-software/portiloop/setup_files && sudo cp asound.conf /etc/asound.conf
	rm *.temp
	echo "Playing test sound to update ALSA:"
	cd ~/portiloop-software/portiloop/sounds && aplay -Dplug:softvol stimulus.wav
	echo "All done! Please reboot the device."


# === vanilla pipeline ===

vstep1.temp:
	echo "--- PORTILOOP V2 INSTALLATION SCRIPT ---"
	echo "The script will now update your system."
	echo "Preparing apt..."
	export LC_ALL="en_US.UTF-8"
	sudo apt remove -y reportbug python3-reportbug
	gpg --keyserver keyserver.ubuntu.com --recv-keys B53DC80D13EDEF05
	gpg --export --armor B53DC80D13EDEF05 | sudo apt-key add -
	echo "Updating apt..."
	sudo apt-get --allow-releaseinfo-change-suite update
	touch vstep1.temp

vstep2.temp: vstep1.temp
	echo "Upgrading pip3..."
	# sudo /usr/bin/python3 -m pip install --upgrade pip
	pip3 install --upgrade pip --user
	echo "pip3 is now at the following location:"
	which pip3
	@if [ $$(which pip3) = "/home/mendel/.local/bin/pip3" ]; then \
		echo "Installed pip3 path is correct"; \
	else \
		echo "Installed pip3 path is incorrect, will now exit with an error."; \
		echo "This is fine, please reboot the device and execute make again."; \
		exit 1; \
	fi
	touch vstep2.temp

vstep3.temp: vstep2.temp
	echo "Installing dependencies..."
	sudo apt-get install -y python3-matplotlib python3-scipy python3-dev libasound2-dev jupyter-notebook jupyter
	sudo apt-get install -y jupyter-nbextension-jupyter-js-widgets
	touch vstep3.temp

vstep4.temp: vstep3.temp
	echo "Installing latest pycoral and tflite-runtime..."
	wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl
	wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp37-cp37m-linux_aarch64.whl
	pip3 install tflite_runtime-2.5.0.post1-cp37-cp37m-linux_aarch64.whl --user
	pip3 install pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl --user
	rm tflite_runtime-2.5.0.post1-cp37-cp37m-linux_aarch64.whl
	rm pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl
	touch vstep4.temp

vstep5.temp: vstep4.temp
	echo "Installing the Portiloop software [This may take a while]"
	cd ~/portiloop-software && sudo apt-get install git-lfs && git lfs pull && pip3 install -e . --user
	echo "Activating the widgets for the jupyter notebook..."
	jupyter nbextension enable --py widgetsnbextension
	echo "Creating workspace directory..."
	cd ~ && mkdir workspace && mkdir workspace/edf_recording
	echo "Copying files..."
	cd ~/portiloop-software/portiloop/setup_files && sudo cp jupyter.service /etc/systemd/system/jupyter.service
	touch vstep5.temp

vstep6.temp: vstep5.temp
	echo "Reloading systemctl daemon..."
	sudo systemctl daemon-reload
	echo "Enabling jupyter service..."
	sudo systemctl enable jupyter.service
	touch vstep6.temp

vanilla: vstep6.temp
	echo "Launching jupyter notebook password manager..."
	jupyter notebook password
	cd ~/portiloop-software/portiloop/setup_files && sudo cp asound.conf /etc/asound.conf
	rm *.temp
	echo "Playing test sound to update ALSA:"
	cd ~/portiloop-software/portiloop/sounds && aplay -Dplug:softvol stimulus.wav
	echo "All done! Please reboot the device."


clean:
	rm *.temp