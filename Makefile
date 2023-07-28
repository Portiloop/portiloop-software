all: miniforge

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
	sudo cp asound.conf /etc/asound.conf
	rm *.temp
	echo "All done! Please reboot the device."


clean:
	rm *.temp