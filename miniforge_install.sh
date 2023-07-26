#!/bin/bash

# For this script to work, the repo must be located in the (mendel) home folder
# This script installs the Miniforge-based solution of the Portiloop software

echo "--- PORTILOOP V2 INSTALLATION SCRIPT (Miniforge version) ---"

cd ~

# echo "Installing dependencies..."
sudo apt-get update
sudo apt-get install -y python3-matplotlib python3-scipy python3-dev libasound2-dev

echo "Installing Miniforge..."
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b
rm Miniforge3-Linux-aarch64.sh

echo "Creating portiloop virtual environment..."
miniforge3/bin/conda create -n portiloop python=3.7 -y

echo "Installing latest pycoral and tflite-runtime..."
# pip3 uninstall tflite-runtime -y
# pip3 uninstall pycoral -y
wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl
wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp37-cp37m-linux_aarch64.whl
miniforge3/envs/portiloop/bin/pip3 install tflite_runtime-2.5.0.post1-cp37-cp37m-linux_aarch64.whl
miniforge3/envs/portiloop/bin/pip3 install pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl
rm tflite_runtime-2.5.0.post1-cp37-cp37m-linux_aarch64.whl
rm pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl

echo "Installing the Portiloop software [This may take a while]"
cd ~/portiloop-software
sudo apt-get install git-lfs
git lfs pull
~/miniforge3/envs/portiloop/bin/pip3 install notebook
~/miniforge3/envs/portiloop/bin/pip3 install -e .

cd ~

echo "Activating the widgets for the jupyter notebook..."
miniforge3/envs/portiloop/bin/jupyter nbextension enable --py widgetsnbextension

echo "Creating workspace directory..."
cd ~
mkdir workspace
mkdir workspace/edf_recording

echo "Copying files..."
cd ~/portiloop-software/portiloop/setup_files
sudo cp jupyter.service /etc/systemd/system/jupyter.service

echo "Reloading systemctl daemon..."
sudo systemctl daemon-reload
echo "Enabling jupyter service..."
sudo systemctl enable jupyter.service

# jupyter notebook --generate-config
echo "Launching jupyter notebook password manager..."
~/miniforge3/envs/portiloop/bin/jupyter notebook password

sudo cp asound.conf /etc/asound.conf
echo "All done! Please reboot the device."