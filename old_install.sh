#!/bin/bash

# For this script to work, the repo must be located in the (mendel) home folder

echo "--- PORTILOOP V2 INSTALLATION SCRIPT ---"

echo "The script will now update your system."

cd ~

echo "Preparing apt..."
export LC_ALL="en_US.UTF-8"
sudo apt remove -y reportbug python3-reportbug
gpg --keyserver keyserver.ubuntu.com --recv-keys B53DC80D13EDEF05
gpg --export --armor B53DC80D13EDEF05 | sudo apt-key add -

echo "Updating apt..."
sudo apt-get --allow-releaseinfo-change-suite update

echo "Upgrading pip3..."
# sudo /usr/bin/python3 -m pip install --upgrade pip
pip3 install --upgrade pip --user
export PATH="$PATH:/home/mendel/.local/bin"
echo "pip3 is now at the following location:"
which pip3
installed_path=$(which pip3)
expected_path="/home/mendel/.local/bin/pip3"
if [ "$installed_path" = "$expected_path" ]; then
    echo "Installed pip3 path is correct: $installed_path"
else
    echo "Installed pip3 path is incorrect. Expected: $expected_path, Actual: $installed_path"
    echo "Please reboot the device and launch installation.sh again."
    exit 1
fi

echo "Installing dependencies..."
sudo apt-get install -y python3-matplotlib python3-scipy python3-dev libasound2-dev jupyter-notebook jupyter
sudo apt-get install -y jupyter-nbextension-jupyter-js-widgets

echo "Installing latest pycoral and tflite-runtime..."
# pip3 uninstall tflite-runtime -y
# pip3 uninstall pycoral -y
wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl
wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp37-cp37m-linux_aarch64.whl
pip3 install tflite_runtime-2.5.0.post1-cp37-cp37m-linux_aarch64.whl --user
pip3 install pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl --user
rm tflite_runtime-2.5.0.post1-cp37-cp37m-linux_aarch64.whl
rm pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl

echo "Installing the Portiloop software [This may take a while]"
cd ~/portiloop-software
sudo apt-get install git-lfs
git lfs pull
pip3 install -e . --user

echo "Activating the widgets for the jupyter notebook..."
jupyter nbextension enable --py widgetsnbextension

echo "Creating workspace directory..."
cd ~
mkdir workspace
mkdir workspace/edf_recording

echo "Copying files..."
cd ~/portiloop-software/portiloop/setup_files
sudo cp miniforge_jupyter.service /etc/systemd/system/jupyter.service

echo "Reloading systemctl daemon..."
sudo systemctl daemon-reload
echo "Enabling jupyter service..."
sudo systemctl enable jupyter.service

# jupyter notebook --generate-config
echo "Launching jupyter notebook password manager..."
jupyter notebook password

sudo cp asound.conf /etc/asound.conf
echo "All done! Please reboot the device."