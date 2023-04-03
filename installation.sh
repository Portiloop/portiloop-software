#!/bin/bash

echo "Installing dependencies with apt-get..."
sudo apt-get update
sudo apt-get install -y python3-matplotlib python3-scipy python3-dev libasound2-dev jupyter-notebook jupyter 
echo "done"

echo "Installing latest pycoral and tflite-runtime..."
wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl
wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp37-cp37m-linux_aarch64.whl
pip3 uninstall tflite-runtime
pip3 uninstall pycoral
pip3 install tflite_runtime-2.5.0.post1-cp37-cp37m-linux_aarch64.whl
pip3 install pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl
echo "done"

echo "Installing the dependencies... [This may take a while]"
pip3 install -e .
echo "done"

echo "Activating the widgets for the jupyter notebook..."
jupyter nbextension enable --py widgetsnbextension
echo "done"

echo "Creating workspace directory..."
mkdir workspace
mkdir workspace/edf_recording
echo "done"

