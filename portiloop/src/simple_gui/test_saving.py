import csv
import numpy as np
import pandas as pd
from portiloop.src.utils import EDFRecorder
from tqdm import tqdm

# Define the parameters
num_channels = 9
sampling_rate = 250  # Hz
duration = 1 * 60 * 60  # hours in seconds

# Calculate the total number of data points
total_samples = duration * sampling_rate

# Generate random data for each channel
data = np.random.rand(total_samples, num_channels)

# np.save('/home/mendel/portiloop-software/random_data', data)

test = EDFRecorder('/home/mendel/portiloop-software/random_data.npy')
for point in tqdm(data):
    test.add_recording_data([point])
