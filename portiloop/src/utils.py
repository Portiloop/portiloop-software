from EDFlib.edfwriter import EDFwriter
from pyedflib import highlevel
from portilooplot.jupyter_plot import ProgressPlot
from pathlib import Path
import numpy as np
import csv
import time
import os
import warnings


EDF_PATH = Path.home() / 'workspace' / 'edf_recording'
# Path to the recordings
RECORDING_PATH = Path.home() / 'portiloop-software' / 'portiloop' / 'recordings'


class DummyAlsaMixer:
    def __init__(self):
        self.volume = 50
    
    def getvolume(self):
        return [self.volume]
    
    def setvolume(self, volume):
        self.volume = volume   


# class EDFRecorder:
#     def __init__(self, signal_labels, filename, frequency):
#         self.filename = filename
#         self.nb_signals = len(signal_labels)
#         self.samples_per_datarecord_array = frequency
#         self.physical_max = 5000000
#         self.physical_min = -5000000
#         self.signal_labels = signal_labels
#         self.edf_buffer = []

#     def open_recording_file(self):
#         nb_signals = self.nb_signals
#         samples_per_datarecord_array = self.samples_per_datarecord_array
#         physical_max = self.physical_max
#         physical_min = self.physical_min
#         signal_labels = self.signal_labels

#         print(f"Will store edf recording in {self.filename}")

#         self.edf_writer = EDFwriter(p_path=str(self.filename),
#                                     f_file_type=EDFwriter.EDFLIB_FILETYPE_EDFPLUS,
#                                     number_of_signals=nb_signals)
        
#         for signal in range(nb_signals):
#             assert self.edf_writer.setSampleFrequency(signal, samples_per_datarecord_array) == 0
#             assert self.edf_writer.setPhysicalMaximum(signal, physical_max) == 0
#             assert self.edf_writer.setPhysicalMinimum(signal, physical_min) == 0
#             assert self.edf_writer.setDigitalMaximum(signal, 32767) == 0
#             assert self.edf_writer.setDigitalMinimum(signal, -32768) == 0
#             assert self.edf_writer.setSignalLabel(signal, signal_labels[signal]) == 0
#             assert self.edf_writer.setPhysicalDimension(signal, 'uV') == 0

#     def close_recording_file(self):
#         assert self.edf_writer.close() == 0
    
#     def add_recording_data(self, data):
#         self.edf_buffer += data
#         if len(self.edf_buffer) >= self.samples_per_datarecord_array:
#             datarecord_array = self.edf_buffer[:self.samples_per_datarecord_array]
#             self.edf_buffer = self.edf_buffer[self.samples_per_datarecord_array:]
#             datarecord_array = np.array(datarecord_array).transpose()
#             assert len(datarecord_array) == self.nb_signals, f"len(data)={len(data)}!={self.nb_signals}"
#             for d in datarecord_array:
#                 assert len(d) == self.samples_per_datarecord_array, f"{len(d)}!={self.samples_per_datarecord_array}"
#                 assert self.edf_writer.writeSamples(d) == 0

class EDFRecorder:
    def __init__(self, signal_labels, filename, frequency):
        self.writing_buffer = []
        self.max_write = 1000
        self.filename = filename
        self.csv_filename = str(filename).split('.')[0] + '.csv'
        self.signal_labels = signal_labels
        self.frequency = frequency

    def open_recording_file(self):
        self.file = open(self.csv_filename, 'w')

    def close_recording_file(self):
        self.file.close()
        data = np.genfromtxt(self.csv_filename, delimiter=',')
        # Convert to float values
        data = data.astype(np.float32)
        data = data.transpose()
        assert data.shape[0] == len(self.signal_labels), f"{data.shape[0]}!={len(self.signal_labels)}"
        signal_headers = []
        for row_i in range(data.shape[0]):
            # If we only have zeros in that row, the channel was not activated so we must set the physical max and min manually
            if np.all(data[row_i] == 0):
                phys_max = 200
                phys_min = -200
            else:
                phys_max = np.amax(data[row_i])
                phys_min = np.amin(data[row_i])

            # Create the signal header
            signal_headers.append(highlevel.make_signal_header(
                self.signal_labels[row_i], 
                sample_frequency=self.frequency,
                physical_max=phys_max,
                physical_min=phys_min,))
        self.filename = str(self.filename)
        print(f"Saving to {self.filename}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            highlevel.write_edf(str(self.filename), data, signal_headers)

        os.remove(self.csv_filename)

    def add_recording_data(self, data):
        self.writing_buffer += data
        # write to file
        if len(self.writing_buffer) >= self.max_write:
            for point in self.writing_buffer:
                self.file.write(','.join([str(elt) for elt in point]) + '\n')
            self.writing_buffer = []

        

class LiveDisplay():
    def __init__(self, channel_names, window_len=100):
        self.datapoint_dim = len(channel_names)
        self.history = []
        self.pp = ProgressPlot(plot_names=channel_names, max_window_len=window_len)
        self.matplotlib = False

    def add_datapoints(self, datapoints):
        """
        Adds 8 lists of datapoints to the plot
        
        Args:
            datapoints: list of 8 lists of floats (or list of 8 floats)
        """
        if self.matplotlib:
            import matplotlib.pyplot as plt
        disp_list = []
        for datapoint in datapoints:
            d = [[elt] for elt in datapoint]
            disp_list.append(d)
            
            if self.matplotlib:
                self.history += d[1]
        
        if not self.matplotlib:
            self.pp.update_with_datapoints(disp_list)
        elif len(self.history) == 1000:
            plt.plot(self.history)
            plt.show()
            self.history = []
    
    def add_datapoint(self, datapoint):
        disp_list = [[elt] for elt in datapoint]
        self.pp.update(disp_list)


class FileReader:
    def __init__(self, filename):
        file = open(filename, 'r')
        # Open a csv file
        print(f"Reading from file {filename}")
        self.csv_reader = csv.reader(file, delimiter=',')
        self.wait_time = 1/250.0
        self.index = -1
        self.last_time = time.time()

    def get_point(self):
        """
        Returns the next point in the file
        """
        try:
            point = next(self.csv_reader)
            self.index += 1
            while time.time() - self.last_time < self.wait_time:
                continue
            self.last_time = time.time()
            return self.index, float(point[0]), float(point[1]), point[2] == '1', point[3] == '1'
        except StopIteration:
            return None