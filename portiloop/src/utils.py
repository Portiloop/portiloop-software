from EDFlib.edfwriter import EDFwriter
from portilooplot.jupyter_plot import ProgressPlot
from pathlib import Path
import numpy as np
import csv
import time


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


class EDFRecorder:
    def __init__(self, signal_labels):
        self.filename = EDF_PATH / 'recording.edf'
        self.nb_signals = 8
        self.samples_per_datarecord_array = self.frequency
        self.physical_max = 5
        self.physical_min = -5
        self.signal_labels = signal_labels

    def open_recording_file(self):
        nb_signals = self.nb_signals
        samples_per_datarecord_array = self.samples_per_datarecord_array
        physical_max = self.physical_max
        physical_min = self.physical_min
        signal_labels = self.signal_labels

        print(f"Will store edf recording in {self.filename}")

        self.edf_writer = EDFwriter(p_path=str(self.filename),
                                    f_file_type=EDFwriter.EDFLIB_FILETYPE_EDFPLUS,
                                    number_of_signals=nb_signals)
        
        for signal in range(nb_signals):
            assert self.edf_writer.setSampleFrequency(signal, samples_per_datarecord_array) == 0
            assert self.edf_writer.setPhysicalMaximum(signal, physical_max) == 0
            assert self.edf_writer.setPhysicalMinimum(signal, physical_min) == 0
            assert self.edf_writer.setDigitalMaximum(signal, 32767) == 0
            assert self.edf_writer.setDigitalMinimum(signal, -32768) == 0
            assert self.edf_writer.setSignalLabel(signal, signal_labels[signal]) == 0
            assert self.edf_writer.setPhysicalDimension(signal, 'V') == 0

    def close_recording_file(self):
        assert self.edf_writer.close() == 0
    
    def add_recording_data(self, data):
        self.edf_buffer += data
        if len(self.edf_buffer) >= self.samples_per_datarecord_array:
            datarecord_array = self.edf_buffer[:self.samples_per_datarecord_array]
            self.edf_buffer = self.edf_buffer[self.samples_per_datarecord_array:]
            datarecord_array = np.array(datarecord_array).transpose()
            assert len(datarecord_array) == self.nb_signals, f"len(data)={len(data)}!={self.nb_signals}"
            for d in datarecord_array:
                assert len(d) == self.samples_per_datarecord_array, f"{len(d)}!={self.samples_per_datarecord_array}"
                assert self.edf_writer.writeSamples(d) == 0


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
        point = next(self.csv_reader)
        self.index += 1
        while time.time() - self.last_time < self.wait_time:
            continue
        self.last_time = time.time()
        return self.index, float(point[0]), float(point[1]), point[2] == '1', point[3] == '1'