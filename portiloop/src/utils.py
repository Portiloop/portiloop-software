# from EDFlib.edfwriter import EDFwriter
from abc import ABC, abstractmethod
import io
from pyedflib import highlevel
from portilooplot.jupyter_plot import ProgressPlot
from pathlib import Path
import numpy as np
import csv
import time
import os
import warnings
import multiprocessing as mp

from portiloop.src.processing import int_to_float


EDF_PATH = Path.home() / 'workspace' / 'edf_recording'
# Path to the recordings
RECORDING_PATH = Path.home() / 'portiloop-software' / 'portiloop' / 'recordings'


def get_portiloop_version():
    # Check if we are on a Portiloop V1 or V2.
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            string = m.read().lower()
            if "phanbell" in string:
                version = 1
            elif "coral" in string:
                version = 2
            else: 
                version = -1
    except Exception:
        version = -1
    return version


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
            # print(disp_list)
            # print(datapoints)
            self.pp.update_with_datapoints(disp_list)
        elif len(self.history) == 1000:
            plt.plot(self.history)
            plt.show()
            self.history = []
    
    def add_datapoint(self, datapoint):
        disp_list = [[elt] for elt in datapoint]
        self.pp.update(disp_list)


class Dummy:
    def __getattr__(self, attr):
        return lambda *args, **kwargs: None

class CaptureFrontend(ABC):
    """
    Interface that defines how we talk to a capture frontend:
    """
    @abstractmethod
    def init_capture(self):
        pass

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def get_msg(self):
        pass

    @abstractmethod
    def send_msg(self, msg):
        pass

    @abstractmethod
    def close(self):
        pass


class ADSFrontend(CaptureFrontend):
    def __init__(self, duration, frequency, python_clock, channel_states, process):
        """
        duration (float): duration of the capture in seconds
        frequency (float): sampling frequency in Hz
        python_clock (bool): if True, use the python clock to time the capture
        channel_states (list): list of channel states
        """
        # Initialize the variables
        self.duration = duration
        self.frequency = frequency
        self.python_clock = python_clock
        self.channel_states = channel_states

        # Initialize The data pipes to talk to the data process
        self.capture_started = False
        self.p_msg_io, self.p_msg_io_2 = mp.Pipe()
        self.p_data_i, self.p_data_o = mp.Pipe(duplex=False)
        self.process = process

    def init_capture(self):
        """
        Actually initialize the capture process
        """
        self._p_capture = mp.Process(target=self.process,
                                        args=(self.p_data_o,
                                            self.p_msg_io_2,
                                            self.duration,
                                            self.frequency,
                                            self.python_clock,
                                            1.0,
                                            self.channel_states)
                                        )
        self._p_capture.start()
        self.capture_started = True
        # If any issue arises, we want to kill this process
        print(f"PID capture: {self._p_capture.pid}. Kill this process if program crashes before end of execution.")

    def send_msg(self, msg):
        """
        Send message STOP to stop capture process
        """
        if self.capture_started:
            self.p_msg_io.send(msg)

    def get_msg(self):
        """
        Returns messages from capture process
        """
        if self.capture_started:
            if self.p_msg_io.poll():
                return self.p_msg_io.recv()
            
    def get_data(self):
        """
        Returns data from capture process if any available. Otherwise returns None.
        The value returned is a numpy array of shape (n, 8) containing the data points in MicroVolts.
        n depends on how many datapoints have been added to the pipe since the last check.
        """
        point = None
        if self.p_data_i.poll(timeout=(1 / self.frequency)):
            point = self.p_data_i.recv()

        # Convert point from int to corresponding value in microvolts
        return int_to_float(np.array([point])) if point is not None else None

    def close(self):
        # Empty pipes 
        while True:
            if self.p_data_i.poll():
                _ = self.p_data_i.recv()
            elif self.p_msg_io.poll():
                _ = self.p_msg_io.recv()
            else:
                break

        self.p_data_i.close()
        self.p_msg_io.close()
        self._p_capture.join()


class FileFrontend(CaptureFrontend):
    def __init__(self, filename, num_channels, channel_detect):
        """
        Frontend that reads from a csv file. Mostly used for debugging.
        """
        self.filename = filename
        print(f"Reading from file {filename}")
        self.stop_msg = False
        self.num_channels = num_channels
        self.channel_detect = channel_detect
    
    def init_capture(self):
        """
        Initialize the file reader
        """
        self.file = open(self.filename, 'r')
        self.csv_reader = csv.reader(self.file, delimiter=',')
        self.wait_time = 1/250.0
        self.index = -1
        self.last_time = time.time()

    def send_msg(self, msg):
        """
        Does nothing
        """
        if msg == "STOP":
            self.stop_msg = True

    def get_msg(self):
        """
        If we have reached the end of the file, this tells the main loop to stop
        """
        if self.stop_msg:
            return "STOP"

    def get_data(self):
        """
        Returns the next point in the file
        """
        try:
            point = next(self.csv_reader)
            self.index += 1
            while time.time() - self.last_time < self.wait_time:
                continue
            self.last_time = time.time()
            n_array_raw = np.zeros(self.num_channels)
            n_array_raw[self.channel_detect-1] = float(point[0])
            n_array_raw = np.reshape(n_array_raw, (1, self.num_channels))
            return n_array_raw
        except StopIteration:
            print("Reached end of file, stopping...")
            self.stop_msg = True

    def close(self):
        self.file.close()


class LSLStreamer:
    def __init__(self, streams, channel_count, frequency, id):
        from pylsl import StreamInfo, StreamOutlet

        self.streams = streams

        lsl_info_raw = StreamInfo(name='Portiloop Raw Data',
                        type='Raw EEG signal',
                        channel_count=channel_count,
                        nominal_srate=frequency,
                        channel_format='float32',
                        source_id=id)  
        self.lsl_outlet_raw = StreamOutlet(lsl_info_raw)

        if streams['filtered']:
            lsl_info = StreamInfo(name='Portiloop Filtered',
                                    type='Filtered EEG',
                                    channel_count=channel_count,
                                    nominal_srate=frequency,
                                    channel_format='float32',
                                    source_id=id)  
            self.lsl_outlet = StreamOutlet(lsl_info)

        if streams['markers']:
            lsl_markers_info = StreamInfo(name='Portiloop_stimuli',
                                    type='Markers',
                                    channel_count=1,
                                    channel_format='string',
                                    source_id=id) 
            self.lsl_outlet_markers = StreamOutlet(lsl_markers_info)

    def push_filtered(self, data):
        self.lsl_outlet.push_sample(data)

    def push_raw(self, data):
        self.lsl_outlet_raw.push_sample(data)

    def push_marker(self, text):
        self.lsl_outlet_markers.push_sample([text])

    def __del__(self):
        print("Closing LSL streams")
        self.lsl_outlet_raw.__del__()
        if self.streams['filtered']:
            self.lsl_outlet.__del__()
        if self.streams['markers']:
            self.lsl_outlet_markers.__del__()

    @staticmethod
    def string_for_detection_activation(pause):
        return "DETECT_OFF" if pause else "DETECT_ON"
