from abc import ABC, abstractmethod
import io
from portilooplot.jupyter_plot import ProgressPlot
from pathlib import Path
import numpy as np
import csv
import time
import os
import warnings
import multiprocessing as mp
import time
from portiloop.src.processing import bin_to_microvolt


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

class CSVRecorder:
    def __init__(self, filename):
        self.writing_buffer = []
        self.max_write = 1
        self.filename = filename
        self.file = open(self.filename, 'a')
        self.writer = csv.writer(self.file)
        print(f"Saving file to {self.filename}")
        self.out_format = 'csv' # 'npy'

    def __del__(self):
        print(f"Closing")
        # self.file.close()

    def add_recording_data(self, points, detection_info, detection_on, stim_on):
        stim_label = 2 if stim_on else 1
        
        #detection_info = (np.array(detection_info).astype(int) * stim_label).tolist()
        # No need to bother np arrays
        detection_info = [stim_label*x for x in detection_info]

        # If detection is on but we do not have any points, we add 0s
        if detection_on and len(detection_info) == 0:
            for point in points:
                point.append(0)
        # If detection is not on we simply pass
        elif not detection_on:
            pass
        # If detection_info has points
        elif len(detection_info) > 0:
            # This takes care of the case when detection is turned on by unpausing between two saves
            diff_points = len(points) - len(detection_info)

            if diff_points != 0:
                detection_info = [0.0] * diff_points + detection_info

            assert len(points) == len(detection_info)
            for idx, point in enumerate(points):
                point.append(detection_info[idx])
            
        data = points
        self.writing_buffer += data
        # write to file

        if len(self.writing_buffer) >= self.max_write:
            if self.out_format == 'csv':
                self.writer.writerows(self.writing_buffer)
                #np.savetxt(self.file, np.array(self.writing_buffer), delimiter=',')
            elif self.out_format == 'npy':
                np.save(self.file, np.array(self.writing_buffer))
            self.writing_buffer = []



class LiveDisplay():
    def __init__(self, channel_names, window_len=100):
        self.history = []
        self.pp = ProgressPlot(plot_names=channel_names,
                               max_window_len=window_len)
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
    Interface that defines how we talk to a capture frontend
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
    def __init__(self, duration, frequency, python_clock, channel_states, vref, process):
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
        self.vref = vref

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
        print(
            f"PID capture: {self._p_capture.pid}. Kill this process if program crashes before end of execution.")

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
        return bin_to_microvolt(np.array([point]), self.vref) if point is not None else None

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
                                  type='eeg',
                                  channel_count=channel_count,
                                  nominal_srate=frequency,
                                  channel_format='float32',
                                  source_id=id)
        self.lsl_outlet_raw = StreamOutlet(lsl_info_raw)

        if streams['filtered']:
            lsl_info = StreamInfo(name='Portiloop Filtered',
                                  type='eeg',
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

def get_temperature_celsius(value_microvolt):
    return (value_microvolt - 145300) / 490 + 25
