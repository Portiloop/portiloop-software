import csv
import multiprocessing as mp
import time
from abc import ABC, abstractmethod

import numpy as np

from portiloop.src.core.hardware.config_hardware import ADS_LSB


class CaptureBackend(ABC):
    """
    Interface that defines how we talk to a capture backend
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


class ADSBackend(CaptureBackend):
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

        self._p_capture = None

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


class FileBackend(CaptureBackend):
    def __init__(self, filename, num_channels, channel_detect, frequency):
        """
        Backend that reads from a csv file. Mostly used for debugging.
        """
        self.filename = filename
        print(f"Reading from file {filename}")
        self.stop_msg = False
        self.num_channels = num_channels
        self.channel_detect = channel_detect
        self.frequency = frequency

        self.file = None
        self.csv_reader = None
        self.wait_time = None
        self.index = None
        self.last_time = None

    def init_capture(self):
        """
        Initialize the file reader
        """
        self.file = open(self.filename, 'r')
        self.csv_reader = csv.reader(self.file, delimiter=',')
        self.wait_time = 1.0 / self.frequency
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


def bin_to_microvolt(value, vref):
    """
    Convert the binary value out of the ADS into a float value in microvolts
    """
    return filter_scale(filter_2scomplement_np(value), vref)


def filter_2scomplement_np(value):
    """
    Converts the binary ADS value into an integer by applying 2's complement
    """
    return np.where((value & (1 << 23)) != 0, value - (1 << 24), value)


def filter_scale(value, vref):
    """
    Scales the integer value into microvolts
    """
    return value * 1e6 * vref * ADS_LSB
