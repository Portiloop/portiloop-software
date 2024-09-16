from abc import ABC, abstractmethod
from enum import Enum
import time
from scipy.signal import find_peaks

import numpy as np
import matplotlib.pyplot as plt


class Delayer(ABC):
    """
    Interface that defines Delayers for stimulation
    """

    @abstractmethod
    def step(self, point):
        pass

    @abstractmethod
    def step_timestep(self, point):
        pass

    @abstractmethod
    def detected(self):
        pass


class TimingStates(Enum):
    READY = 0
    DELAYING = 1
    WAITING = 2


class TimingDelayer(Delayer):
    def __init__(self, stimulation_delay=0.0, inter_stim_delay=0.0, sample_freq=250):
        """
        Delays based on the timing
        params:
            stimulation_delay (float): How much time to wait after a detection before stimulation
            inter_stim_delay (float): How much time to wait after a stimulation before going back to a detection state
        """
        self.state = TimingStates.READY
        self.stimulation_delay = stimulation_delay
        self.inter_stim_delay = inter_stim_delay
        self.time_counter = 0
        self.sample_freq = sample_freq

    def step(self, point):
        """
        Moves through the state machine
        """
        if self.state == TimingStates.READY:
            return False
        elif self.state == TimingStates.DELAYING:
            if time.time() - self.delaying_start > self.stimulation_delay:
                # Actually stimulate the patient after the delay
                if self.stimulate is not None:
                    self.stimulate()
                self.state = TimingStates.WAITING
                self.waiting_start = time.time()
                return True
            return False
        elif self.state == TimingStates.WAITING:
            if time.time() - self.waiting_start > self.inter_stim_delay:
                self.state = TimingStates.READY
            return False

    def step_timestep(self, point):
        """
        Moves through the state machine
        """
        if self.state == TimingStates.READY:
            return False
        elif self.state == TimingStates.DELAYING:
            self.delaying_counter += 1
            if self.delaying_counter > self.stimulation_delay * self.sample_freq:
                # Actually stimulate the patient after the delay
                if self.stimulate is not None:
                    self.stimulate()
                self.state = TimingStates.WAITING
                self.waiting_counter = 0
                return True
            return False
        elif self.state == TimingStates.WAITING:
            self.waiting_counter += 1
            if self.waiting_counter > self.inter_stim_delay * self.sample_freq:
                self.state = TimingStates.READY
            return False

    def detected(self):
        """
        Defines what happens when a detection comes depending on what state you are in
        """
        if self.state == TimingStates.READY:
            self.state = TimingStates.DELAYING
            self.delaying_start = time.time()
            self.delaying_counter = 0


class UpStateStates(Enum):
    NO_SPINDLE = 0
    BUFFERING = 1
    DELAYING = 2


# Class that delays stimulation to always stimulate peak or through
class UpStateDelayer(Delayer):

    def __init__(self, sample_freq, peak, time_to_buffer, stimulate=None):
        """
        args:
            sample_freq: int -> Sampling frequency of signal in Hz
            time_to_wait: float -> Time to wait to build buffer in seconds
        """
        # Get number of timesteps for a whole spindle
        self.sample_freq = sample_freq
        self.peak = peak
        self.buffer = []
        self.time_to_buffer = time_to_buffer
        self.stimulate = stimulate

        self.state = UpStateStates.NO_SPINDLE

    def step(self, point):
        """
        Step the delayer, ads a point to buffer if necessary.
        Returns True if stimulation is actually done
        """
        if self.state == UpStateStates.NO_SPINDLE:
            return False
        elif self.state == UpStateStates.BUFFERING:
            self.buffer.append(point)
            # If we are done buffering, move on to the waiting stage
            if time.time() - self.time_started >= self.time_to_buffer:
                # Compute the necessary time to wait
                self.time_to_wait = self.compute_time_to_wait()
                self.state = UpStateStates.DELAYING
                self.buffer = []
                self.time_started = time.time()
            return False
        elif self.state == UpStateStates.DELAYING:
            # Check if we are done delaying
            if time.time() - self.time_started >= self.time_to_wait:
                # Actually stimulate the patient after the delay
                if self.stimulate is not None:
                    self.stimulate()
                # Reset state
                self.time_to_wait = -1
                self.state = UpStateStates.NO_SPINDLE
                return True
            return False

    def step_timesteps(self, point):
        """
        Step the delayer, ads a point to buffer if necessary.
        Returns True if stimulation is actually done
        """
        if self.state == UpStateStates.NO_SPINDLE:
            return False
        elif self.state == UpStateStates.BUFFERING:
            self.buffer.append(point)
            # If we are done buffering, move on to the waiting stage
            if len(self.buffer) >= self.time_to_buffer * self.sample_freq:
                # Compute the necessary time to wait
                self.time_to_wait = self.compute_time_to_wait()
                self.state = UpStateStates.DELAYING
                self.buffer = []
                self.delaying_counter = 0
            return False
        elif self.state == UpStateStates.DELAYING:
            # Check if we are done delaying
            self.delaying_counter += 1
            if self.delaying_counter >= self.time_to_wait * self.sample_freq:
                # Actually stimulate the patient after the delay
                if self.stimulate is not None:
                    self.stimulate()
                # Reset state
                self.time_to_wait = -1
                self.state = UpStateStates.NO_SPINDLE
                return True
            return False

    def detected(self):
        if self.state == UpStateStates.NO_SPINDLE:
            self.state = UpStateStates.BUFFERING

    def compute_time_to_wait(self):
        """
        Computes the time we want to wait in total based on the spindle frequency and the buffer
        """
        # If we want to look at the valleys, we search for peaks on the inversed signal
        if not self.peak:
            self.buffer = -self.buffer

        # Returns the index of the last peak in the buffer
        peaks, _ = find_peaks(self.buffer, prominence=1)

        # Make a figure to show the peaks
        if False:
            plt.figure()
            plt.plot(self.buffer)
            for peak in peaks:
                plt.axvline(x=peak)
            plt.plot(np.zeros_like(self.buffer), "--", color="gray")
            plt.show()

        if len(peaks) == 0:
            print("No peaks found, increase buffer size")
            return (self.sample_freq / 10) * (1.0 / self.sample_freq)

        # Compute average distance between each peak
        avg_dist = np.mean(np.diff(peaks))

        # Compute the time until next peak and return it
        if avg_dist < len(self.buffer) - peaks[-1]:
            print(
                "Average distance between peaks is smaller than the time to last peak, decrease buffer size"
            )
            return (len(self.buffer) - peaks[-1]) * (1.0 / self.sample_freq)
        return (avg_dist - (len(self.buffer) - peaks[-1])) * (1.0 / self.sample_freq)
