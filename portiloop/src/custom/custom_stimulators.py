import copy
import time
from abc import ABC, abstractmethod
from enum import Enum
from threading import Thread, Lock

import wave
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from portiloop.src.core.stimulation import Stimulator
from portiloop.src.core.utils import Dummy
from portiloop.src.core.constants import SOUNDS_FOLDER

from portiloop.src import ADS
if ADS:
    import alsaaudio


# ================== DELAYERS ==================


class TimingStates(Enum):
    READY = 0
    DELAYING = 1
    WAITING = 2


class Delayer(ABC):
    """
    Interface that defines Delayers for stimulation
    """
    @abstractmethod
    def step(self, point):
        """
        Moves through the state machine (by one single step)

        Args:
            point: single-step signal point

        Returns:
            stimulate: Boolean: whether to fire stimulation
        """
        pass

    # @abstractmethod
    # def step_timestep(self, point):
    #     pass

    @abstractmethod
    def detected(self):
        """
        Called before step() when the detection signal is True
        """
        pass


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

        self.stimulate = None
        self.waiting_start = time.time()
        self.delaying_start = time.time()
        self.delaying_counter = 0

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

    # def step_timestep(self, point):
    #     """
    #     Moves through the state machine
    #     """
    #     if self.state == TimingStates.READY:
    #         return False
    #     elif self.state == TimingStates.DELAYING:
    #         self.delaying_counter += 1
    #         if self.delaying_counter > self.stimulation_delay * self.sample_freq:
    #             # Actually stimulate the patient after the delay
    #             if self.stimulate is not None:
    #                 self.stimulate()
    #             self.state = TimingStates.WAITING
    #             self.waiting_counter = 0
    #             return True
    #         return False
    #     elif self.state == TimingStates.WAITING:
    #         self.waiting_counter += 1
    #         if self.waiting_counter > self.inter_stim_delay * self.sample_freq:
    #             self.state = TimingStates.READY
    #         return False

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
        '''
        args:
            sample_freq: int -> Sampling frequency of signal in Hz
            time_to_wait: float -> Time to wait to build buffer in seconds
        '''
        # Get number of timesteps for a whole spindle
        self.sample_freq = sample_freq
        self.peak = peak
        self.buffer = []
        self.time_to_buffer = time_to_buffer
        self.stimulate = stimulate

        self.state = UpStateStates.NO_SPINDLE
        self.time_started = time.time()

    def step(self, point):
        '''
        Step the delayer, ads a point to buffer if necessary.
        Returns True if stimulation is actually done
        '''
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
        '''
        Step the delayer, ads a point to buffer if necessary.
        Returns True if stimulation is actually done
        '''
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
        if (avg_dist < len(self.buffer) - peaks[-1]):
            print("Average distance between peaks is smaller than the time to last peak, decrease buffer size")
            return (len(self.buffer) - peaks[-1]) * (1.0 / self.sample_freq)
        return (avg_dist - (len(self.buffer) - peaks[-1])) * (1.0 / self.sample_freq)


class DelayedStimulator(Stimulator, ABC):
    def __init__(self, config_dict, lsl_streamer, csv_recorder):
        super().__init__(config_dict, lsl_streamer, csv_recorder)

        # Initialize stimulation delayer if requested
        delay = not ((config_dict['stim_delay'] == 0.0) and (config_dict['inter_stim_delay'] == 0.0))
        delay_phase = (not delay) and (not config_dict['spindle_detection_mode'] == 'Fast')
        if delay:
            stimulation_delayer = TimingDelayer(
                stimulation_delay=config_dict['stim_delay'],
                inter_stim_delay=config_dict['inter_stim_delay']
            )
        elif delay_phase:
            stimulation_delayer = UpStateDelayer(
                config_dict['frequency'],
                config_dict['spindle_detection_mode'] == 'Peak', 0.3)
        else:
            stimulation_delayer = None

        self.delayer = stimulation_delayer
        if self.delayer is not None:
            # When there is a delayer, delayer.step is in charge of calling delayer.stimulate
            self.delayer.stimulate = lambda: self.send_stimulation("DELAY_STIM", True)

    def stimulate(self, detection_signal):
        """
        In this group of custom Detectors/Stimulators, the Detector output signal is made of two lists:
        - The detection signal per-se
        - The input of the Detector so that the Stimulator can detect additional stuff, such as signal phase

        Args:
            detection_signal: (List, List)
        Returns:
            None
        """
        detection_points, filtered_points = detection_signal
        size = len(detection_points)
        assert len(filtered_points) == size

        for i in range(size):
            filtered_point = filtered_points[i]
            detection_point = detection_points[i]
            res_stim = self._stimulate(detection_point)
            if self.delayer is not None:
                res_del = self.delayer.step(filtered_point[self.config_dict['channel_detection'] - 1])
                self.csv_recorder.append_stimulation_signal_buffer([int(res_del)])
            else:
                self.csv_recorder.append_stimulation_signal_buffer([int(res_stim)])

    @abstractmethod
    def _stimulate(self, detection_point):
        """
        If there is a delayer, this method only calls delayer.detected in case of detection.
        Otherwise, it handles send_stimulation normally.

        Args:
            detection_point: single detection point

        Returns:
            Boolean: whether stimulation would occur without a delayer
        """
        raise NotImplementedError

    @abstractmethod
    def send_stimulation(self, lsl_text, sound):
        raise NotImplementedError


# ================== STIMULATORS ==================


class SleepSpindleRealTimeStimulator(DelayedStimulator):
    def __init__(self, config_dict, lsl_streamer, csv_recorder):
        super().__init__(config_dict, lsl_streamer, csv_recorder)
        # soundname = None
        # lsl_streamer = Dummy()

        soundname = config_dict['detection_sound']

        if soundname is None:
            self.soundname = 'stimulus.wav'  # CHANGE HERE TO THE SOUND THAT YOU WANT. ONLY ADD THE FILE NAME, NOT THE ENTIRE PATH
        else:
            self.soundname = soundname
        self._sound = SOUNDS_FOLDER / self.soundname
        self._thread = None
        self._lock = Lock()
        self.last_detected_ts = time.time()
        self.wait_t = 0.4  # 400 ms
        # self.delayer = None
        # self.lsl_streamer = lsl_streamer

        # Initialize Alsa stuff
        # Open WAV file and set PCM device
        with wave.open(str(self._sound), 'rb') as f:
            device = 'softvol'

            self.duration = f.getnframes() / float(f.getframerate())

            # 8bit is unsigned in wav files
            if f.getsampwidth() == 1:
                frmt = alsaaudio.PCM_FORMAT_U8
            # Otherwise we assume signed data, little endian
            elif f.getsampwidth() == 2:
                frmt = alsaaudio.PCM_FORMAT_S16_LE
            elif f.getsampwidth() == 3:
                frmt = alsaaudio.PCM_FORMAT_S24_3LE
            elif f.getsampwidth() == 4:
                frmt = alsaaudio.PCM_FORMAT_S32_LE
            else:
                raise ValueError('Unsupported format')

            self.periodsize = f.getframerate() // 8

            try:
                self.pcm = alsaaudio.PCM(channels=f.getnchannels(), rate=f.getframerate(), format=frmt, periodsize=self.periodsize, device=device)
            except alsaaudio.ALSAAudioError as e:
                self.pcm = Dummy()
                raise e

            # Store data in list to avoid reopening the file
            self.wav_list = []
            while True:
                data = f.readframes(self.periodsize)
                if data:
                    self.wav_list.append(data)
                else:
                    break

        # print(f"DEBUG: Stimulator will play sound {self.soundname}, duration: {self.duration:.3f} seconds")

    def play_sound(self):
        '''
        Open the wav file and play a sound
        '''
        self.end = time.time()
        for data in self.wav_list:
            self.pcm.write(data)

        # Added this to make sure the thread does not stop before the sound is done playing
        time.sleep(self.duration)

    def _stimulate(self, detection_point):
        res = False
        if detection_point:
            ts = time.time()
            # Check if time since last stimulation is long enough
            if ts - self.last_detected_ts > self.wait_t:
                res = True
                if self.delayer is not None:
                    # Notify the delayer that a detection has happened
                    self.delayer.detected()
                    # Send an LSL marker showing where stimulation would have been without a delayer
                    # (NB: The actual stimulation will be sent later by self.delayer.step)
                    self.send_stimulation("FAST_STIM", False)
                else:
                    # No delayer: send actual stimulation immediately
                    self.send_stimulation("STIM", True)
            self.last_detected_ts = ts
        return res

    def send_stimulation(self, lsl_text, sound):
        # Send lsl marker
        self.lsl_streamer.push_marker(lsl_text)
        # Send sound to patient
        if sound:
            with self._lock:
                if self._thread is None:
                    self._thread = Thread(target=self._t_sound, daemon=True)
                    self._thread.start()

    def _t_sound(self):
        self.play_sound()
        with self._lock:
            self._thread = None

    def test_stimulus(self):
        start = time.time()
        with self._lock:
            if self._thread is None:
                self._thread = Thread(target=self._t_sound, daemon=True)
                self._thread.start()

    def __del__(self):
        # print("DEBUG: releasing PCM")
        del self.pcm


class SpindleTrainRealTimeStimulator(SleepSpindleRealTimeStimulator):
    def __init__(self, config_dict, lsl_streamer, csv_recorder):
        super().__init__(config_dict, lsl_streamer, csv_recorder)
        self.max_spindle_train_t = 6.0

    def _stimulate(self, detection_point):
        res = False
        if detection_point:
            ts = time.time()
            elapsed = ts - self.last_detected_ts
            if self.wait_t < elapsed < self.max_spindle_train_t:
                res = True
                if self.delayer is not None:
                    self.delayer.detected()
                    self.send_stimulation("FAST_STIM", False)
                else:
                    self.send_stimulation("STIM", True)
            self.last_detected_ts = ts
        return res


class IsolatedSpindleRealTimeStimulator(SpindleTrainRealTimeStimulator):
    def _stimulate(self, detection_point):
        res = False
        if detection_point:
            ts = time.time()
            elapsed = ts - self.last_detected_ts
            if self.max_spindle_train_t < elapsed:
                res = True
                if self.delayer is not None:
                    self.delayer.detected()
                    self.send_stimulation("FAST_STIM", False)
                else:
                    self.send_stimulation("STIM", True)
            self.last_detected_ts = ts
        return res


class AlternatingStimulator(Stimulator):
    def __init__(self, config_dict, lsl_streamer, csv_recorder):
        super().__init__(config_dict, lsl_streamer, csv_recorder)

        # soundname = None
        # lsl_streamer = Dummy()

        stim_interval = 0.250

        # soundname = config_dict['detection_sound']

        self.pos_soundname = 'syllPos120.wav'  # CHANGE HERE TO THE SOUND THAT YOU WANT. ONLY ADD THE FILE NAME, NOT THE ENTIRE PATH
        self.neg_soundname = 'syllNeg120.wav'

        self.pos_sound = SOUNDS_FOLDER / self.pos_soundname
        self.neg_sound = SOUNDS_FOLDER / self.neg_soundname

        self._thread = None
        self._lock = Lock()
        self.lsl_streamer = lsl_streamer

        # Stimulation parameters
        self.stim_interval = stim_interval
        self.stim_polarity = True
        self.last_stim = 0.0

        # Initialize Alsa stuff
        # Open WAV file and set PCM device
        with wave.open(str(self.pos_sound), 'rb') as f:
            device = 'softvol'

            self.duration = f.getnframes() / float(f.getframerate())

            # 8bit is unsigned in wav files
            if f.getsampwidth() == 1:
                frmt = alsaaudio.PCM_FORMAT_U8
            # Otherwise we assume signed data, little endian
            elif f.getsampwidth() == 2:
                frmt = alsaaudio.PCM_FORMAT_S16_LE
            elif f.getsampwidth() == 3:
                frmt = alsaaudio.PCM_FORMAT_S24_3LE
            elif f.getsampwidth() == 4:
                frmt = alsaaudio.PCM_FORMAT_S32_LE
            else:
                raise ValueError('Unsupported format')

            self.periodsize = f.getframerate() // 8

            try:
                self.pcm = alsaaudio.PCM(channels=f.getnchannels(), rate=f.getframerate(), format=frmt, periodsize=self.periodsize, device=device)
            except alsaaudio.ALSAAudioError as e:
                self.pcm = Dummy()
                raise e

            # Store data in list to avoid reopening the file
            self.pos_wav_list = []
            while True:
                data = f.readframes(self.periodsize)
                if data:
                    self.pos_wav_list.append(data)
                else:
                    break

        with wave.open(str(self.neg_sound), 'rb') as f:
            device = 'softvol'

            self.duration = f.getnframes() / float(f.getframerate())

            # 8bit is unsigned in wav files
            if f.getsampwidth() == 1:
                frmt = alsaaudio.PCM_FORMAT_U8
            # Otherwise we assume signed data, little endian
            elif f.getsampwidth() == 2:
                frmt = alsaaudio.PCM_FORMAT_S16_LE
            elif f.getsampwidth() == 3:
                frmt = alsaaudio.PCM_FORMAT_S24_3LE
            elif f.getsampwidth() == 4:
                frmt = alsaaudio.PCM_FORMAT_S32_LE
            else:
                raise ValueError('Unsupported format')

            self.periodsize = f.getframerate() // 8

            try:
                self.pcm = alsaaudio.PCM(channels=f.getnchannels(), rate=f.getframerate(), format=frmt, periodsize=self.periodsize, device=device)
            except alsaaudio.ALSAAudioError as e:
                self.pcm = Dummy()
                raise e

            # Store data in list to avoid reopening the file
            self.neg_wav_list = []
            while True:
                data = f.readframes(self.periodsize)
                if data:
                    self.neg_wav_list.append(data)
                else:
                    break
        # print(f"DEBUG: Stimulator will play sound {self.soundname}, duration: {self.duration:.3f} seconds")

    def play_sound(self, polarity):
        '''
        Open the wav file and play a sound
        '''
        print(polarity)
        played_sound = self.pos_wav_list if polarity else self.neg_wav_list
        for data in played_sound:
            self.pcm.write(data)

        # Added this to make sure the thread does not stop before the sound is done playing
        time.sleep(self.duration)

    def stimulate(self, detection_signal):
        stim = []
        for _ in detection_signal:
            # We ignore the input signal and simply make sure we stimulate at the given interval
            current_time = time.time()
            if current_time - self.last_stim >= self.stim_interval:
                stim.append(1)
                # Check if we are in the inverted phase:
                if self.stim_polarity:
                    stim_text = 'STIM_POS'
                else:
                    stim_text = 'STIM_NEG'
                self.send_stimulation(stim_text, True)
                self.last_stim = current_time
                self.stim_polarity = not self.stim_polarity
            else:
                stim.append(0)
        assert len(detection_signal) == len(stim)
        self.csv_recorder.append_stimulation_signal_buffer(stim)

    def send_stimulation(self, lsl_text, sound):
        # Send lsl stimulation
        # print(f"Stimulating at time: {time.time()} with text: {lsl_text}")
        self.lsl_streamer.push_marker(lsl_text)
        polarity_copy = copy.deepcopy(self.stim_polarity)
        # Send sound to patient
        if sound:
            with self._lock:
                if self._thread is None:
                    self._thread = Thread(target=self._t_sound, args=(polarity_copy, ), daemon=True)
                    self._thread.start()

    def _t_sound(self, polarity):
        self.play_sound(polarity)
        with self._lock:
            self._thread = None

    def test_stimulus(self):
        start = time.time()
        with self._lock:
            if self._thread is None:
                self._thread = Thread(target=self._t_sound, daemon=True)
                self._thread.start()

    def __del__(self):
        del self.pcm

    # def add_delayer(self, delayer):
    #     self.delayer = delayer
    #     self.delayer.stimulate = lambda: self.send_stimulation("DELAY_STIM", False)

