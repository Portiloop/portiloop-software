import time
from abc import ABC, abstractmethod
from enum import Enum
from threading import Thread, Lock

import wave
import numpy as np
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

    def detected(self):
        """
        May be called when a pattern of interest is detected by the detector.
        """
        pass

    def not_detected(self):
        """
        May be called when no pattern of interest is detected by the detector.
        """
        pass


# class RandomDelayer(Delayer):
#     def __init__(self, config_dict, stimulate_fn: callable = None):
#         """
#         Randomly delays stimulations between min_delay and max_delay whenever a detection happens.
#         While delaying a stimulation, no new detection is taken in account.

#         Args:
#             config_dict: configuration dictionary
#         """
#         self.min_delay = config_dict['min_delay']
#         self.max_delay = config_dict['max_delay']
#         self.sample_freq = config_dict['frequency']

#         self._t = 0
#         self._t_next_detection = None
#         self.stimulate = stimulate_fn

#     def step(self, point):
#         """
#         Moves through the state machine
#         """

#         self._t += 1

#         if self._t_next_detection is None:
#             return False
#         else:
#             if self._t >= self._t_next_detection:
#                 # Actually stimulate the patient after the delay
#                 if self.stimulate is not None:
#                     self.stimulate()
#                 self._t_next_detection = None
#                 return True
#             else:
#                 return False

#     def detected(self):
#         """
#         Defines what happens on detection
#         """
#         if self._t_next_detection is None:
#             delay = np.random.uniform(low=self.min_delay, high=self.max_delay)
#             delay_steps = int(self.sample_freq * delay)
#             self._t_next_detection = self._t + delay_steps


# class TimingDelayer(Delayer):
#     def __init__(self, config_dict, stimulate_fn=None):
#         """
#         Delays based on the timing.

#         Args:
#             config_dict: configuration dictionary
#         """
#         self.state = TimingStates.READY
#         self.stimulation_delay = config_dict['min_delay']
#         self.inter_stim_delay = config_dict['inter_stim_delay']
#         self.sample_freq = config_dict['frequency']

#         self.stimulate = stimulate_fn
#         self.waiting_start = time.time()
#         self.delaying_start = time.time()

#     def step(self, point):
#         """
#         Moves through the state machine
#         """
#         if self.state == TimingStates.READY:
#             return False
#         elif self.state == TimingStates.DELAYING:
#             if time.time() - self.delaying_start > self.stimulation_delay:
#                 # Actually stimulate the patient after the delay
#                 if self.stimulate is not None:
#                     self.stimulate()
#                 self.state = TimingStates.WAITING
#                 self.waiting_start = time.time()
#                 return True
#             return False
#         elif self.state == TimingStates.WAITING:
#             if time.time() - self.waiting_start > self.inter_stim_delay:
#                 self.state = TimingStates.READY
#             return False

#     def detected(self):
#         """
#         Defines what happens when a detection comes depending on what state you are in
#         """
#         if self.state == TimingStates.READY:
#             self.state = TimingStates.DELAYING
#             self.delaying_start = time.time()


class RandomTimingDelayer(Delayer):
    def __init__(self, config_dict, stimulate_fn=None):
        """
        Delays based on the timing.

        Args:
            config_dict: configuration dictionary
        """
        self.state = TimingStates.READY
        self.min_delay = config_dict['min_delay']
        self.max_delay = config_dict['max_delay']
        if self.max_delay < self.min_delay:  # constant delay
            self.max_delay = self.min_delay
        self.inter_stim_delay = config_dict['inter_stim_delay']
        self.sample_freq = config_dict['frequency']

        self.stimulate = stimulate_fn
        self.waiting_start = time.time()
        self.delaying_start = time.time()
        self._delay = 0

    def step(self, point):
        """
        Moves through the state machine
        """
        if self.state == TimingStates.READY:
            return False
        elif self.state == TimingStates.DELAYING:
            if time.time() - self.delaying_start >= self._delay:
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

    def detected(self):
        """
        Defines what happens when a detection comes depending on what state you are in
        """
        if self.state == TimingStates.READY:
            self._delay = np.random.uniform(low=self.min_delay, high=self.max_delay)
            self.delaying_start = time.time()
            self.state = TimingStates.DELAYING


class UpStateStates(Enum):
    NO_SPINDLE = 0
    BUFFERING = 1
    DELAYING = 2


# Class that delays stimulation to always stimulate peak or through

# FIXME: this class implementation is losing a lot of time buffering
class UpStateDelayer(Delayer):

    def __init__(self, config_dict, stimulate_fn=None, time_to_buffer=0.3):
        '''
        args:
            config_dict: configuration dictionary
            time_to_buffer: float -> Time to wait to build buffer in seconds
        '''
        self.sample_freq = config_dict['frequency']
        self.peak = config_dict['stim_delay_mode'] == 'Peak'
        self.buffer = []
        self.time_to_buffer = time_to_buffer
        self.channel_idx = config_dict['channel_detection'] - 1
        self.stimulate = stimulate_fn

        self.time_to_wait = -1

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
            self.buffer.append(point[self.channel_idx])
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

    def detected(self):
        if self.state == UpStateStates.NO_SPINDLE:
            self.state = UpStateStates.BUFFERING
            self.time_started = time.time()

    def compute_time_to_wait(self):
        """
        Computes the time we want to wait in total based on the spindle frequency and the buffer
        """
        # If we want to look at the valleys, we search for peaks on the inverted signal
        buffer = np.array(self.buffer)
        if not self.peak:
            buffer = -buffer

        # Returns the index of the last peak in the buffer
        peaks, _ = find_peaks(buffer, prominence=1)

        if len(peaks) < 2:
            print("No peaks found, increase buffer size")
            return (self.sample_freq / 10) * (1.0 / self.sample_freq)

        # Compute average distance between each peak
        avg_dist = np.mean(np.diff(peaks))

        # Compute the time until next peak and return it
        if (avg_dist < len(buffer) - peaks[-1]):
            print("Average distance between peaks is smaller than the time to last peak, decrease buffer size")
            return (len(buffer) - peaks[-1]) * (1.0 / self.sample_freq)
        return (avg_dist - (len(buffer) - peaks[-1])) * (1.0 / self.sample_freq)


# class SOPhaseDelayer(Delayer):  # FIXME: This class is not tested and has memory leaks
#     def __init__(self,
#                  config_dict,
#                  k_p: float = 0.05,
#                  k_i: float = 5e-8,
#                  k_0: float = 0.03):
#         """
#         Phase Locked Loop for In-Phase Slow Oscillation Detection
#         params:
#             config_dict: configuration dictionary
#             k_p, k_i, k_0: PLL tuning parameters
#         """
#         self.k_p = k_p
#         self.k_i = k_i
#         self.k_0 = k_0
#         self.fs = config_dict['frequency']

#         self.target_phase = 0

#         self.sin_out = 0
#         self.cos_out = 1
#         self.pd_output = 0      # phase detector output
#         self.lf_output = 0      # loop filter output
#         self.integrator = 0

#         self.freq_const = 2 * np.pi * (1/self.fs)
#         self.init_estimate = 0
#         self.phase_estimate = self.freq_const

#         self.atol = np.deg2rad(10)
#         self.channel_idx = config_dict['channel_detection'] - 1

#         self.prev_cos_out = 1
#         self.cos_outs = []
#         self.phase_estimates = []
#         self.phase_indicators = []
#         self.stimulate_flag = False

#         self.phase_indicator = None
#         self.stimulate = None

#     def wrap_phase(self, phase):
#         return np.angle(np.exp(1j * phase))

#     def pll_detect(self, point):
#         self.pd_output = point * self.sin_out

#         self.integrator += self.k_i * self.pd_output
#         self.lf_output = self.k_p * self.pd_output + self.integrator

#         next_phase = self.phase_estimate + self.init_estimate
#         self.init_estimate = self.freq_const + self.k_0 * self.lf_output

#         self.sin_out = -np.sin(self.phase_estimate)
#         next_cos_out = np.cos(self.phase_estimate)

#         self.phase_indicator = (
#             (np.isclose(self.wrap_phase(self.phase_estimate), self.target_phase, atol=self.atol)) and
#             (self.prev_cos_out <= self.cos_out >= next_cos_out)
#         )
#         self.cos_outs.append(self.cos_out)
#         self.phase_estimates.append(self.phase_estimate)
#         self.phase_indicators.append(self.phase_indicator)

#         self.prev_cos_out = self.cos_out
#         self.cos_out = next_cos_out
#         self.phase_estimate = next_phase

#         return self.phase_indicator

#     def step(self, point):
#         """
#         Moves through the state machine
#         """
#         pll_output = self.pll_detect(point[self.channel_idx])
#         if self.stimulate_flag and pll_output:
#             if self.stimulate is not None:
#                 self.stimulate()
#             self.stimulate_flag = False
#             return True
#         return False

#     def detected(self):
#         self.stimulate_flag = True

#     def not_detected(self):
#         self.stimulate_flag = False


# ================== STIMULATORS ==================


class DelayedStimulator(Stimulator, ABC):
    def __init__(self, config_dict, lsl_streamer=None, csv_recorder=None):
        print("F0")
        super().__init__(config_dict, lsl_streamer, csv_recorder)
        print("F1")

        if self.lsl_streamer is None:
            self.lsl_streamer = Dummy()
        if self.csv_recorder is None:
            self.csv_recorder = Dummy()

        print("F2")

        # Initialize stimulation delayer if requested
        stimulate_fn = lambda: self.send_stimulation("DELAY_STIM", True)

        print("F3")

        time_delay = not ((config_dict['min_delay'] == 0.0) and (config_dict['max_delay'] == 0.0) and (config_dict['inter_stim_delay'] == 0.0))

        print("F4")

        if time_delay:
            print("F5")
            stimulation_delayer = RandomTimingDelayer(config_dict, stimulate_fn=stimulate_fn)
            print("F6")
        elif config_dict['stim_delay_mode'] in ['Peak', 'Valley']:
            print("F7")
            stimulation_delayer = UpStateDelayer(config_dict, stimulate_fn=stimulate_fn)
            print("F8")
        else:
            print("F9")
            stimulation_delayer = None
        print("F10")
        self.delayer = stimulation_delayer

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
                res_del = self.delayer.step(filtered_point)
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


class SleepSpindleRealTimeStimulator(DelayedStimulator):
    def __init__(self, config_dict, lsl_streamer=None, csv_recorder=None):
        print("E0")
        super().__init__(config_dict, lsl_streamer, csv_recorder)

        print("E1")

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

        print("E2")

        # Initialize Alsa stuff
        # Open WAV file and set PCM device
        with wave.open(str(self._sound), 'rb') as f:
            print("E3")
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

            print("E4")

            try:
                print("E5")
                self.pcm = alsaaudio.PCM(channels=f.getnchannels(), rate=f.getframerate(), format=frmt, periodsize=self.periodsize, device=device)
            except alsaaudio.ALSAAudioError as e:
                print("E6")
                self.pcm = Dummy()
                raise e

            # Store data in list to avoid reopening the file
            self.wav_list = []
            print("E7")
            while True:
                data = f.readframes(self.periodsize)
                if data:
                    print("E8")
                    self.wav_list.append(data)
                else:
                    print("E9")
                    break

    def play_sound(self):
        '''
        Open the wav file and play a sound
        '''
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
        with self._lock:
            if self._thread is None:
                self._thread = Thread(target=self._t_sound, daemon=True)
                self._thread.start()

    def __del__(self):
        if hasattr(self, 'pcm'):
            del self.pcm


class SpindleTrainRealTimeStimulator(SleepSpindleRealTimeStimulator):
    def __init__(self, config_dict, lsl_streamer=None, csv_recorder=None):
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


class SlowOscillationStimulator(SleepSpindleRealTimeStimulator):
    def __init__(self, config_dict, lsl_streamer=None, csv_recorder=None):
        super().__init__(config_dict, lsl_streamer, csv_recorder)
        self.wait_t = .1  # Stimulate the first point of a detected SO only

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
        elif self.delayer is not None:
            self.delayer.not_detected()  # used by the SO phase delayer
        return res

