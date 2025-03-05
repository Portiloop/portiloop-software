import copy
import time
from threading import Thread, Lock
from pathlib import Path

import wave

from portiloop.src.core.stimulation import Stimulator, time
from portiloop.src.core.utils import Dummy

from portiloop.src import ADS
if ADS:
    import alsaaudio


SOUNDS_FOLDER = Path(__file__).parent.parent.parent / 'sounds'  # TODO: remove hardcoded pathes


class SleepSpindleRealTimeStimulator(Stimulator):
    def __init__(self, soundname=None, lsl_streamer=Dummy(), sham=False):
        """
        params:
            stimulation_delay (float): simple delay between a detection and a stimulation
            inter_stim_delay (float): time to wait between a stimulation and the next detection
        """
        if soundname is None:
            self.soundname = 'stimulus.wav' # CHANGE HERE TO THE SOUND THAT YOU WANT. ONLY ADD THE FILE NAME, NOT THE ENTIRE PATH
        else:
            self.soundname = soundname
        self._sound = SOUNDS_FOLDER / self.soundname
        self._thread = None
        self._lock = Lock()
        self.last_detected_ts = time.time()
        self.wait_t = 0.4  # 400 ms
        self.delayer = None
        self.lsl_streamer = lsl_streamer
        self.sham = sham

        # Initialize Alsa stuff
        # Open WAV file and set PCM device
        with wave.open(str(self._sound), 'rb') as f:
            device = 'softvol'

            self.duration = f.getnframes() / float(f.getframerate())

            format = None

            # 8bit is unsigned in wav files
            if f.getsampwidth() == 1:
                format = alsaaudio.PCM_FORMAT_U8
            # Otherwise we assume signed data, little endian
            elif f.getsampwidth() == 2:
                format = alsaaudio.PCM_FORMAT_S16_LE
            elif f.getsampwidth() == 3:
                format = alsaaudio.PCM_FORMAT_S24_3LE
            elif f.getsampwidth() == 4:
                format = alsaaudio.PCM_FORMAT_S32_LE
            else:
                raise ValueError('Unsupported format')

            self.periodsize = f.getframerate() // 8

            try:
                self.pcm = alsaaudio.PCM(channels=f.getnchannels(), rate=f.getframerate(), format=format, periodsize=self.periodsize, device=device)
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

    def stimulate(self, detection_signal):
        stim = []
        for sig in detection_signal:
            # We detect a stimulation
            if sig:
                # Record time of stimulation
                ts = time.time()

                # Check if time since last stimulation is long enough
                if ts - self.last_detected_ts > self.wait_t:
                    stim.append(True)
                    if not isinstance(self.delayer, Dummy):
                        # If we have a delayer, notify it
                        self.delayer.detected()
                        # Send the LSL marer for the fast stimulation
                        self.send_stimulation("FAST_STIM", False)
                    else:
                        self.send_stimulation("STIM", not self.sham)

                self.last_detected_ts = ts
            else:
                stim.append(False)
        return stim

    def send_stimulation(self, lsl_text, sound):
        # Send lsl stimulation
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

        # print(f"DEBUG: Stimulation delay: {((self.end - start) * 1000):.2f}ms")

    def add_delayer(self, delayer):
        self.delayer = delayer
        self.delayer.stimulate = lambda: self.send_stimulation("DELAY_STIM", True)

    def __del__(self):
        # print("DEBUG: releasing PCM")
        del self.pcm


class SpindleTrainRealTimeStimulator(SleepSpindleRealTimeStimulator):
    def __init__(self):
        self.max_spindle_train_t = 6.0
        super().__init__()

    def stimulate(self, detection_signal):
        for sig in detection_signal:
            # We detect a stimulation
            if sig:
                # Record time of stimulation
                ts = time.time()

                # Check if time since last stimulation is long enough
                elapsed = ts - self.last_detected_ts
                if self.wait_t < elapsed < self.max_spindle_train_t:
                    if self.delayer is not None:
                        # If we have a delayer, notify it
                        self.delayer.detected()
                        # Send the LSL marer for the fast stimulation
                        self.send_stimulation("FAST_STIM", False)
                    else:
                        self.send_stimulation("STIM", True)

                self.last_detected_ts = ts


class IsolatedSpindleRealTimeStimulator(SpindleTrainRealTimeStimulator):
    def stimulate(self, detection_signal):
        for sig in detection_signal:
            # We detect a stimulation
            if sig:
                # Record time of stimulation
                ts = time.time()

                # Check if time since last stimulation is long enough
                elapsed = ts - self.last_detected_ts
                if self.max_spindle_train_t < elapsed:
                    if self.delayer is not None:
                        # If we have a delayer, notify it
                        self.delayer.detected()
                        # Send the LSL marer for the fast stimulation
                        self.send_stimulation("FAST_STIM", False)
                    else:
                        self.send_stimulation("STIM", True)

                self.last_detected_ts = ts


class AlternatingStimulator(Stimulator):
    def __init__(self, soundname=None, lsl_streamer=Dummy(), stim_interval=0.250):
        """
        params:
            stimulation_delay (float): simple delay between a detection and a stimulation
            inter_stim_delay (float): time to wait between a stimulation and the next detection
        """
        self.pos_soundname = 'syllPos120.wav' # CHANGE HERE TO THE SOUND THAT YOU WANT. ONLY ADD THE FILE NAME, NOT THE ENTIRE PATH
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

            format = None

            # 8bit is unsigned in wav files
            if f.getsampwidth() == 1:
                format = alsaaudio.PCM_FORMAT_U8
            # Otherwise we assume signed data, little endian
            elif f.getsampwidth() == 2:
                format = alsaaudio.PCM_FORMAT_S16_LE
            elif f.getsampwidth() == 3:
                format = alsaaudio.PCM_FORMAT_S24_3LE
            elif f.getsampwidth() == 4:
                format = alsaaudio.PCM_FORMAT_S32_LE
            else:
                raise ValueError('Unsupported format')

            self.periodsize = f.getframerate() // 8

            try:
                self.pcm = alsaaudio.PCM(channels=f.getnchannels(), rate=f.getframerate(), format=format, periodsize=self.periodsize, device=device)
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

            format = None

            # 8bit is unsigned in wav files
            if f.getsampwidth() == 1:
                format = alsaaudio.PCM_FORMAT_U8
            # Otherwise we assume signed data, little endian
            elif f.getsampwidth() == 2:
                format = alsaaudio.PCM_FORMAT_S16_LE
            elif f.getsampwidth() == 3:
                format = alsaaudio.PCM_FORMAT_S24_3LE
            elif f.getsampwidth() == 4:
                format = alsaaudio.PCM_FORMAT_S32_LE
            else:
                raise ValueError('Unsupported format')

            self.periodsize = f.getframerate() // 8

            try:
                self.pcm = alsaaudio.PCM(channels=f.getnchannels(), rate=f.getframerate(), format=format, periodsize=self.periodsize, device=device)
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
        # We ignore the input signal and simply make sure we stimulate at the given interval
        current_time = time.time()
        if current_time - self.last_stim >= self.stim_interval:
            # Check if we are in the inverted phase:
            if self.stim_polarity:
                stim_text = 'STIM_POS'
            else:
                stim_text = 'STIM_NEG'
            self.send_stimulation(stim_text, True)
            self.last_stim = current_time
            self.stim_polarity = not self.stim_polarity

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

    def add_delayer(self, delayer):
        self.delayer = delayer
        self.delayer.stimulate = lambda: self.send_stimulation("DELAY_STIM", False)
