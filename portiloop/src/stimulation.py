from abc import ABC, abstractmethod
from enum import Enum
import time
from threading import Thread, Lock
from pathlib import Path

from portiloop.src import ADS

if ADS:
    import alsaaudio
    
import wave
import pylsl
from scipy.signal import find_peaks


# Abstract interface for developers:

class Stimulator(ABC):

    @abstractmethod
    def stimulate(self, detection_signal):
        """
        Stimulates accordingly to the output of the Detector.
        
        Args:
            detection_signal: Object: the output of the Detector.add_datapoints method.
        """
        raise NotImplementedError
    
    def test_stimulus(self):
        """
        Optional: this is called when the 'Test stimulus' button is pressed.
        """
        pass

        
# Example implementation for sleep spindles

class SleepSpindleRealTimeStimulator(Stimulator):
    def __init__(self):
        self._sound = Path(__file__).parent.parent / 'sounds' / 'stimulus.wav'
        print(f"DEBUG:{self._sound}")
        self._thread = None
        self._lock = Lock()
        self.last_detected_ts = time.time()
        self.wait_t = 0.4  # 400 ms
        
        lsl_markers_info = pylsl.StreamInfo(name='Portiloop_stimuli',
                                  type='Markers',
                                  channel_count=1,
                                  channel_format='string',
                                  source_id='portiloop1')  # TODO: replace this by unique device identifier
        
        lsl_markers_info_fast = pylsl.StreamInfo(name='Portiloop_stimuli_fast',
                                  type='Markers',
                                  channel_count=1,
                                  channel_format='string',
                                  source_id='portiloop1')  # TODO: replace this by unique device identifier
        
        self.lsl_outlet_markers = pylsl.StreamOutlet(lsl_markers_info)
        self.lsl_outlet_markers_fast = pylsl.StreamOutlet(lsl_markers_info_fast)
        
        # Initialize Alsa stuff
        # Open WAV file and set PCM device
        with wave.open(str(self._sound), 'rb') as f: 
            device = 'default'

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

            self.pcm = alsaaudio.PCM(channels=f.getnchannels(), rate=f.getframerate(), format=format, periodsize=self.periodsize, device=device)
            
            # Store data in list to avoid reopening the file
            data = f.readframes(self.periodsize)
            self.wav_list = [data]
            while data:
                self.wav_list.append(data)
                data = f.readframes(self.periodsize)            

    def play_sound(self):
        '''
        Open the wav file and play a sound
        '''
        for data in self.wav_list:
            self.pcm.write(data) 
    
    def stimulate(self, detection_signal):
        for sig in detection_signal:
            # We detect a stimulation
            if sig:
                # Record time of stimulation
                ts = time.time()
                
                # Check if time since last stimulation is long enough
                if ts - self.last_detected_ts > self.wait_t:
                    if self.delayer is not None:
                        # If we have a delayer, notify it
                        self.delayer.detected()
                        # Send the LSL marer for the fast stimulation 
                        self.send_stimulation("FAST_STIM", False)
                    else:
                        self.send_stimulation("STIM", True)

                self.last_detected_ts = ts

    def send_stimulation(self, lsl_text, sound):
        # Send lsl stimulation
        self.lsl_outlet_markers.push_sample([lsl_text])
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

    def add_delayer(self, delayer):
        self.delayer = delayer
        self.delayer.stimulate = lambda: self.send_stimulation("DELAY_STIM", True)

# Class that delays stimulation to always stimulate peak or through
class UpStateDelayer:
    def __init__(self, sample_freq, spindle_freq, peak, time_to_buffer): 
        '''
        args:
            sample_freq: int -> Sampling frequency of signal in Hz
            time_to_wait: float -> Time to wait to build buffer in seconds
        '''
        # Get number of timesteps for a whole spindle
        self.spindle_timesteps = (1/spindle_freq) * sample_freq # s * 
        self.sample_freq = sample_freq
        self.buffer_size = 1.5 * self.spindle_timesteps
        self.peak = peak
        self.buffer = []
        self.time_to_buffer = time_to_buffer
        self.stimulate = None
        
        self.state = States.NO_SPINDLE

    def step(self, point):
        '''
        Step the delayer, ads a point to buffer if necessary.
        Returns True if stimulation is actually done
        '''
        if self.state == States.NO_SPINDLE:
            return False
        elif self.state == States.BUFFERING:
            self.buffer.append(point)
            # If we are done buffering, move on to the waiting stage
            if time.time() - self.time_started >= self.time_to_buffer:
                # Compute the necessary time to wait
                self.time_to_wait = self.compute_time_to_wait()
                self.state = States.DELAYING
                self.buffer = []
                self.time_started = time.time()
            return False
        elif self.state == States.DELAYING:
            # Check if we are done delaying
            if time.time() - self.time_started >= self.time_to_wait:
                # Actually stimulate the patient after the delay
                if self.stimulate is not None:
                    self.stimulate()
                # Reset state
                self.time_to_wait = -1
                self.state = States.NO_SPINDLE
                return True
            return False

    def detected(self):
        if self.state == States.NO_SPINDLE:
            self.state = States.BUFFERING
            self.time_started = time.time()

    def compute_time_to_wait(self):
        """
        Computes the time we want to wait in total based on the spindle frequency and the buffer
        """
        # If we want to look at the valleys, we search for peaks on the inversed signal
        if not self.peak: 
            self.buffer = -self.buffer

        # Returns the index of the last peak in the buffer
        peaks, _ = find_peaks(self.buffer, prominence=1)

        # Compute the time until next peak and return it
        return (len(self.buffer) - peaks[-1]) * (1 / self.sample_freq)

class States(Enum):
    NO_SPINDLE = 0
    BUFFERING = 1
    DELAYING = 2 


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    freq = 250
    spindle_freq = 10
    time = 10
    x = np.linspace(0, time * np.pi, num=time*freq)
    n = np.random.normal(scale=1, size=x.size)
    y = np.sin(x) + n
    plt.plot(x, y)
    plt.show()

