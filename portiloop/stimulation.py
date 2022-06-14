from abc import ABC, abstractmethod
import time
from threading import Thread, Lock
from pathlib import Path
import alsaaudio
import wave
import pylsl


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
        self._sound = Path(__file__).parent / 'sounds' / 'stimulus.wav'
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
        self.lsl_outlet_markers = pylsl.StreamOutlet(lsl_markers_info)
        
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
            if sig:
                ts = time.time()
                if ts - self.last_detected_ts > self.wait_t:
                    with self._lock:
                        if self._thread is None:
                            self._thread = Thread(target=self._t_sound, daemon=True)
                            self._thread.start()
                self.last_detected_ts = ts
                
    def _t_sound(self):
        self.lsl_outlet_markers.push_sample(['STIM'])
        self.play_sound()
        with self._lock:
            self._thread = None
    
    def test_stimulus(self):
        with self._lock:
            if self._thread is None:
                self._thread = Thread(target=self._t_sound, daemon=True)
                self._thread.start()
