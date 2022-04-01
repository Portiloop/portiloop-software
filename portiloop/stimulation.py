from abc import ABC, abstractmethod
import time
from playsound import playsound
from threading import Thread, Lock
from pathlib import Path


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

        
# Example implementation for sleep spindles

class SleepSpindleRealTimeStimulator(Stimulator):
    def __init__(self):
        self._sound = Path(__file__).parent / 'sounds' / 'stimulus.wav'
        print(f"DEBUG:{self._sound}")
        self._thread = None
        self._lock = Lock()
        self.last_detected_ts = time.time()
        self.wait_t = 0.4  # 400 ms
    
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
        playsound(self._sound)
        with self._lock:
            self._thread = None
