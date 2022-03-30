from abc import ABC, abstractmethod
import time


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

        
# Example implementation for sleep spindles:

class SleepSpindleRealTimeStimulator(Stimulator):
    def __init__(self):
        self.last_detected_ts = time.time()
        self.wait_t = 0.4  # 400 ms
    
    def stimulate(self, detection_signal):
        for sig in detection_signal:
            if sig:
                ts = time.time()
                if ts - self.last_detected_ts > self.wait_t:
                    print("stimulation")
                else:
                    print("same spindle")
                self.last_detected_ts = ts
