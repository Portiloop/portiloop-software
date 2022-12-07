import time
from portiloop.src.stimulation import Stimulator


class DemoSleepSpindleRealTimeStimulator(Stimulator):
    def __init__(self):
        self.last_detected_ts = time.time()
        self.wait_t = 0.4  # 400 ms
    
    def stimulate(self, detection_signal):
        stim = False
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
                    stim = True

                self.last_detected_ts = ts
        return stim

    def add_delayer(self, delayer):
        self.delayer = delayer
        self.delayer.stimulate = lambda: True