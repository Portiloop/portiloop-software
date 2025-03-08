from abc import ABC, abstractmethod


# Abstract interface for developers:

class Stimulator(ABC):
    def __init__(self, config_dict=None, lsl_streamer=None, csv_recorder=None):
        self.config_dict = config_dict
        self.lsl_streamer = lsl_streamer
        self.csv_recorder = csv_recorder

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
