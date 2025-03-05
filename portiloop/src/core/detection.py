from abc import ABC, abstractmethod


class Detector(ABC):
    
    def __init__(self, threshold=None, channel=None):
        """
        Mandatory arguments are from the in the Portiloop GUI.
        """
        self.threshold = threshold
        self.channel = channel

    @abstractmethod
    def detect(self, datapoints):
        """
        Takes datapoints as input and outputs a detection signal.
        
        Args:
            datapoints: list of lists of n channels: may contain several datapoints.
                A datapoint is a list of n floats, 1 for each channel.
                In the current version of Portiloop, there is always only one datapoint per datapoints list.
        
        Returns:
            signal: Object: output detection signal (for instance, the output of a neural network);
                this output signal is the input of the Stimulator.stimulate method.
                If you don't mean to use a Stimulator, you can simply return None.
        """
        raise NotImplementedError

