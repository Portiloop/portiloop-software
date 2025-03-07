from abc import ABC, abstractmethod


class Detector(ABC):
    def __init__(self, config_dict=None, lsl_streamer=None):
        self.config_dict = config_dict
        self.lsl_streamer = lsl_streamer

    @abstractmethod
    def detect(self, datapoints):
        """
        Takes datapoints as input and outputs a detection signal.

        Args:
            datapoints: 2d array of n channels: may contain several datapoints.
                A datapoint is an array of n floats, 1 for each channel.
                In the current version of Portiloop, there is always only one datapoint per datapoints list.

        Returns:
            detection_signal: Object: arbitrary output detection signal (for instance, the output of a neural network);
                this output signal is the input of the Stimulator.stimulate method.
                If you don't mean to use a Stimulator, you can simply set to None.
        """
        raise NotImplementedError
