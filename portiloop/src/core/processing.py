from abc import ABC, abstractmethod


class Processor(ABC):
    def __init__(self, config_dict=None, lsl_streamer=None, csv_recorder=None):
        self.config_dict = config_dict
        self.lsl_streamer = lsl_streamer
        self.csv_recorder = csv_recorder

    @abstractmethod
    def filter(self, value):
        """
        value: a numpy array of shape (data series, channels)
        """
        raise NotImplementedError
