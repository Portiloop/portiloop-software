from abc import ABC, abstractmethod
import time
from pathlib import Path

from pycoral.utils import edgetpu
import numpy as np


# Abstract interface for developers:

class Detector(ABC):
    
    def __init__(self, threshold=None):
        """
        If implementing __init__() in your subclass, it must take threshold as a keyword argument.
        This is the value of the threshold that the user can set in the Portiloop GUI.
        Caution: even if you don't need this manual threshold in your application,
        your implementation of __init__() still needs to have this keyword argument.
        """
        self.threshold = threshold

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


# Example implementation for sleep spindles:
        
DEFAULT_MODEL_PATH = str(Path(__file__).parent / "models/portiloop_model_quant.tflite")
# print(DEFAULT_MODEL_PATH)

class SleepSpindleRealTimeDetector(Detector):
    def __init__(self, threshold=0.5, num_models_parallel=8, window_size=54, seq_stride=42, model_path=None, verbose=False, channel=2):
        model_path = DEFAULT_MODEL_PATH if model_path is None else model_path
        self.verbose = verbose
        self.channel = channel
        self.num_models_parallel = num_models_parallel
        
        self.interpreters = []
        for i in range(self.num_models_parallel):
            self.interpreters.append(edgetpu.make_interpreter(model_path))
            self.interpreters[i].allocate_tensors()
        self.interpreter_counter = 0
        
        self.input_details = self.interpreters[0].get_input_details()
        self.output_details = self.interpreters[0].get_output_details()
        
        self.buffer = []
        self.seq_stride = seq_stride
        self.window_size = window_size 
        
        self.stride_counters = [np.floor((self.seq_stride / self.num_models_parallel) * (i + 1)) for i in range(self.num_models_parallel)]
        for idx in reversed(range(1, len(self.stride_counters))):
            self.stride_counters[idx] -= self.stride_counters[idx-1]
        assert sum(self.stride_counters) == self.seq_stride, f"{self.stride_counters} does not sum to {self.seq_stride}"
            
        self.current_stride_counter = self.stride_counters[0] - 1
        
        super().__init__(threshold)

    def detect(self, datapoints):
        res = []
        for inp in datapoints:
            result = self.add_datapoint(inp)
            if result is not None:
                res.append(result >= self.threshold)
        return res

    def add_datapoint(self, input_float):
        input_float = input_float[self.channel - 1]
        result = None
        self.buffer.append(input_float)
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[1:]
            self.current_stride_counter += 1
            if self.current_stride_counter == self.stride_counters[self.interpreter_counter]:
                result = self.call_model(self.interpreter_counter, self.buffer)
                self.interpreter_counter += 1
                self.interpreter_counter %= self.num_models_parallel
                self.current_stride_counter = 0
        return result

    def call_model(self, idx, input_float=None):
        if input_float is None:
            # For debugging purposes
            input_shape = self.input_details[0]['shape']
            input = np.array(np.random.random_sample(input_shape), dtype=np.int8)
        else:
            # Convert float input to Int
            input_scale, input_zero_point = self.input_details[0]["quantization"]
            input = np.asarray(input_float) / input_scale + input_zero_point
            input = input.astype(self.input_details[0]["dtype"])
        input = input.reshape((1, 1, -1))

        # FIXME: bad sequence length: 50 instead of 1:
        # self.interpreters[idx].set_tensor(self.input_details[0]['index'], input)
        #
        # if self.verbose:
        #     start_time = time.time()
        #
        # self.interpreters[idx].invoke()
        #
        # if self.verbose:
        #     end_time = time.time()
        # output = self.interpreters[idx].get_tensor(self.output_details[0]['index'])
        # output_scale, output_zero_point = self.input_details[0]["quantization"]
        # output = float(output - output_zero_point) * output_scale
        output = np.random.uniform()  # FIXME: remove

        if self.verbose:
            print(f"Computed output {output} in {end_time - start_time} seconds")

        return output
    
    