from abc import ABC, abstractmethod
import time
from pathlib import Path
from portiloop.src import ADS

if ADS:
    from pycoral.utils import edgetpu
else:
    import tensorflow as tf
import numpy as np


# Abstract interface for developers:

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


# Example implementation for sleep spindles:
        
DEFAULT_MODEL_PATH = str(Path(__file__).parent.parent / "models/portiloop_model_quant.tflite")
# print(DEFAULT_MODEL_PATH)
DEMO_MODEL_PATH = str(Path(__file__).parent.parent / "models/demo_model.tflite")
def sigmoid(x):
    if x > -4.05384:
        return 0.978
    else:
        return 0.014

class DataBuffer:
    """
    A class to get the data in the right format for the model from a stream of data
    """
    def __init__(self, window_size, num_channels):
        self.window_size = window_size

        # Compute the total number of points to keep in memory as the buffer
        self.data = np.zeros((num_channels, window_size))

    def step(self, point):
        # Shift the data
        self.data[:, :-1] = self.data.copy()[:, 1:]
        self.data[:, -1] = point
        current_data = self.data.copy()
        current_data = np.expand_dims(current_data, 1)
        return current_data


class DemoDetector(Detector):
    def __init__(self, threshold=0.5, channel=None, model_path=None):
        model_path = DEMO_MODEL_PATH if model_path is None else model_path
        self.interpreter = edgetpu.make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.window_size = 1250
        self.buffer = DataBuffer(self.window_size, 4)
        self.index = 0
        self.threshold = threshold
        
    def detect(self, datapoints):
        out_positive = []
        for point in datapoints:
            window = self.buffer.step(point)
            self.index += 1
            if self.index > self.window_size and self.index % 50 == 0:
                output = [self.forward_tflite(np.expand_dims(wind, -1)) for wind in window]
                for idx, out in enumerate(output):
                    out = sigmoid(out)
                    if out > self.threshold:
                        out_positive.append(idx)
        return out_positive
                    
    def forward_tflite(self, input):
        # convert input to int 
        input_scale, input_zero_point = self.input_details[0]["quantization"]
        input = np.asarray(input) / input_scale + input_zero_point
        input_data_x = input.astype(self.input_details[0]["dtype"])

        # Test the model on random input data.
        input_shape_x = self.input_details[0]['shape']

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data_x)

        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data_y = self.interpreter.get_tensor(self.output_details[0]['index'])

        output_scale, output_zero_point = self.output_details[0]["quantization"]
        output_data_y = (int(output_data_y) - output_zero_point) * output_scale

        return output_data_y


class SleepSpindleRealTimeDetector(Detector):
    def __init__(self,
                 threshold=0.5,
                 num_models_parallel=8,
                 window_size=54,
                 seq_stride=42,
                 model_path=None,
                 verbose=False,
                 channel=2):
        model_path = DEFAULT_MODEL_PATH if model_path is None else model_path
        self.verbose = verbose
        self.num_models_parallel = num_models_parallel
        
        self.interpreters = []
        for i in range(self.num_models_parallel):
            if ADS:
                self.interpreters.append(edgetpu.make_interpreter(model_path))
            else:
                self.interpreters.append(tf.lite.Interpreter(model_path=model_path))
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
        
        self.h = [np.zeros((1, 7), dtype=np.int8) for _ in range(self.num_models_parallel)]
            
        self.current_stride_counter = self.stride_counters[0] - 1
        
        super().__init__(threshold, channel)

    def detect(self, datapoints):
        """
        Takes datapoints as input and outputs a detection signal.
        datapoints is a list of lists of n channels: may contain several datapoints.
        
        The output signal is a list of tuples (is_spindle, is_train_of_spindles).
        
        """
        res = []
        for inp in datapoints:
            result = self.add_datapoint(inp)
            if result is not None:
                res.append(result >= self.threshold)
        return res

    def add_datapoint(self, input_float):
        '''
        Add one datapoint to the buffer
        '''
        input_float = input_float[self.channel - 1]
        result = None
        # Add to current buffer
        self.buffer.append(input_float)
        if len(self.buffer) > self.window_size:
            # Remove the end of the buffer
            self.buffer = self.buffer[1:]
            self.current_stride_counter += 1
            if self.current_stride_counter == self.stride_counters[self.interpreter_counter]:
                # If we have reached the next window size, we send the current buffer to the inference function and update the hidden state
                result, self.h[self.interpreter_counter] = self.forward_tflite(self.interpreter_counter, self.buffer, self.h[self.interpreter_counter])
                self.interpreter_counter += 1
                self.interpreter_counter %= self.num_models_parallel
                self.current_stride_counter = 0
        return result
        
    def forward_tflite(self, idx, input_x, input_h):
        input_details = self.interpreters[idx].get_input_details()
        output_details = self.interpreters[idx].get_output_details()
        
        # convert input to int 
        input_scale, input_zero_point = input_details[1]["quantization"]
        input_x = np.asarray(input_x) / input_scale + input_zero_point
        input_data_x = input_x.astype(input_details[1]["dtype"])
        input_data_x = input_data_x.reshape((1, 1) + input_data_x.shape)

        # input_scale, input_zero_point = input_details[0]["quantization"]
        # input = np.asarray(input) / input_scale + input_zero_point

        # Test the model on random input data.
        input_shape_h = input_details[0]['shape']
        input_shape_x = input_details[1]['shape']

        # input_data_h = np.array(np.random.random_sample(input_shape_h), dtype=np.int8)
        # input_data_x = np.array(np.random.random_sample(input_shape_x), dtype=np.int8)
        self.interpreters[idx].set_tensor(input_details[0]['index'], input_h)
        self.interpreters[idx].set_tensor(input_details[1]['index'], input_data_x)
        
        if self.verbose:
            start_time = time.time()

        self.interpreters[idx].invoke()
        
        if self.verbose:
            end_time = time.time()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data_h = self.interpreters[idx].get_tensor(output_details[0]['index'])
        output_data_y = self.interpreters[idx].get_tensor(output_details[1]['index'])

        output_scale, output_zero_point = output_details[1]["quantization"]
        output_data_y = (int(output_data_y) - output_zero_point) * output_scale
        
        if self.verbose:
            print(f"Computed output {output_data_y} in {end_time - start_time} seconds")

        return output_data_y, output_data_h
