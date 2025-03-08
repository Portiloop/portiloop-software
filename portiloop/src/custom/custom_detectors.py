import time
import numpy as np

from portiloop.src.core.detection import Detector
from portiloop.src.core.constants import DEFAULT_MODEL_PATH
from portiloop.src.core.utils import Dummy

from portiloop.src import ADS
if ADS:
    from pycoral.utils import edgetpu
else:
    import tensorflow as tf


class SleepSpindleRealTimeDetector(Detector):
    def __init__(self, config_dict, lsl_streamer, csv_recorder):
        super().__init__(config_dict, lsl_streamer, csv_recorder)

        self.threshold = config_dict['threshold']
        self.channel = config_dict['channel_detection']
        self.record_csv = not isinstance(self.csv_recorder, Dummy)

        # threshold = 0.5,
        num_models_parallel = 8
        window_size = 54
        seq_stride = 42
        model_path = None
        verbose = False
        # channel = 2

        model_path = str(DEFAULT_MODEL_PATH if model_path is None else model_path)
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
        # If we don't have a detection, it means false for us.
        if len(res) == 0:
            res = [False]
        if self.record_csv:
            self.csv_recorder.append_detection_signal_buffer([int(r) for r in res])
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
