from abc import ABC, abstractmethod
import time
from pathlib import Path
from portiloop.src import ADS

if ADS:
    from pycoral.utils import edgetpu
else:
    import tensorflow as tf
import numpy as np
from scipy import signal

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

DEFAULT_MODEL_PATH = str(
    Path(__file__).parent.parent / "models/portiloop_model_quant.tflite"
)
# print(DEFAULT_MODEL_PATH)


class SleepSpindleRealTimeDetector(Detector):
    def __init__(
        self,
        threshold=0.5,
        num_models_parallel=8,
        window_size=54,
        seq_stride=42,
        model_path=None,
        verbose=False,
        channel=2,
    ):
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

        self.stride_counters = [
            np.floor((self.seq_stride / self.num_models_parallel) * (i + 1))
            for i in range(self.num_models_parallel)
        ]
        for idx in reversed(range(1, len(self.stride_counters))):
            self.stride_counters[idx] -= self.stride_counters[idx - 1]
        assert (
            sum(self.stride_counters) == self.seq_stride
        ), f"{self.stride_counters} does not sum to {self.seq_stride}"

        self.h = [
            np.zeros((1, 7), dtype=np.int8) for _ in range(self.num_models_parallel)
        ]

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
        # If we don't have a detection, it means false for us.
        if len(res) == 0:
            res = [False]
        return res

    def add_datapoint(self, input_float):
        """
        Add one datapoint to the buffer
        """
        input_float = input_float[self.channel - 1]
        result = None
        # Add to current buffer
        self.buffer.append(input_float)
        if len(self.buffer) > self.window_size:
            # Remove the end of the buffer
            self.buffer = self.buffer[1:]
            self.current_stride_counter += 1
            if (
                self.current_stride_counter
                == self.stride_counters[self.interpreter_counter]
            ):
                # If we have reached the next window size, we send the current buffer to the inference function and update the hidden state
                result, self.h[self.interpreter_counter] = self.forward_tflite(
                    self.interpreter_counter,
                    self.buffer,
                    self.h[self.interpreter_counter],
                )
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
        input_shape_h = input_details[0]["shape"]
        input_shape_x = input_details[1]["shape"]

        # input_data_h = np.array(np.random.random_sample(input_shape_h), dtype=np.int8)
        # input_data_x = np.array(np.random.random_sample(input_shape_x), dtype=np.int8)
        self.interpreters[idx].set_tensor(input_details[0]["index"], input_h)
        self.interpreters[idx].set_tensor(input_details[1]["index"], input_data_x)

        if self.verbose:
            start_time = time.time()

        self.interpreters[idx].invoke()

        if self.verbose:
            end_time = time.time()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data_h = self.interpreters[idx].get_tensor(output_details[0]["index"])
        output_data_y = self.interpreters[idx].get_tensor(output_details[1]["index"])

        output_scale, output_zero_point = output_details[1]["quantization"]
        output_data_y = (int(output_data_y) - output_zero_point) * output_scale

        if self.verbose:
            print(f"Computed output {output_data_y} in {end_time - start_time} seconds")

        return output_data_y, output_data_h

class SlowOscillationDetector:
    def __init__(
        self,
        fs=250,
        # window_size=1501,
        numtaps=501,
        th_PaP=75,
        th_Neg=40,
        min_tNe=125,
        max_tNe=1500,
        max_tPo=1000,
        fmin_max=[0.16, 4],
        verbose=True,
        channel=2,
        threshold=0.5,
    ):
        self.fs = fs
        self.th_PaP = th_PaP
        self.th_Neg = th_Neg
        self.min_tNe = min_tNe
        self.max_tNe = max_tNe
        self.max_tPo = max_tPo
        self.fmin_max = fmin_max
        self.verbose = verbose
        self.channel = channel

        # self.window_size = window_size

        self.N_SSW = 0
        self.Duree = 0
        self.marker = []
        self.buffer = []
        self.filtered_buffer = []
        self.data_f_sw = None

        self.wn = np.array(fmin_max) / (fs / 2)
        self.numtaps = numtaps
        if self.verbose:
            print(self.wn, self.numtaps)
        self.ssw_filter = signal.firwin(5, self.wn, pass_zero=False)

        # Initialize filter state
        self.zi = signal.lfilter_zi(self.ssw_filter, 1)

        self.so_results = []
        self.count = 0

    def detect(self, datapoints):
        results = []
        for point in datapoints:
            self.count += 1
            result = self.detect_point(point[self.channel - 1])
            results.append(result)
            if result:
                self.so_results.append(self.count)
        return results

    def detect_point(self, point):
        # Apply online filtering
        filtered_point, self.zi = signal.lfilter(
            self.ssw_filter, [1], [point], zi=self.zi
        )

        self.buffer.append(point)
        self.filtered_buffer.append(filtered_point[0])

        if len(self.buffer) <= self.numtaps*3:
            return False

        # Analyze the window
        sig = np.array(self.buffer)
        filtered_signal = np.array(self.filtered_buffer)

        # Remove oldest point from buffers
        self.buffer.pop(0)
        self.filtered_buffer.pop(0)

        # find zero crossings
        f1 = filtered_signal[1:] * filtered_signal[:-1] < 0
        f2 = filtered_signal[:-1] > 0
        n_zc = np.where(f1 & f2)[0]

        if len(n_zc) < 1:
            return False

        n_t = np.zeros((len(n_zc) - 1, 2), dtype=int)
        P2P = np.zeros(len(n_zc) - 1)
        Neg = np.zeros(len(n_zc) - 1)
        tNe = np.zeros(len(n_zc) - 1)
        tPo = np.zeros(len(n_zc) - 1)
        PaP_raw = np.zeros(len(n_zc) - 1)
        Neg_raw = np.zeros(len(n_zc) - 1)
        mfr = np.zeros(len(n_zc) - 1)

        if self.verbose:
            print(len(n_zc))

        # Analyze each zero crossing
        for i in range(len(n_zc) - 1):
            n_t[i, :] = [n_zc[i], n_zc[i + 1]]
            segment = filtered_signal[
                n_zc[i] + 1 : n_zc[i + 1] - 1
            ]  # exclude endpoints
            segNeg = segment < 0
            segPos = segment >= 0

            P2P[i] = abs(np.max(segment) - np.min(segment))
            Neg[i] = np.max(np.abs(segment[segNeg]))
            tNe[i] = np.sum(segNeg) / self.fs * 1000  # msec
            tPo[i] = np.sum(segPos) / self.fs * 1000  # msec
            mfr[i] = self.fs / (n_zc[i + 1] - n_zc[i])

            # Raw signal analysis
            raw_segment = sig[n_zc[i] + 1 : n_zc[i + 1] - 1]  # exclude endpoints
            PaP_raw[i] = abs(np.max(raw_segment) - np.min(raw_segment))
            Neg_raw[i] = np.max(np.abs(raw_segment[segNeg]))

            # Apply Carrier Criterion
            if (
                P2P[i] > self.th_PaP
                and Neg[i] > self.th_Neg
                and self.min_tNe < tNe[i] < self.max_tNe
                and tPo[i] < self.max_tPo
            ):
                return True

        return False
