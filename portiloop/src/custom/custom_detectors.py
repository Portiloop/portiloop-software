import time

import numpy as np
from scipy import signal

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
        self.record_csv = not isinstance(self.csv_recorder, Dummy) and csv_recorder is not None

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
        return res, datapoints

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


class SlowOscillationDetector(Detector):
    def __init__(self, config_dict, lsl_streamer=None, csv_recorder=None):
        super().__init__(config_dict, lsl_streamer, csv_recorder)

        self.record_csv = not isinstance(self.csv_recorder, Dummy) and csv_recorder is not None

        fs = config_dict["frequency"]
        numtaps = 17
        verbose = False
        channel = config_dict["channel_detection"]
        record = False

        self.fs = fs
        self.numtaps = numtaps
        self.verbose = verbose
        self.channel = channel

        self.th_PaP = 75
        self.th_Neg = 40
        self.min_tNe = 125
        self.max_tNe = 1500
        self.max_tPo = 1000
        self.fmin_max = [0.16, 4]

        self.marker = []
        self.buffer = []
        self.filtered_buffer = []
        self.so_results = []
        self.count = 0
        self.record = record

        self.ssw_filter = signal.firwin(self.numtaps, self.fmin_max, fs=self.fs, pass_zero="bandpass")
        self.zi = signal.lfilter_zi(self.ssw_filter, 1)

        self.max_peak = None
        self.min_peak = None
        self.down_duration = None
        self.up_duration = None
        self.duration = None
        self.prev_signal = None

        self.init_segment()

    def init_segment(self):
        self.max_peak = -1
        self.min_peak = 1000
        self.down_duration = 0
        self.up_duration = 0
        self.duration = 0
        self.prev_signal = None

    def detect(self, datapoints):
        results = []
        for point in datapoints:
            self.count += 1
            result = self.detect_point(point[self.channel - 1])
            results.append(result)
            if result and self.record:
                self.so_results.append(self.count)
        if self.record_csv:
            self.csv_recorder.append_detection_signal_buffer([int(r) for r in results])
        return results, datapoints

    def detect_point(self, point):
        filtered_point, self.zi = signal.lfilter(
            self.ssw_filter, [1], [point], zi=self.zi
        )

        self.buffer.append(point)
        self.filtered_buffer.append(filtered_point[0])

        tsignal = filtered_point[0]
        if tsignal > self.max_peak:
            self.max_peak = tsignal
        if tsignal < self.min_peak:
            self.min_peak = tsignal

        if tsignal >= 0:
            self.up_duration += 1
        else:
            self.down_duration += 1

        self.duration += 1

        tp2p = abs(self.max_peak - self.min_peak)
        tneg = abs(self.min_peak)
        tne = self.down_duration / self.fs * 1000
        tpo = self.up_duration / self.fs * 1000
        tmfr = self.fs / self.duration

        self.prev_signal = self.prev_signal or tsignal

        if self.prev_signal * tsignal <= 0 < self.prev_signal:
            self.init_segment()

        self.prev_signal = tsignal

        if (
            tp2p > self.th_PaP
            and tneg > self.th_Neg
            and self.min_tNe < tne < self.max_tNe
            and tpo < self.max_tPo
        ):
            return True
        return False


def carrier_detect(data: np.ndarray,
                   fs,
                   th_PaP=75,
                   th_Neg=40,
                   filter_size=501,
                   min_tNe=125,
                   max_tNe=1500,  # duree_min_max_Neg
                   max_tPo=1000,  # duree_max_Pos
                   fmin_max=[0.16, 4],
                   verbose=False):
    """
    Output of the function
    SSW.marker : One marker field per epoch with the following fields,
    . nsamp is a table of [start end], the number of lines is the number of SSW
    . Neg is an array of DOWN phase amplitudes (uV) [filtered signal]
    . P2P is an array of peak-to-peak amplitudes (uV) [filtered signal]
    . Neg_raw is an array of DOWN phase amplitudes (uV) [raw signal]
    . P2P_raw is an array of peak-to-peak amplitudes (uV) [raw signal]
    . tNe is an array of duration of the DOWN phase
    . tPo is an array of duration of the UP phase
    . mfr is an array of the mean frequency"
    """
    # Extract parameters
    Ntrials, _ = data.shape

    # Design SSW filter
    wn = np.array(fmin_max) / (fs / 2)
    ssw_filter = signal.firwin(filter_size, wn, pass_zero=False)

    N_SSW = 0
    Duree = 0
    marker = []
    data_f_sw = np.zeros_like(data)

    for it in range(Ntrials):

        sig = data[it, :].astype(float)
        Duree += len(sig) / fs / 60  # minutes

        # Filter in SSW band
        sigf = signal.filtfilt(ssw_filter, [1], sig)
        data_f_sw[it, :] = sigf

        # Find zero crossings (from positives to negatives)
        f1 = (sigf[1:] * sigf[:-1]) < 0
        f2 = sigf[:-1] > 0
        n_zc = np.where(f1 & f2)[0]

        # Initialize arrays for SSW properties
        n_t = np.zeros((len(n_zc) - 1, 2), dtype=int)
        P2P = np.zeros(len(n_zc) - 1)
        Neg = np.zeros(len(n_zc) - 1)
        tNe = np.zeros(len(n_zc) - 1)
        tPo = np.zeros(len(n_zc) - 1)
        PaP_raw = np.zeros(len(n_zc) - 1)
        Neg_raw = np.zeros(len(n_zc) - 1)
        mfr = np.zeros(len(n_zc) - 1)
        keep = []

        # Analyze each zero crossing
        for i in range(len(n_zc) - 1):
            n_t[i, :] = [n_zc[i], n_zc[i + 1]]
            segment = sigf[n_zc[i] + 1 : n_zc[i + 1] - 1]  # exclude endpoints
            segNeg = segment < 0
            segPos = segment >= 0

            P2P[i] = abs(np.max(segment) - np.min(segment))
            Neg[i] = np.max(np.abs(segment[segNeg]))
            tNe[i] = np.sum(segNeg) / fs * 1000  # msec
            tPo[i] = np.sum(segPos) / fs * 1000  # msec
            mfr[i] = fs / (n_zc[i + 1] - n_zc[i])

            # Raw signal analysis
            raw_segment = sig[n_zc[i] + 1 : n_zc[i + 1] - 1]  # exclude endpoints
            PaP_raw[i] = abs(np.max(raw_segment) - np.min(raw_segment))
            Neg_raw[i] = np.max(np.abs(raw_segment[segNeg]))

            # Apply Carrier Criterion
            if (
                P2P[i] > th_PaP
                and Neg[i] > th_Neg
                and min_tNe < tNe[i] < max_tNe
                and tPo[i] < max_tPo
            ):
                keep.append(i)

        # Store SSW found in this trial
        marker.append(
            {
                "Thresholds_PaP_Neg": [th_PaP, th_Neg],
                "nsamp": n_t[keep],
                "P2P": P2P[keep],
                "Neg": Neg[keep],
                "tNe": tNe[keep],
                "tPo": tPo[keep],
                "P2P_raw": PaP_raw[keep],
                "Neg_raw": Neg_raw[keep],
                "mfr": mfr[keep],
            }
        )

        N_SSW += len(keep)
        if verbose:
            print(f"Trial {it}: ")
            print(marker[it]["nsamp"])
    # Calculate statistics
    Stat_SSW = {"N_SSW": N_SSW, "d_SSW": N_SSW / Duree, "duree": Duree}

    # Print detection results
    if verbose:
        print(f"\twe found {N_SSW} SSW over {Ntrials} trials")
        print(f"\t({N_SSW/Duree:.2f} SSW/minute)")

    # Prepare output
    SSW = {
        "markers": marker,
        "stat": Stat_SSW,
        "filtered_signals": data_f_sw,
    }

    return N_SSW, SSW
