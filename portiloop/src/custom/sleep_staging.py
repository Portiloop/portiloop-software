"""
Sleep staging pipelines: every 30 s, a stage (W, N1, N2, N3 or REM) from the Fpz - Cz derivation, with TinySleepNet
(TensorFlow Lite) or NanoSleepNet-TCN (NumPy). Use and output are described in the README.
"""
import hashlib
import logging
from collections import deque
from pathlib import Path

import numpy as np
from scipy.signal import butter, firwin, lfilter, sosfilt
from scipy.special import erf

from portiloop.src.core.capture import dump_metadata
from portiloop.src.core.constants import DEFAULT_MODEL_PATH
from portiloop.src.core.detection import Detector
from portiloop.src.core.utils import Dummy

from portiloop.src import ADS
if ADS:
    import tflite_runtime.interpreter as tflite
else:
    import tensorflow as tf

logger = logging.getLogger(__name__)

STAGES = ("W", "N1", "N2", "N3", "REM")  # detection signal 1 to 5
NO_STAGE = 0  # detection signal before the first stage
DEFAULT_FPZ_CHANNEL = 8  # CLAS protocol montage (Cz on channel 6, the default detection channel); set it in the GUI for other montages
MODELS_FOLDER = DEFAULT_MODEL_PATH.parent

# The models were trained on Sleep-EDF Fpz-Cz: 100 Hz, 30 s epochs, +-192 uV
FS_IN = 250
UP, DOWN = 2, 5  # 250 Hz * 2 / 5 = 100 Hz
EPOCH_SAMPLES = 3000  # 30 s at 100 Hz
EPOCH_ROWS = EPOCH_SAMPLES * DOWN // UP  # 30 s at 250 Hz: recording rows per epoch
EDF_LIMIT_UV = 192.0
CHUNK = 25  # raw points per preprocessing call (0.1 s); a performance setting, results do not depend on it
HIGHPASS_SOS = butter(1, 0.5, btype="highpass", fs=FS_IN, output="sos")
# The anti-aliasing filter that scipy.signal.resample_poly(x, 2, 5) designs by default
RESAMPLE_FIR = firwin(20 * max(UP, DOWN) + 1, 1.0 / max(UP, DOWN), window=("kaiser", 5.0)) * UP
LFS_POINTER = b"version https://git-lfs"  # start of the placeholder Git LFS leaves when git-lfs is not installed


class Preprocessor:
    """
    Model input from the raw Portiloop signal, causally, in any chunk sizes:
    Fpz - Cz (both referenced to the same electrode), minus its first value
    -> 1st-order Butterworth high-pass at 0.5 Hz (Sleep-EDF's recorder high-passes at 0.5 Hz)
    -> 250 Hz to 100 Hz: zero-stuffing by 2, the low-pass FIR that scipy's resample_poly designs, every 5th sample
    -> clip to +-192 uV (Sleep-EDF's range)
    -> 30 s epochs.
    """
    def __init__(self, fpz, cz):
        self.fpz = fpz  # 0-based
        self.cz = cz
        self.offset = None
        self.highpass_state = np.zeros((HIGHPASS_SOS.shape[0], 2))
        self.fir_state = np.zeros(len(RESAMPLE_FIR) - 1)
        self.nb_stuffed = 0  # samples of the zero-stuffed (500 Hz) signal so far
        self.epoch = []

    def add_datapoints(self, points):
        """
        points: (n, channels) raw uV. Returns the list of epochs completed by these points.
        """
        points = np.asarray(points, dtype=np.float64)
        x = points[:, self.fpz] - points[:, self.cz]
        if self.offset is None:
            self.offset = x[0]
        x, self.highpass_state = sosfilt(HIGHPASS_SOS, x - self.offset, zi=self.highpass_state)
        stuffed = np.zeros(len(x) * UP)
        stuffed[::UP] = x
        y, self.fir_state = lfilter(RESAMPLE_FIR, [1.0], stuffed, zi=self.fir_state)
        # keep the samples whose index in the whole zero-stuffed signal is a multiple of DOWN
        first = (-self.nb_stuffed) % DOWN
        self.nb_stuffed += len(stuffed)
        self.epoch.extend(np.clip(y[first::DOWN], -EDF_LIMIT_UV, EDF_LIMIT_UV).tolist())
        epochs = []
        while len(self.epoch) >= EPOCH_SAMPLES:
            epochs.append(np.array(self.epoch[:EPOCH_SAMPLES], dtype=np.float32))
            del self.epoch[:EPOCH_SAMPLES]
        return epochs


class TinySleepNetStep:
    """
    TinySleepNet as a TensorFlow Lite model; the LSTM state is carried from epoch to epoch.

    probe.npz holds one epoch and the LSTM state the original model reaches on it from a zero state: it tells the
    two state outputs of the TFLite model (same shape) apart.
    """
    NAME = "TinySleepNet"
    FOLDER = MODELS_FOLDER / 'tinysleepnet'
    FILES = ("tinysleepnet_float.tflite", "probe.npz")

    def __init__(self, model_folder):
        model_path = str(Path(model_folder) / self.FILES[0])
        if ADS:
            self.interpreter = tflite.Interpreter(model_path=model_path)
        else:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        input_index = {d['name']: d['index'] for d in self.interpreter.get_input_details()}
        self.input_x = input_index['x']
        self.input_c = input_index['c']
        self.input_h = input_index['h']
        state_outputs = []
        for output in self.interpreter.get_output_details():
            if tuple(output['shape']) == (1, len(STAGES)):
                self.output_logits = output['index']
            else:
                state_outputs.append(output['index'])

        with np.load(Path(model_folder) / self.FILES[1]) as probe:
            probe_x, probe_c = probe['x'], probe['c']
        self.reset()
        first, second = self.forward_tflite(probe_x, state_outputs)
        if np.abs(first - probe_c).max() < np.abs(second - probe_c).max():
            self.output_c, self.output_h = state_outputs
        else:
            self.output_h, self.output_c = state_outputs
        self.reset()

    def reset(self):
        self.c = np.zeros((1, 128), dtype=np.float32)
        self.h = np.zeros((1, 128), dtype=np.float32)

    def forward_tflite(self, epoch, outputs):
        epoch = np.asarray(epoch, dtype=np.float32).reshape(1, EPOCH_SAMPLES, 1, 1)
        self.interpreter.set_tensor(self.input_x, epoch)
        self.interpreter.set_tensor(self.input_c, self.c)
        self.interpreter.set_tensor(self.input_h, self.h)
        self.interpreter.invoke()
        return [self.interpreter.get_tensor(index) for index in outputs]

    def step(self, epoch):
        """
        Logits (W, N1, N2, N3, REM) of one epoch; the LSTM state carries over to the next call.
        """
        logits, self.c, self.h = self.forward_tflite(epoch, (self.output_logits, self.output_c, self.output_h))
        return logits[0]


def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def conv1d(x, w, stride=1, groups=1):
    """
    Grouped 1-D convolution without bias, with TensorFlow-like "same" padding (the model's SamePadConv1d).

    x: (in_channels, length); w: (out_channels, in_channels / groups, kernel).
    """
    length = x.shape[1]
    kernel = w.shape[2]
    out_length = int(np.ceil(length / stride))
    pad = max((out_length - 1) * stride + kernel - length, 0)
    x = np.pad(x, ((0, 0), (pad // 2, pad - pad // 2)))
    out = []
    for x_group, w_group in zip(np.split(x, groups), np.split(w, groups)):
        # kernel tap j sees the samples j, j + stride, j + 2 * stride, ...
        out.append(sum(w_group[:, :, j] @ x_group[:, j:j + stride * out_length:stride] for j in range(kernel)))
    return np.concatenate(out)


def channel_shuffle(x, groups):
    """
    x: (channels, length)
    """
    return x.reshape(groups, x.shape[0] // groups, -1).transpose(1, 0, 2).reshape(x.shape)


class NanoSleepNetStep:
    """
    NanoSleepNet-TCN (causal) in NumPy. Each epoch gives a 64-value token (epoch encoder); the temporal network
    stages it from the tokens of the last 10 epochs (fewer at the start of a recording), as in training.

    nanosleepnet.npz holds the weights of the trained PyTorch model, each BatchNorm as a per-channel scale and shift.
    """
    NAME = "NanoSleepNet"
    FOLDER = MODELS_FOLDER / 'nanosleepnet'
    FILES = ("nanosleepnet.npz",)
    WINDOW = 10  # epochs seen by the temporal network: the length of the training sequences

    def __init__(self, model_folder):
        with np.load(Path(model_folder) / self.FILES[0]) as weights:
            self.w = {name: weights[name].astype(np.float64) for name in weights.files}
        nb_blocks = len([name for name in self.w if name.endswith(".dilation")])
        self.blocks = [(f"block{i}", int(self.w[f"block{i}.dilation"])) for i in range(nb_blocks)]
        self.reset()

    def reset(self):
        self.tokens = deque(maxlen=self.WINDOW)

    def batch_norm(self, name, x):
        return x * self.w[name + ".scale"][:, None] + self.w[name + ".shift"][:, None]

    def epoch_token(self, epoch):
        """
        Epoch encoder: (3000,) epoch -> (64,) token.
        """
        w = self.w
        x = np.asarray(epoch, dtype=np.float64)[None]
        x = gelu(self.batch_norm("stem", conv1d(x, w["stem.w"], stride=6)))  # (32, 500)
        y = channel_shuffle(gelu(self.batch_norm("res1", conv1d(x, w["res1.w"], groups=4))), 4)
        x = gelu(self.batch_norm("res2", conv1d(y, w["res2.w"], groups=4)) + x)
        x = channel_shuffle(gelu(self.batch_norm("down", conv1d(x, w["down.w"], stride=4, groups=8))), 8)  # (64, 125)
        # AvgPool1d(4, ceil_mode=True): the last window averages the remaining samples
        x = np.stack([x[:, i:i + 4].mean(axis=1) for i in range(0, x.shape[1], 4)], axis=1)  # (64, 32)
        # ECA: one weight per channel, from a convolution across the channel means, through a sigmoid
        eca = w["eca.w"][0, 0]
        gate = np.correlate(np.pad(x.mean(axis=1), len(eca) // 2), eca, mode="valid")
        x = x / (1 + np.exp(-gate))[:, None]
        return x.mean(axis=1)

    def temporal(self, tokens):
        """
        Temporal network: (64, n) tokens -> (64, n) features; causal.
        """
        w = self.w
        x = gelu(self.batch_norm("reduce", w["reduce.w"][:, :, 0] @ tokens))
        for name, dilation in self.blocks:
            kernel = w[name + ".dw.w"][:, 0]
            # padding on the left only: each output sees only the current and past tokens
            x_pad = np.pad(x, ((0, 0), (dilation * (kernel.shape[1] - 1), 0)))
            x = sum(kernel[:, [j]] * x_pad[:, j * dilation:j * dilation + x.shape[1]] for j in range(kernel.shape[1]))
            x = gelu(self.batch_norm(name + ".dw", x))
            x = gelu(self.batch_norm(name + ".pw", w[name + ".pw.w"][:, :, 0] @ x))
        return gelu(self.batch_norm("expand", w["expand.w"][:, :, 0] @ x) + tokens)

    def step(self, epoch):
        """
        Logits (W, N1, N2, N3, REM) of one epoch, from it and the up to 9 epochs before it.
        """
        self.tokens.append(self.epoch_token(epoch))
        features = self.temporal(np.stack(self.tokens, axis=1))[:, -1]
        return (self.w["classifier.w"] @ features + self.w["classifier.b"]).astype(np.float32)


def staging_channels(config_dict):
    """
    (Fpz, Cz) channels of a run dictionary, 1-based.
    """
    fpz = config_dict.get('staging_fpz_channel') or DEFAULT_FPZ_CHANNEL
    cz = config_dict.get('staging_cz_channel') or config_dict['channel_detection']
    return int(fpz), int(cz)


def staging_model_folder(config_dict, model_class):
    return Path(config_dict.get('staging_model_dir') or model_class.FOLDER)


def stages_file_path(config_dict):
    """
    Path of the per-epoch log, or None when the recording is not saved.
    """
    if config_dict.get('staging_log'):
        return Path(config_dict['staging_log'])
    if config_dict.get('record') and config_dict.get('filename'):
        recording = Path(config_dict['filename'])
        return recording.with_name(recording.stem + "_stages.tsv")
    return None


class SleepStagingDetector(Detector):
    """
    Sleep staging with TinySleepNet; NanoSleepNetStagingDetector uses NanoSleepNet-TCN.

    The stage of the last completed epoch goes to the detection signal (0 before the first epoch), each epoch to
    <recording>_stages.tsv, and whether the staging runs, or why not, to <recording>_staging.json.
    When the capture loop skips the detector (paused, Fpz or Cz electrode off), the staging restarts from scratch.
    If anything goes wrong with the staging, it stops and the recording goes on.

    Keys of the run dictionary (all optional):
        staging_fpz_channel: 1-based channel of Fpz (default: DEFAULT_FPZ_CHANNEL)
        staging_cz_channel: 1-based channel of Cz (default: channel_detection)
        staging_model_dir: folder of the model files (default: portiloop/models/tinysleepnet or nanosleepnet)
        staging_log: path of the per-epoch log (default: next to the recording)
    """
    model_class = TinySleepNetStep

    @classmethod
    def configuration_error(cls, config_dict):
        """
        Why the staging cannot run with this run dictionary, or None. Also checked by the Simple GUI at start.
        """
        fpz, cz = staging_channels(config_dict)
        if float(config_dict['frequency']) != FS_IN:
            return f"the sampling frequency must be {FS_IN} Hz, not {config_dict['frequency']}"
        for name, channel in (("Fpz", fpz), ("Cz", cz)):
            if not 1 <= channel <= int(config_dict['nb_channels']):
                return f"{name} channel {channel} does not exist ({config_dict['nb_channels']} channels)"
        if fpz == cz:
            return f"Fpz and Cz are the same channel ({fpz})"
        channel_states = config_dict.get('channel_states') or []
        for name, channel in (("Fpz", fpz), ("Cz", cz)):
            if channel <= len(channel_states) and channel_states[channel - 1] == 'disabled':
                return f"the {name} channel ({channel}) is disabled (electrode off at the impedance check?)"
        model_folder = staging_model_folder(config_dict, cls.model_class)
        for name in cls.model_class.FILES:
            if not (model_folder / name).is_file():
                return f"missing model file {model_folder / name}"
        with open(model_folder / cls.model_class.FILES[0], 'rb') as f:
            if f.read(len(LFS_POINTER)) == LFS_POINTER:
                return f"{model_folder / cls.model_class.FILES[0]} is a Git LFS pointer, not the model (git lfs pull)"
        return None

    def __init__(self, config_dict, lsl_streamer, csv_recorder):
        super().__init__(config_dict, lsl_streamer, csv_recorder)
        self.record_csv = not isinstance(self.csv_recorder, Dummy) and csv_recorder is not None
        self.fpz, self.cz = staging_channels(config_dict)
        # the capture loop pauses the detection when the electrode of one of these channels is off
        self.channels = [self.fpz, self.cz]
        self.model_folder = staging_model_folder(config_dict, self.model_class)

        self.stage = NO_STAGE
        self.segment = 0  # staging restarts so far
        self.nb_epochs = 0  # epochs staged in the current segment
        self.nb_rows = 0  # points seen so far (passed to detect or skip): the recording row of the next point
        self.segment_start = 0  # recording row of the first point of the current segment
        self.pending = []
        self.skipped = 0
        self.stages_file = None

        self.error = self.configuration_error(config_dict)
        if self.error is None:
            try:
                self.model = self.model_class(self.model_folder)
            except Exception as e:  # whatever is wrong with the model, the recording must go on
                logger.exception("Cannot load the sleep staging model from %s", self.model_folder)
                self.error = f"cannot load the model from {self.model_folder}: {e!r}"
        if self.error is not None:
            logger.error("Sleep staging disabled: %s", self.error)
            self.write_status()
            return

        self.preprocessor = Preprocessor(fpz=self.fpz - 1, cz=self.cz - 1)
        path = stages_file_path(config_dict)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.stages_file = open(path, 'w')
            header = ("segment", "epoch", "first_row", "stage") + tuple("logit_" + s for s in STAGES)
            self.stages_file.write("\t".join(header) + "\n")
            self.stages_file.flush()
        self.write_status()
        logger.info("Sleep staging (%s) on channels Fpz=%d, Cz=%d, log: %s",
                    self.model_class.NAME, self.fpz, self.cz, path)

    def write_status(self):
        """
        <recording>_staging.json (if the recording is saved): whether the staging runs, or why not.
        """
        status = {
            "enabled": self.error is None,
            "error": self.error,
            "model": self.model_class.NAME,
            "model_dir": str(self.model_folder),
            "fpz_channel": self.fpz,
            "cz_channel": self.cz,
        }
        if self.error is None:
            with open(self.model_folder / self.model_class.FILES[0], 'rb') as f:
                status["model_sha256"] = hashlib.sha256(f.read()).hexdigest()
        dump_metadata(self.config_dict, status, "staging")

    def skip(self, n_points):
        """
        Called by the capture loop for the points it does not pass to the detector (paused, electrode off).
        """
        self.skipped += n_points
        self.nb_rows += n_points

    def restart(self):
        logger.info("Sleep staging restarts after %d skipped points", self.skipped)
        self.preprocessor = Preprocessor(fpz=self.fpz - 1, cz=self.cz - 1)
        self.model.reset()
        self.pending = []
        self.stage = NO_STAGE
        self.segment += 1
        self.nb_epochs = 0
        self.segment_start = self.nb_rows
        self.skipped = 0

    def stage_epochs(self, points):
        for epoch in self.preprocessor.add_datapoints(points):
            logits = self.model.step(epoch)
            self.stage = int(np.argmax(logits)) + 1
            if self.stages_file is not None:
                first_row = self.segment_start + self.nb_epochs * EPOCH_ROWS
                row = [self.segment, self.nb_epochs, first_row, STAGES[self.stage - 1]]
                row += [f"{v:.6f}" for v in logits]
                self.stages_file.write("\t".join(map(str, row)) + "\n")
                self.stages_file.flush()
            self.nb_epochs += 1

    def detect(self, datapoints):
        """
        Writes the current stage to the detection signal; returns None (no stimulation).
        """
        if self.error is None:
            if self.skipped > 0:
                self.restart()
            self.pending.extend(datapoints)
            if len(self.pending) >= CHUNK:
                try:
                    self.stage_epochs(self.pending)
                except Exception as e:  # the recording must go on: only the staging stops
                    logger.exception("Sleep staging failed; it stops for the rest of the recording")
                    self.error = f"staging failed at recording row {self.nb_rows}: {e!r}"
                    self.stage = NO_STAGE
                    self.write_status()
                self.pending = []
        self.nb_rows += len(datapoints)
        if self.record_csv:
            self.csv_recorder.append_detection_signal_buffer([self.stage] * len(datapoints))
        return None


class NanoSleepNetStagingDetector(SleepStagingDetector):
    """
    Sleep staging with NanoSleepNet-TCN.
    """
    model_class = NanoSleepNetStep


# The Simple GUI shows the Fpz channel and checks the configuration for pipelines with these detectors
STAGING_DETECTORS = (SleepStagingDetector, NanoSleepNetStagingDetector)
