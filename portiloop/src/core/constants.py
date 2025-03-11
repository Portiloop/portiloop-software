from pathlib import Path

from portiloop.src.core.hardware.backend import Backend
from portiloop.src.core.utils import get_portiloop_version

try:
    VERSION = get_portiloop_version()
    backend = Backend(VERSION)
    nb_channels = backend.get_version()
finally:
    backend.close()
    del backend


# Home:
HOME_FOLDER = Path.home()

# Portiloop repository:
RECORDING_FOLDER = HOME_FOLDER / 'portiloop-software' / 'portiloop' / 'recordings'
SOUNDS_FOLDER = HOME_FOLDER / 'portiloop-software' / 'portiloop' / 'sounds'
DEFAULT_MODEL_PATH = HOME_FOLDER / 'portiloop-software' / 'portiloop' / 'models' / 'portiloop_model_quant.tflite'

# Workspace:
CSV_PATH = HOME_FOLDER / 'workspace' / 'edf_recordings'
WORKSPACE_DIR_SD = Path("/media/sd_card/workspace") / 'edf_recordings'
WORKSPACE_DIR_IN = HOME_FOLDER / 'workspace' / 'edf_recordings'
# TODO: remove all mentions of EDF format

# This dictionary contains the default options that are relevant to core functions
# It can be copied and extended by custom modules
DEFAULT_CONFIG_DICT = {
    "version": VERSION,
    "nb_channels": nb_channels,
    "frequency": 250,
    "duration": 36000,
    "filter": True,
    "record": True,
    "detect": True,
    "stimulate": False,
    "lsl": False,
    "display": False,
    "threshold": 0.75,
    "signal_input": "ADS",
    "python_clock": True,
    "signal_labels": [f"ch{i+1}" for i in range(nb_channels)],
    "channel_states": ["simple" for _ in range(nb_channels)],
    "channel_detection": 2,
    "detection_sound": "15msPN_48kHz_norm_stereo.wav",
    "spindle_detection_mode": "Fast",
    "spindle_freq": 10,
    "stim_delay": 0.0,
    "inter_stim_delay": 0.0,
    "volume": 100,
    "filter_settings": {
        "power_line": 60,
        "custom_fir": False,
        "custom_fir_order": 20,
        "custom_fir_cutoff": 30,
        "polyak_mean": 0.1,
        "polyak_std": 0.001,
        "epsilon": 1e-06,
        "filter_args": [
            True,
            True,
            True
        ]
    },
    "width_display": 1250,
    "filename": "/home/mendel/workspace/edf_recordings/recording_test1.csv",
    "vref": 2.64  # FIXME: this value is a temporary fix for what seems to be a hardware bug
}
