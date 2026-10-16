import os
from pathlib import Path

from portiloop import __version__
from portiloop.src.core.hardware.backend import Backend
from portiloop.src.core.utils import get_hardware_version

backend = None
try:
    HARDWARE_VERSION = get_hardware_version()
    backend = Backend(HARDWARE_VERSION)
    NB_CHANNELS = backend.get_nb_channels()
finally:
    if backend is not None:
        backend.close()
    del backend


# Home:
HOME_FOLDER = Path.home()

# Portiloop repository:
SIGNAL_SAMPLES_FOLDER = HOME_FOLDER / 'portiloop-software' / 'portiloop' / 'signal_samples'
SOUNDS_FOLDER = HOME_FOLDER / 'portiloop-software' / 'portiloop' / 'sounds'
DEFAULT_MODEL_PATH = HOME_FOLDER / 'portiloop-software' / 'portiloop' / 'models' / 'portiloop_model_quant.tflite'

if not (SIGNAL_SAMPLES_FOLDER.exists() and SOUNDS_FOLDER.exists() and DEFAULT_MODEL_PATH.exists()):
    raise RuntimeError("The current version of the library must be installed locally in the home folder.")

# SD card:
SD_CARD = Path("/media/sd_card/")


def sd_card_detected():
    """
    Whether the SD card is mounted and writable. The card can be inserted at any time (it is mounted by udev), so this
    is checked again before each recording.
    """
    return os.path.ismount(SD_CARD) and os.access(SD_CARD, os.W_OK)


def csv_path():
    """
    Returns:
        Path: the recording folder (created if needed): on the SD card when it is detected, in the home folder otherwise
    """
    path = (SD_CARD if sd_card_detected() else HOME_FOLDER) / 'workspace' / 'recordings'
    path.mkdir(parents=True, exist_ok=True)
    return path


# At import time (use the functions above to follow the insertion of the SD card):
SD_CARD_DETECTED = sd_card_detected()
CSV_PATH = csv_path()
WORKSPACE = CSV_PATH.parent

# State path:
STATE_PATH = HOME_FOLDER / 'workspace' / 'app_state'
STATE_PATH.mkdir(parents=True, exist_ok=True)


# This dictionary contains the default options that are relevant to core functions
# It can be copied and extended by custom modules
DEFAULT_CONFIG_DICT = {
    "software_version": __version__,
    "hardware_version": HARDWARE_VERSION,
    "nb_channels": NB_CHANNELS,
    "frequency": 250,
    "duration": 36000,
    "filter": True,
    "record": True,
    "detect": True,
    "stimulate": False,
    "lsl": False,
    "display": False,
    "display_raw": True,
    "threshold": 0.75,
    "signal_input": "ADS",
    "python_clock": True,
    "signal_labels": [f"ch{i+1}" for i in range(NB_CHANNELS)],
    "channel_states": ["disabled" for _ in range(NB_CHANNELS)],
    "channel_detection": 6,
    "detection_sound": "15msPN_48kHz_norm_stereo.wav",
    "stim_delay_mode": "Time",  # possible values: 'Time', 'Peak', 'Valley'
    "min_delay": 0.0,
    "max_delay": 0.0,
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
    "filename": str(CSV_PATH / "recording_test1.csv"),
    "record_raw": True,
    "record_filtered": False,
    "signal_sample": str(SIGNAL_SAMPLES_FOLDER / "test_spindles.csv"),
    "offline_speed": 1.0,
    "vref": 4.5,  # Valid only after Portiloop 2.3
    "leadoff_detection": True,  # remove fallen electrodes from the bias loop during recordings (core/leadoff.py)
}
