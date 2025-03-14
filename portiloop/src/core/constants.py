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
SIGNAL_SAMPLES_FOLDER = HOME_FOLDER / 'portiloop-software' / 'portiloop' / 'signal_samples'
SOUNDS_FOLDER = HOME_FOLDER / 'portiloop-software' / 'portiloop' / 'sounds'
DEFAULT_MODEL_PATH = HOME_FOLDER / 'portiloop-software' / 'portiloop' / 'models' / 'portiloop_model_quant.tflite'

if not (SIGNAL_SAMPLES_FOLDER.exists() and SOUNDS_FOLDER.exists() and DEFAULT_MODEL_PATH.exists()):
    raise RuntimeError("The current version of the library must be installed locally in the home folder.")

# SD card:
SD_CARD = Path("/media/sd_card/")

try:
    SD_CARD_TEST_FILE = SD_CARD / "test.tmp"
    SD_CARD_TEST_FILE.write_text("test")
    SD_CARD_TEST_FILE.unlink()
    SD_CARD_DETECTED = True
except Exception as e:
    print(f"DEV: caught exception {e}")
    SD_CARD_DETECTED = False

# Workspace:
if SD_CARD_DETECTED:
    WORKSPACE = SD_CARD / 'workspace'
else:
    WORKSPACE = HOME_FOLDER / 'workspace'

# Recording path:
CSV_PATH = WORKSPACE / 'recordings'

# Create folders if they don't exist:
CSV_PATH.mkdir(parents=True, exist_ok=True)


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
    "display_raw": True,
    "threshold": 0.75,
    "signal_input": "ADS",
    "python_clock": True,
    "signal_labels": [f"ch{i+1}" for i in range(nb_channels)],
    "channel_states": ["disabled" for _ in range(nb_channels)],
    "channel_detection": 2,
    "detection_sound": "15msPN_48kHz_norm_stereo.wav",
    "spindle_detection_mode": "Fast",
    "spindle_freq": 10,
    "stim_delay": 0.0,
    "inter_stim_delay": 0.0,
    "so_phase_delay": None,  # None or target phase in radians
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
    "signal_sample": str(SIGNAL_SAMPLES_FOLDER / "test_spindles.csv"),
    "vref": 2.64  # FIXME: this value is a temporary fix for what seems to be a hardware bug
}
