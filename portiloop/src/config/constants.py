from pathlib import Path

from portiloop.src.hardware.frontend import Frontend
from portiloop.src.utils import get_portiloop_version


HOME_PATH = Path.home()
CSV_PATH = HOME_PATH / 'workspace' / 'edf_recordings'
RECORDING_PATH = HOME_PATH / 'portiloop-software' / 'portiloop' / 'recordings'
CALIBRATION_PATH = HOME_PATH / 'portiloop-software' / 'portiloop' / 'calibration'

try:
    version = get_portiloop_version()
    frontend = Frontend(version)
    nb_channels = frontend.get_version()
finally:
    frontend.close()
    del frontend

# version = 2
# nb_channels = 6

RUN_SETTINGS = {
    "version": version,
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
    "channel_states": [
        "simple",
        "simple",
        "simple",
        "simple",
        "disabled",
        "disabled"],
    "channel_detection": 2,
    "detection_sound": "15msPN_48kHz_norm_stereo.wav",
    "spindle_detection_mode": "Fast",
    "spindle_freq": 10,
    "stim_delay": 0.0,
    "inter_stim_delay": 0.0,
    "so_phase_delay": True,
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
    "input_filename": "test_recording.csv",
    "width_display": 1250,
    "filename": "/home/mendel/workspace/edf_recordings/recording_test1.csv",
    "vref": 2.64  # FIXME: this value is a temporary fix for what seems to be a hardware bug
}
