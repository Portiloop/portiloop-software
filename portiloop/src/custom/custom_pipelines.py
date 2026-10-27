"""
Custom pipelines available in the GUI.
"""

from portiloop.src.custom.custom_processors import SpindleFilter, SlowOscillationFilter
from portiloop.src.custom.custom_detectors import SleepSpindleRealTimeDetector, SlowOscillationDetector, SOUpstateDetector
from portiloop.src.custom.custom_stimulators import SleepSpindleRealTimeStimulator, SlowOscillationStimulator
from portiloop.src.core.fast_capture import fast_capture_status
from portiloop.src.custom.sleep_staging import SleepStagingDetector, NanoSleepNetStagingDetector

PIPELINES = {
    "Acquisition only": {
        "processor": None,
        "detector": None,
        "stimulator": None,
        "config_modifiers": {}
    },
    "Sleep spindles": {
        "processor": SpindleFilter,
        "detector": SleepSpindleRealTimeDetector,
        "stimulator": SleepSpindleRealTimeStimulator,
        "config_modifiers": {}
    },
    "Sleep slow oscillations": {
        "processor": SlowOscillationFilter,
        "detector": SlowOscillationDetector,
        "stimulator": SlowOscillationStimulator,
        "config_modifiers": {}
    },
    "Sleep slow oscillations (upstate)": {
        "processor": SlowOscillationFilter,
        "detector": SOUpstateDetector,
        "stimulator": SlowOscillationStimulator,
        "config_modifiers": {}
    },
    "Sleep staging (TinySleepNet)": {
        "processor": None,
        "detector": SleepStagingDetector,
        "stimulator": None,
        "config_modifiers": {}
    },
    "Sleep staging (NanoSleepNet)": {
        "processor": None,
        "detector": NanoSleepNetStagingDetector,
        "stimulator": None,
        "config_modifiers": {}
    }
}

FAST_PIPELINE_KEY = "Acquisition only (fast)"

# 2 to 16 kSPS, recorded by a C program (core/fast_capture.py), only where it is installed (fast_capture/install.sh)
_fast_status = fast_capture_status()
if _fast_status is None:
    PIPELINES[FAST_PIPELINE_KEY] = {
        "processor": None,
        "detector": None,
        "stimulator": None,
        "config_modifiers": {},
        "fast_capture": True,
    }

DEFAULT_PIPELINE_KEY = "Acquisition only"
