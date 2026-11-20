"""
Custom pipelines available in the GUI.
"""

from portiloop.src.custom.custom_processors import SpindleFilter, SlowOscillationFilter
from portiloop.src.custom.custom_detectors import SleepSpindleRealTimeDetector, SlowOscillationDetector, SOUpstateDetector
from portiloop.src.custom.custom_stimulators import SleepSpindleRealTimeStimulator, SlowOscillationStimulator
from portiloop.src.core.fast_capture import fast_capture_status
from portiloop.src.core.ffr import ffr_status
from portiloop.src.core.click_abr import click_abr_status

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

FFR_PIPELINE_KEY = "FFR (fast)"

# Fast acquisition with the sounds of the FFR protocol and their triggers (core/ffr.py)
_ffr_status = ffr_status()
if _fast_status is None and _ffr_status is None:
    PIPELINES[FFR_PIPELINE_KEY] = {
        "processor": None,
        "detector": None,
        "stimulator": None,
        "config_modifiers": {},
        "fast_capture": True,
        "ffr": True,
    }

CLICK_ABR_PIPELINE_KEY = "Click ABR (fast)"

# Fast acquisition with clicks in alternating polarity and their triggers (core/click_abr.py)
_click_abr_status = click_abr_status()
if _fast_status is None and _click_abr_status is None:
    PIPELINES[CLICK_ABR_PIPELINE_KEY] = {
        "processor": None,
        "detector": None,
        "stimulator": None,
        "config_modifiers": {},
        "fast_capture": True,
        "click_abr": True,
    }

DEFAULT_PIPELINE_KEY = "Acquisition only"
