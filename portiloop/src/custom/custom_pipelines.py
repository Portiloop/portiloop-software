"""
Custom pipelines available in the GUI.
"""

from portiloop.src.custom.custom_processors import SpindleFilter, SlowOscillationFilter
from portiloop.src.custom.custom_detectors import SleepSpindleRealTimeDetector, SlowOscillationDetector
from portiloop.src.custom.custom_stimulators import SleepSpindleRealTimeStimulator, SlowOscillationStimulator, RandomStimulator

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
    "Random (sleep spindles)": {
        "processor": SpindleFilter,
        "detector": SleepSpindleRealTimeDetector,
        "stimulator": RandomStimulator,
        "config_modifiers": {}
    },
    "Random (slow oscillations)": {
        "processor": SlowOscillationFilter,
        "detector": SlowOscillationDetector,
        "stimulator": RandomStimulator,
        "config_modifiers": {}
    }
}

DEFAULT_PIPELINE_KEY = "Acquisition only"
