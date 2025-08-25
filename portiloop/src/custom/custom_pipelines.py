"""
Custom pipelines available in the GUI.
"""

from portiloop.src.custom.custom_processors import SpindleFilter, SlowOscillationFilter
from portiloop.src.custom.custom_detectors import SleepSpindleRealTimeDetector, SlowOscillationDetector
from portiloop.src.custom.custom_stimulators import SleepSpindleRealTimeStimulator, SlowOscillationStimulator

PIPELINES = {
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
    }
}
