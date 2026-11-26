"""
portiloop_analysis: offline analysis of Portiloop recordings.

Typical use:

    from portiloop_analysis import load_recording, EpochSettings, event_locked_responses, plot_responses

    recording = load_recording("night1.csv", channel="Cz", montage="default")
    responses = event_locked_responses(recording, EpochSettings(notch_hz=60))
    plot_responses(responses)

Modules:
    recording       loading recordings (current and older formats, LSL .xdf files, sleep scoring)
    preprocessing   filtering and artifact detection
    epochs          epochs around stimulations, rejection, averages
    measures        values in time windows, statistics between conditions
    plotting        figures
    demo            synthetic recordings for trying the examples
"""
from .demo import make_demo_recording
from .epochs import EpochSettings, EventLockedResponses, deduplicate, event_locked_responses, filter_recording
from .measures import (compare_time_courses, evoked_slow_wave_amplitude, fdr_bh, value_at, window_mean,
                       window_rms)
from .plotting import plot_correlation, plot_groups, plot_responses
from .preprocessing import BROADBAND, SIGMA_BAND, SLOW_BAND, bad_sections, detect_jumps, filter_band
from .recording import (EEG_CHANNELS, MONTAGES, Recording, load_hypnogram, load_legacy_recording, load_recording,
                        read_leadoff_periods, read_metadata, recording_start, xdf_to_dataframe)

__all__ = [
    "BROADBAND", "EEG_CHANNELS", "EpochSettings", "EventLockedResponses", "MONTAGES", "Recording", "SIGMA_BAND",
    "SLOW_BAND", "bad_sections", "compare_time_courses", "deduplicate", "detect_jumps", "event_locked_responses",
    "evoked_slow_wave_amplitude", "fdr_bh", "filter_band", "filter_recording", "load_hypnogram",
    "load_legacy_recording", "load_recording", "make_demo_recording", "plot_correlation", "plot_groups",
    "plot_responses", "read_leadoff_periods", "read_metadata", "recording_start", "value_at", "window_mean",
    "window_rms", "xdf_to_dataframe",
]
