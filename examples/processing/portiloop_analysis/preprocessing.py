"""
Offline filtering and artifact detection.

All filters here are zero-phase (applied forwards and backwards with scipy.signal.sosfiltfilt): unlike the causal
filters of the Portiloop itself, they do not delay the signal, which is what an offline analysis of event-locked
responses needs.

Recordings contain missing data (gaps between files, electrodes off). A filter must never run across a gap, so
``filter_band`` filters each uninterrupted stretch of signal separately and discards its first and last seconds,
where filtering a finite stretch distorts the signal.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

# Frequency bands (Hz) used in the examples.
SLOW_BAND = (0.1, 4.0)      # slow oscillations and slow-wave activity
SIGMA_BAND = (11.0, 16.0)   # sleep spindles
BROADBAND = (0.1, 20.0)     # used to detect artifacts


def continuous_segments(valid: np.ndarray, breaks: Iterable[int] = ()) -> Tuple[np.ndarray, np.ndarray]:
    """First and end (excluded) samples of every run of valid samples, also split at the given break points."""
    valid = np.asarray(valid, dtype=bool).copy()
    for sample in breaks:
        valid[sample] = False  # a new file starts here: split even without a gap
    starts = np.flatnonzero(np.diff(valid.astype(int), prepend=0) == 1)
    stops = np.flatnonzero(np.diff(valid.astype(int), append=0) == -1) + 1
    return starts, stops


def filter_band(eeg: np.ndarray, band: Sequence[float], fs: float, notch_hz: Optional[float] = None,
                order: int = 4, edge_s: float = 10.0, breaks: Iterable[int] = ()) -> np.ndarray:
    """Band-pass filter a signal that may contain missing data (NaN).

    Every uninterrupted stretch of signal is filtered on its own: its mean is removed, an optional notch filter
    (first-order Butterworth band-stop, notch_hz +/- 0.5 Hz) removes mains interference, then a Butterworth
    band-pass filter keeps the band. The first and last ``edge_s`` seconds of every stretch are set to NaN, as are
    stretches shorter than twice ``edge_s``.

    Args:
        eeg: signal in microvolts, NaN where missing.
        band: (low, high) cut-off frequencies in Hz.
        fs: sampling frequency in Hz.
        notch_hz: mains frequency to remove (50 or 60 Hz), or None.
        order: order of the band-pass filter (applied twice, forwards and backwards).
        edge_s: seconds discarded at both ends of every stretch.
        breaks: samples where a new stretch must start even without missing data (file boundaries).

    Returns:
        the filtered signal, same length as eeg, NaN where missing or discarded.
    """
    eeg = np.asarray(eeg, dtype=float)
    low, high = band
    if not 0 < low < high < fs / 2:
        raise ValueError(f"The band {band} Hz must lie between 0 and fs/2 = {fs / 2} Hz")
    notch = None
    if notch_hz is not None:
        if not 0 < notch_hz - 0.5 < notch_hz + 0.5 < fs / 2:
            raise ValueError(f"The notch {notch_hz} +/- 0.5 Hz must lie between 0 and fs/2")
        notch = butter(1, [notch_hz - 0.5, notch_hz + 0.5], btype="bandstop", fs=fs, output="sos")
    bandpass = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    edge = round(edge_s * fs)
    filtered = np.full(len(eeg), np.nan)
    for start, stop in zip(*continuous_segments(np.isfinite(eeg), breaks)):
        if stop - start <= 2 * edge:
            continue
        segment = eeg[start:stop] - eeg[start:stop].mean()
        if notch is not None:
            segment = sosfiltfilt(notch, segment)
        segment = sosfiltfilt(bandpass, segment)
        filtered[start + edge:stop - edge] = segment[edge:len(segment) - edge]
    return filtered


def detect_jumps(eeg: np.ndarray, limit_uv: float, breaks: Iterable[int] = ()) -> np.ndarray:
    """Samples whose value differs from the previous sample by more than limit_uv (electrode pops, movements).

    Must be run on the raw, unfiltered signal. The first sample of each file is never flagged, since a jump there
    only reflects the change of file.
    """
    if not np.isfinite(limit_uv) or limit_uv <= 0:
        raise ValueError("The jump limit must be a positive number of microvolts")
    eeg = np.asarray(eeg, dtype=float)
    jumps = np.zeros(len(eeg), dtype=bool)
    both_valid = np.isfinite(eeg[:-1]) & np.isfinite(eeg[1:])
    jumps[1:] = both_valid & (np.abs(np.diff(eeg)) > limit_uv)
    for sample in breaks:
        jumps[sample] = False
    return jumps


def bad_sections(eeg: np.ndarray, fs: float, offhead_uv: float = 105600.0, jump_uv: float = 222.0,
                 min_good_s: float = 300.0) -> np.ndarray:
    """Mask of usable samples for a whole night, as in the analysis of the BigStim study.

    Three kinds of samples are excluded:

    1. runs of at least ``min_good_s`` seconds above ``offhead_uv`` (the device was off the head: the raw signal of
       a floating input drifts to large values);
    2. samples that differ from the previous one by more than ``jump_uv``;
    3. stretches of usable samples shorter than ``min_good_s`` between excluded samples (too short to analyse).

    The default thresholds are those of the BigStim analysis, in microvolts (the full scale of the ADS1299 is
    +/- 187,500 uV): 105,600 uV for the off-head level and 222 uV for jumps.

    Args:
        eeg: raw signal in microvolts.
        fs: sampling frequency in Hz.
        offhead_uv: raw value above which the device is considered off the head.
        jump_uv: largest accepted difference between two consecutive samples.
        min_good_s: minimum duration of an off-head run, and of a usable stretch, in seconds.

    Returns:
        boolean array, True for usable samples.
    """
    eeg = np.asarray(eeg, dtype=float)
    min_samples = round(min_good_s * fs)
    good = np.isfinite(eeg)
    # 1. long runs above the off-head threshold
    for start, stop in zip(*continuous_segments(np.nan_to_num(eeg, nan=-np.inf) > offhead_uv)):
        if stop - start >= min_samples:
            good[start:stop] = False
    # 2. jumps
    good[1:] &= ~(np.abs(np.diff(eeg)) > jump_uv)
    # 3. short usable stretches
    for start, stop in zip(*continuous_segments(good)):
        if stop - start < min_samples:
            good[start:stop] = False
    return good
