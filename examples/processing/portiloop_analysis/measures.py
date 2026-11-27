"""
Numbers extracted from event-locked responses, and statistics comparing two conditions.

Every function works on epochs (an array: epochs x time, or one averaged time course) and on the matching time
axis, in seconds relative to the stimulation marker, so windows are given in seconds, never in sample indices.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import stats


def _window(time: np.ndarray, window: Sequence[float]) -> np.ndarray:
    start, stop = window
    mask = (time >= start) & (time <= stop)
    if not mask.any():
        raise ValueError(f"The window {window} s contains no sample (epochs go from {time[0]} to {time[-1]} s)")
    return mask


def value_at(time: np.ndarray, epochs: np.ndarray, t: float = 0.0) -> np.ndarray:
    """Value of each epoch (or of one time course) at the time point closest to t, e.g. at the marker (t = 0)."""
    return np.asarray(epochs)[..., int(np.argmin(np.abs(time - t)))]


def window_mean(time: np.ndarray, epochs: np.ndarray, window: Sequence[float],
                baseline: Optional[Sequence[float]] = None) -> np.ndarray:
    """Mean of each epoch in a window, minus its mean in an optional baseline window."""
    epochs = np.asarray(epochs)
    value = epochs[..., _window(time, window)].mean(axis=-1)
    if baseline is not None:
        value = value - epochs[..., _window(time, baseline)].mean(axis=-1)
    return value


def window_rms(time: np.ndarray, epochs: np.ndarray, window: Sequence[float]) -> np.ndarray:
    """Root mean square of each epoch in a window (an amplitude measure that ignores the sign)."""
    epochs = np.asarray(epochs)
    return np.sqrt(np.mean(np.square(epochs[..., _window(time, window)]), axis=-1))


def evoked_slow_wave_amplitude(time: np.ndarray, epochs: np.ndarray, n550: Tuple[float, float] = (0.5, 0.6),
                               p900: Tuple[float, float] = (0.8, 1.0)) -> np.ndarray:
    """Peak-to-peak amplitude of the slow wave evoked by a sound during sleep: mean of the P900 window minus mean
    of the N550 window (the negative and positive components of the auditory K-complex)."""
    return window_mean(time, epochs, p900) - window_mean(time, epochs, n550)


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg false discovery rate correction (same result as statsmodels' multipletests fdr_bh).

    NaN p-values are ignored and stay NaN.
    """
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    n = int(valid.sum())
    if n == 0:
        return out
    order = np.argsort(p[valid])
    ranked = p[valid][order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]  # make the adjusted p-values monotonic
    corrected = np.empty(n)
    corrected[order] = np.minimum(adjusted, 1.0)
    out[valid] = corrected
    return out


def compare_time_courses(epochs_a: np.ndarray, epochs_b: np.ndarray, time: np.ndarray,
                         window: Optional[Sequence[float]] = None, alternative: str = "two-sided",
                         paired: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Compare two conditions time point by time point with t-tests, corrected with the FDR.

    Args:
        epochs_a, epochs_b: arrays (observations x time), for example one averaged response per participant in
            the stimulation (a) and sham (b) conditions.
        time: time axis in seconds.
        window: (start, stop) in seconds: only test (and correct across) the time points in this window.
            Testing fewer time points makes the correction less severe; choose the window before looking at the data.
        alternative: "two-sided", "greater" (a > b) or "less" (a < b).
        paired: True when the rows of a and b are the same participants in the same order (within-subject
            design, e.g. a stimulation night and a sham night per participant): paired t-tests. False for two
            different groups of participants: independent t-tests.

    Returns:
        (p, p_fdr): uncorrected and FDR-corrected p-values, NaN outside the window.
    """
    a, b = np.asarray(epochs_a, dtype=float), np.asarray(epochs_b, dtype=float)
    tested = np.ones(len(time), dtype=bool) if window is None else _window(time, window)
    p = np.full(len(time), np.nan)
    if paired:
        if a.shape != b.shape:
            raise ValueError("Paired comparison: both arrays must have the same participants in the same order")
        p[tested] = stats.ttest_rel(a[:, tested], b[:, tested], axis=0, alternative=alternative).pvalue
    else:
        p[tested] = stats.ttest_ind(a[:, tested], b[:, tested], axis=0, alternative=alternative).pvalue
    return p, fdr_bh(p)
