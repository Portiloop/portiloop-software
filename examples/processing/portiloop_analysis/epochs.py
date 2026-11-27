"""
Event-locked responses: the average EEG around stimulations.

For every stimulation marker, an epoch (a window of signal centred on the marker) is cut from three filtered
versions of the signal:

- slow: the slow band (0.1 to 4 Hz by default), where slow oscillations and evoked slow waves appear;
- sigma: the amplitude envelope of the sigma band (11 to 16 Hz), computed with the Hilbert transform, which
  rises when spindles appear;
- broadband: a wide band (0.1 to 20 Hz by default), used to reject artifacts and as the evoked response (ERP).

Epochs are rejected when they reach outside the recording, overlap a period with the electrode off, contain a
jump of the raw signal, contain missing or discarded samples, or exceed an amplitude limit in the broadband
signal. The same epochs are kept in every band, so the slow and sigma responses describe the same events.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import hilbert

from .preprocessing import BROADBAND, SIGMA_BAND, SLOW_BAND, detect_jumps, filter_band
from .recording import Recording

# A baseline is either a time window (start, stop) in seconds relative to the marker, "epoch" for the mean of the
# whole epoch, or None for no baseline correction.
Baseline = Union[Tuple[float, float], str, None]
BANDS = ("slow", "sigma", "broadband")


@dataclass(frozen=True)
class EpochSettings:
    """Parameters of the event-locked analysis. The defaults are those of the slow and sigma analysis example.

    Attributes:
        window_s: epochs go from -window_s to +window_s seconds around each marker.
        notch_hz: mains frequency removed before filtering (50 or 60 Hz), or None.
        slow_band, sigma_band, broadband: (low, high) frequencies in Hz.
        edge_s: seconds discarded at both ends of every uninterrupted stretch of signal after filtering.
        max_ptp_uv: reject epochs whose broadband peak-to-peak amplitude exceeds this value (None: no limit).
        max_abs_uv: reject epochs in which the broadband signal exceeds this absolute value (None: no limit).
        jump_limit_uv: reject epochs containing a jump of the raw signal larger than this (None: no check).
        slow_baseline, sigma_baseline, broadband_baseline: baseline subtracted from each epoch of each band.
        min_interval_s: markers closer than this to the previous kept marker are treated as duplicates (None: keep all).
        stages: keep only markers during these sleep stages, for example ("N2", "N3") (None: all markers).
    """
    window_s: float = 5.0
    notch_hz: Optional[float] = 60.0
    slow_band: Tuple[float, float] = SLOW_BAND
    sigma_band: Tuple[float, float] = SIGMA_BAND
    broadband: Tuple[float, float] = BROADBAND
    edge_s: float = 10.0
    max_ptp_uv: Optional[float] = 350.0
    max_abs_uv: Optional[float] = None
    jump_limit_uv: Optional[float] = 1000.0
    slow_baseline: Baseline = "epoch"
    sigma_baseline: Baseline = (-1.5, -1.0)
    broadband_baseline: Baseline = None
    min_interval_s: Optional[float] = None
    stages: Optional[Tuple[str, ...]] = None

    def with_changes(self, **changes) -> "EpochSettings":
        """A copy with some settings changed, e.g. settings.with_changes(window_s=15)."""
        return replace(self, **changes)


@dataclass
class EventLockedResponses:
    """Epochs around the stimulation markers of one recording.

    Attributes:
        time: time of each epoch sample relative to the marker, in seconds.
        events: one row per marker: sample, time_s and status ("kept" or the reason of rejection).
        epochs: for each band ("slow", "sigma", "broadband"), an array (kept epochs x time).
        settings: the settings used.
        channel: the electrode analysed.
    """
    time: np.ndarray
    events: pd.DataFrame
    epochs: Dict[str, np.ndarray]
    settings: EpochSettings
    channel: str = ""
    fs: float = 250.0

    @property
    def n_kept(self) -> int:
        return int((self.events["status"] == "kept").sum())

    @property
    def kept_samples(self) -> np.ndarray:
        return self.events.loc[self.events["status"] == "kept", "sample"].to_numpy()

    def mean(self, band: str) -> np.ndarray:
        """Average epoch of a band."""
        return self.epochs[band].mean(axis=0) if self.n_kept else np.full(len(self.time), np.nan)

    def sem(self, band: str) -> np.ndarray:
        """Standard error of the mean of a band (standard deviation across epochs / sqrt(number of epochs)).

        Epochs closer than their length share samples, so the SEM may underestimate the uncertainty.
        """
        if self.n_kept < 2:
            return np.full(len(self.time), np.nan)
        return self.epochs[band].std(axis=0, ddof=1) / np.sqrt(self.n_kept)

    def rejection_summary(self) -> pd.Series:
        """Number of markers per status."""
        return self.events["status"].value_counts().rename("markers")

    def averages(self) -> pd.DataFrame:
        """Mean and SEM of every band, one row per time point."""
        table = {"time_s": self.time}
        for band in self.epochs:
            table[f"{band}_mean_uV"] = self.mean(band)
            table[f"{band}_sem_uV"] = self.sem(band)
        return pd.DataFrame(table)


def filter_recording(recording: Recording, settings: EpochSettings) -> Dict[str, np.ndarray]:
    """The three filtered signals of a recording (slow, sigma and broadband), NaN where unusable."""
    common = dict(fs=recording.fs, notch_hz=settings.notch_hz, edge_s=settings.edge_s, breaks=recording.breaks)
    return {
        "slow": filter_band(recording.eeg, settings.slow_band, **common),
        "sigma": filter_band(recording.eeg, settings.sigma_band, **common),
        "broadband": filter_band(recording.eeg, settings.broadband, **common),
    }


def deduplicate(events: np.ndarray, min_interval: int) -> np.ndarray:
    """Keep the first marker of every burst: a marker is kept when it comes at least min_interval samples after
    the previous marker (the original analysis compared each marker with the previous one, kept or not)."""
    events = np.asarray(events)
    if len(events) == 0:
        return events
    return events[np.concatenate(([0], np.flatnonzero(np.diff(events) >= min_interval) + 1))]


def _baseline_mask(time: np.ndarray, baseline: Baseline, name: str) -> Optional[np.ndarray]:
    if baseline is None or isinstance(baseline, str):
        if baseline not in (None, "epoch"):
            raise ValueError(f"{name}: use a (start, stop) window in seconds, 'epoch' or None")
        return None
    start, stop = baseline
    if not (time[0] <= start < stop <= time[-1]):
        raise ValueError(f"{name} {baseline} must lie inside the epoch ({time[0]} to {time[-1]} s)")
    mask = (time >= start) & (time <= stop)
    if not mask.any():
        raise ValueError(f"{name} {baseline} contains no sample")
    return mask


def _apply_baseline(epoch: np.ndarray, baseline: Baseline, mask: Optional[np.ndarray]) -> np.ndarray:
    if baseline == "epoch":
        return epoch - epoch.mean()
    if mask is not None:
        return epoch - epoch[mask].mean()
    return epoch


def event_locked_responses(recording: Recording, settings: EpochSettings = EpochSettings(),
                           filtered: Optional[Dict[str, np.ndarray]] = None) -> EventLockedResponses:
    """Cut, check and baseline-correct the epochs around every stimulation marker of a recording.

    Args:
        recording: the recording (see load_recording).
        settings: analysis parameters.
        filtered: the output of filter_recording, to reuse it when analysing the same recording several times.

    Returns:
        EventLockedResponses with every marker and its status, and the kept epochs of each band.
    """
    fs = recording.fs
    half = round(settings.window_s * fs)
    time = np.arange(-half, half + 1) / fs
    masks = {band: _baseline_mask(time, getattr(settings, f"{band}_baseline"), f"{band}_baseline") for band in BANDS}
    if filtered is None:
        filtered = filter_recording(recording, settings)
    jumps = (detect_jumps(recording.eeg, settings.jump_limit_uv, recording.breaks)
             if settings.jump_limit_uv is not None else np.zeros(recording.n_samples, dtype=bool))

    candidates = recording.events
    status = {}
    if settings.stages is not None:
        if recording.stages is None:
            raise ValueError("settings.stages is set, but the recording has no sleep stages")
        outside = ~np.isin(recording.stages[candidates], list(settings.stages))
        status.update({int(sample): "stage" for sample in candidates[outside]})
        candidates = candidates[~outside]
    if settings.min_interval_s is not None:
        unique = deduplicate(candidates, round(settings.min_interval_s * fs))
        status.update({int(sample): "duplicate" for sample in np.setdiff1d(candidates, unique)})
        candidates = unique

    kept = {band: [] for band in BANDS}
    for event in candidates:
        start, stop = event - half, event + half + 1
        reason = None
        if start < 0 or stop > recording.n_samples:
            reason = "edge"  # the epoch reaches outside the recording
        elif recording.leadoff[start:stop].any():
            reason = "leadoff"  # electrode, reference or bias off
        elif jumps[start + 1:stop].any():
            reason = "jump"  # abrupt change of the raw signal
        else:
            pieces = {band: filtered[band][start:stop] for band in BANDS}
            if not all(np.isfinite(piece).all() for piece in pieces.values()):
                reason = "missing"  # missing data, or too close to a gap or to the start/end of a file
            elif settings.max_ptp_uv is not None and np.ptp(pieces["broadband"]) > settings.max_ptp_uv:
                reason = "artifact"
            elif settings.max_abs_uv is not None and np.abs(pieces["broadband"]).max() > settings.max_abs_uv:
                reason = "artifact"
        if reason is not None:
            status[int(event)] = reason
            continue
        status[int(event)] = "kept"
        pieces["sigma"] = np.abs(hilbert(pieces["sigma"]))  # amplitude envelope of the sigma band
        for band in BANDS:
            kept[band].append(_apply_baseline(pieces[band], getattr(settings, f"{band}_baseline"), masks[band]))

    samples = np.array(sorted(status), dtype=int)
    events = pd.DataFrame({"sample": samples, "time_s": samples / fs, "status": [status[s] for s in samples]})
    epochs = {band: np.array(kept[band]).reshape(-1, len(time)) for band in BANDS}
    return EventLockedResponses(time=time, events=events, epochs=epochs, settings=settings,
                                channel=recording.channel, fs=fs)
