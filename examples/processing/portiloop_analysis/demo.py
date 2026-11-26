"""
Synthetic Portiloop recordings, so that the examples run without real data.

The files have the layout of real recordings (CSV with raw_ch1 ... raw_ch8, detection, stimulation and
stimulation_on, plus _metadata.json and _leadoff.json), and a sleep scoring file. The signal is NOT real EEG:
a 1/f background with slow oscillations and spontaneous spindles, a DC offset as in raw ADS1299 data, mains
interference, and, after every stimulation, an evoked slow wave (negative near 0.55 s, positive near 0.9 s) followed
by a spindle. A few artifacts (electrode pops, a movement, a period with the reference off) let the rejection
rules do some work.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .recording import MONTAGES


def _background(rng: np.random.Generator, n: int, fs: float, amplitude: float) -> np.ndarray:
    """About 1/f noise (sum of first-order autoregressive processes) with slow oscillations."""
    x = np.zeros(n)
    for pole, gain in zip([0.999, 0.99, 0.95, 0.8], [0.035, 0.11, 0.3, 0.45]):
        x += lfilter([gain], [1, -pole], rng.standard_normal(n))
    x = amplitude * (0.55 * x + 0.12 * rng.standard_normal(n))
    seconds = n // int(fs) + 1
    frequency = np.clip(0.8 + np.cumsum(0.1 * rng.standard_normal(seconds)), 0.5, 1.5)
    phase = 2 * np.pi * np.cumsum(np.repeat(frequency, int(fs))[:n]) / fs
    return x + 0.6 * amplitude * np.sin(phase)


def _spindle(t: np.ndarray, onset: float, duration: float, frequency: float) -> np.ndarray:
    """Waxing and waning oscillation of unit amplitude."""
    u = (t - onset) / duration
    inside = (u >= 0) & (u <= 1)
    return np.where(inside, np.sin(np.pi * np.clip(u, 0, 1)) ** 2 * np.sin(2 * np.pi * frequency * (t - onset)), 0.0)


def make_demo_recording(folder: Union[str, Path], name: str = "demo", start: Optional[datetime] = None,
                        duration_min: float = 30.0, fs: int = 250, seed: int = 0, slow_wave_uv: float = 60.0,
                        spindle_uv: float = 12.0, background_uv: float = 18.0, stim_interval_s=(6.0, 10.0),
                        montage: str = "default", power_line: int = 60, artifacts: bool = True) -> Path:
    """Write a synthetic recording and return the path of its .csv file.

    Args:
        folder: where to write the files (created if needed).
        name: participant or session name; the date and time are appended as in real recordings.
        start: start date and time (default: 2026-01-01 23:00:00).
        duration_min: duration in minutes.
        fs: sampling frequency in Hz.
        seed: random seed; the same seed gives the same recording.
        slow_wave_uv: amplitude of the evoked negative wave (the positive wave is half of it).
        spindle_uv: amplitude of the evoked spindle.
        background_uv: amplitude of the background EEG.
        stim_interval_s: (min, max) interval between stimulations, in seconds.
        montage: montage name in MONTAGES: the EEG is written on the Cz column.
        power_line: mains frequency of the interference (50 or 60 Hz).
        artifacts: add electrode pops, a movement and a period with the reference off.
    """
    rng = np.random.default_rng(seed)
    start = start or datetime(2026, 1, 1, 23, 0, 0)
    stem = f"{name}_{start:%Y-%m-%d_%H-%M-%S}_CONDB"  # CONDB: the stimulation condition
    folder = Path(folder) / stem
    folder.mkdir(parents=True, exist_ok=True)
    n = int(duration_min * 60 * fs)
    t = np.arange(n) / fs

    # Stimulations at random intervals, from 20 s after the start to 20 s before the end.
    times = []
    current = 20.0
    while current < n / fs - 20:
        times.append(current)
        current += rng.uniform(*stim_interval_s)
    stim_samples = np.round(np.array(times) * fs).astype(int)

    cz = _background(rng, n, fs, background_uv)
    for onset in rng.uniform(0, n / fs - 2, int(n / fs / 60 * 3)):  # spontaneous spindles, about 3 per minute
        cz += rng.uniform(0.5, 1.0) * spindle_uv * _spindle(t, onset, rng.uniform(0.6, 1.4), rng.uniform(12, 14.5))
    for sample in stim_samples:  # evoked responses, with some variability from one stimulation to the next
        s = sample / fs
        gain = rng.uniform(0.6, 1.4)
        window = slice(sample, min(n, sample + 3 * fs))
        tt = t[window] - s
        cz[window] += gain * slow_wave_uv * (-np.exp(-((tt - 0.55) / 0.12) ** 2) + 0.5 * np.exp(-((tt - 0.9) / 0.2) ** 2))
        cz[window] += gain * spindle_uv * _spindle(t[window], s + rng.uniform(0.9, 1.2), rng.uniform(0.8, 1.2),
                                                    rng.uniform(12, 14))

    # Artifacts: a few electrode pops (jumps), a movement, and a period with the reference off.
    leadoff_events = []
    if artifacts:
        for sample in rng.integers(0, n, 3):
            cz[sample:] += rng.choice([-1, 1]) * rng.uniform(1500, 3000)
        movement = int(rng.uniform(0.3, 0.6) * n)
        cz[movement:movement + 4 * fs] += 400 * rng.standard_normal(4 * fs)
        leadoff_start = int(rng.uniform(0.65, 0.8) * n)
        leadoff_end = leadoff_start + 60 * fs
        cz[leadoff_start:leadoff_end] += 3000 * np.sin(2 * np.pi * power_line * t[leadoff_start:leadoff_end])
        leadoff_events = [("reference_or_bias_off", leadoff_start), ("reference_or_bias_recovered", leadoff_end)]

    columns = {}
    for i, electrode in enumerate(MONTAGES[montage], start=1):
        signal = cz if electrode == "Cz" else _background(rng, n, fs, background_uv)
        offset = rng.uniform(-40000, 40000)  # raw ADS1299 data carry a large DC offset
        columns[f"raw_ch{i}"] = offset + signal + 3 * np.sin(2 * np.pi * power_line * t + rng.uniform(0, 6.3))
    stimulation = np.zeros(n, dtype=int)
    stimulation[stim_samples] = 1
    table = pd.DataFrame(columns)
    table["detection"] = stimulation
    table["stimulation"] = stimulation
    table["stimulation_on"] = 1
    csv = folder / f"{stem}.csv"
    table.to_csv(csv, index=False, float_format="%.3f")

    metadata = {"software_version": "demo", "nb_channels": 8, "frequency": fs, "duration": duration_min * 60,
                "channel_detection": MONTAGES[montage].index("Cz") + 1, "filter_settings": {"power_line": power_line},
                "note": "synthetic recording made by portiloop_analysis.demo, not real EEG"}
    (folder / f"{stem}_metadata.json").write_text(json.dumps(metadata, indent=4))
    clock = lambda sample: (start + timedelta(seconds=sample / fs)).isoformat(timespec="milliseconds")
    leadoff = {"events": [{"event": event, "channels": list(range(1, 9)), "sample": sample, "time": clock(sample)}
                          for event, sample in leadoff_events]}
    (folder / f"{stem}_leadoff.json").write_text(json.dumps(leadoff, indent=4))

    # Sleep scoring in 30 s periods: wake and N1 at the start, then mostly N2 and N3, with a short awakening.
    n_periods = int(np.ceil(n / fs / 30))
    stages = rng.choice(["N2", "N3", "N2", "N2", "N1"], n_periods).astype(object)
    stages[:4] = ["W", "W", "N1", "N1"]
    awake = int(0.5 * n_periods)
    stages[awake:awake + 2] = "W"
    pd.DataFrame({"Annotation": stages, "Duration": 30.0}).to_csv(folder / f"{stem}_hypnogram.csv", index=False)
    return csv
