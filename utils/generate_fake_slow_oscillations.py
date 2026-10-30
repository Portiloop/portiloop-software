"""
Generates a CSV signal sample compatible with FileBackend, containing one fake Slow Oscillation (SO) every 10 seconds.

The background is a low-amplitude 2 Hz sine wave.
Its regular falling zero-crossings reset the state of SlowOscillationDetector between SOs
(with a flat signal, the time spent "above" zero would accumulate and prevent detection).

Each SO starts at a falling zero-crossing of the background, and consists of:
- a negative half-wave (down-state) of SO_NEG_DURATION seconds and amplitude SO_NEG_AMPLITUDE,
- followed by a positive half-wave (up-state) of SO_POS_DURATION seconds and amplitude SO_POS_AMPLITUDE.
The background resumes at a falling zero-crossing right after the up-state, which ends the SO.

Values are in microvolts. FileBackend only reads the first column; the other columns are zeros, like in the other samples.

Usage:
    python generate_fake_slow_oscillations.py [output.csv]
"""

import sys
from pathlib import Path

import numpy as np

FREQUENCY = 250  # sampling frequency (Hz), must match the "frequency" setting of the Portiloop
DURATION = 300  # total duration of the signal (s)
NB_COLUMNS = 4  # FileBackend only reads the first column

SO_PERIOD = 10.0  # one SO every SO_PERIOD seconds
SO_NEG_DURATION = 0.5  # duration of the down-state (s), SlowOscillationDetector expects 125 ms to 1500 ms
SO_POS_DURATION = 0.5  # duration of the up-state (s), SlowOscillationDetector expects less than 1000 ms
SO_NEG_AMPLITUDE = 150.0  # down-state amplitude (uV), SlowOscillationDetector expects more than 40 uV after filtering
SO_POS_AMPLITUDE = 100.0  # up-state amplitude (uV), SlowOscillationDetector expects more than 75 uV peak-to-peak after filtering

BG_FREQUENCY = 2.0  # background sine frequency (Hz)
BG_AMPLITUDE = 5.0  # background sine amplitude (uV), small enough not to be detected as an SO

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "portiloop" / "signal_samples" / "fake_slow_oscillations.csv"


def generate_signal():
    t = np.arange(int(DURATION * FREQUENCY)) / FREQUENCY
    signal = BG_AMPLITUDE * np.sin(2 * np.pi * BG_FREQUENCY * t)

    so_duration = SO_NEG_DURATION + SO_POS_DURATION
    bg_period = 1.0 / BG_FREQUENCY
    # the SO duration must be a multiple of the background period, so that the background resumes in phase
    assert np.isclose(so_duration / bg_period, round(so_duration / bg_period)), "the SO duration must be a multiple of the background period"

    so_onset = SO_PERIOD / 2 + bg_period / 2  # falling zero-crossing of the background
    while so_onset + so_duration < DURATION:
        neg = (t >= so_onset) & (t < so_onset + SO_NEG_DURATION)
        pos = (t >= so_onset + SO_NEG_DURATION) & (t < so_onset + so_duration)
        signal[neg] = -SO_NEG_AMPLITUDE * np.sin(np.pi * (t[neg] - so_onset) / SO_NEG_DURATION)
        signal[pos] = SO_POS_AMPLITUDE * np.sin(np.pi * (t[pos] - so_onset - SO_NEG_DURATION) / SO_POS_DURATION)
        so_onset += SO_PERIOD

    return signal


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    data = np.zeros((int(DURATION * FREQUENCY), NB_COLUMNS))
    data[:, 0] = generate_signal()
    np.savetxt(output, data, delimiter=',', fmt=['%.6f'] + ['%d'] * (NB_COLUMNS - 1))
    print(f"Wrote {len(data)} samples ({DURATION} s at {FREQUENCY} Hz) to {output}")


if __name__ == '__main__':
    main()
