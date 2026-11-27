"""
build_responses_data.py: data for the phase widget of tutorial 07 (docs/assets/data/07-responses-phase.json).

    python docs/tools/build_responses_data.py DATASET_FOLDER

DATASET_FOLDER is the unzipped tutorial dataset (the folder holding <name>.csv and <name>_metadata.json).
Needs numpy, scipy, pandas and examples/processing/portiloop_analysis.

For every stimulation marker of the night (same order as the events of epochs.json), the file gives the phase of
the slow oscillation at the marker: the Cz signal is filtered offline between 0.5 and 2 Hz (zero phase, with
portiloop_analysis.filter_band: notch from the metadata, each uninterrupted stretch filtered on its own, 10 s
discarded at both ends), then the Hilbert transform gives the instantaneous phase and amplitude.

Phase convention (cosine): 0 degrees = positive peak (up-state), +/-180 = negative peak (down-state),
-90 = rising zero crossing (from the down-state to the up-state), +90 = falling zero crossing.

Output (about 20 KB): {"band_hz", "convention", "phase_deg": [int or null], "amplitude_uv": [float or null]}.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "examples" / "processing"))
from portiloop_analysis import filter_band, load_recording  # noqa: E402
from portiloop_analysis.preprocessing import continuous_segments  # noqa: E402

BAND = (0.5, 2.0)
OUT = HERE.parents[1] / "assets" / "data" / "07-responses-phase.json"


def main(folder):
    folder = Path(folder)
    csv = next(f for f in folder.glob("*.csv") if not f.stem.endswith(("_nanosleepnet", "_simulated")))
    recording = load_recording(csv, channel="Cz", montage="default", verbose=False)
    notch = recording.metadata.get("filter_settings", {}).get("power_line", 60)
    slow = filter_band(recording.eeg, BAND, fs=recording.fs, notch_hz=notch, edge_s=10.0, breaks=recording.breaks)
    analytic = np.full(len(slow), np.nan, dtype=complex)
    for start, stop in zip(*continuous_segments(np.isfinite(slow))):
        analytic[start:stop] = hilbert(slow[start:stop])
    events = recording.events
    phase = np.degrees(np.angle(analytic[events]))
    amplitude = np.abs(analytic[events])
    ok = np.isfinite(phase)
    out = {
        "description": "Phase of the 0.5-2 Hz Cz signal (offline, zero-phase filter, Hilbert transform) at every "
                       "stimulation marker of the tutorial night, in the order of the events of epochs.json.",
        "band_hz": list(BAND), "notch_hz": notch,
        "convention": "degrees; 0 = positive peak (up-state), +/-180 = negative peak (down-state), -90 = rising zero "
                      "crossing, +90 = falling zero crossing",
        "phase_deg": [int(round(p)) if k else None for p, k in zip(phase, ok)],
        "amplitude_uv": [round(float(a), 1) if k else None for a, k in zip(amplitude, ok)],
    }
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    z = np.exp(1j * np.radians(phase[ok]))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB); {ok.sum()} of {len(events)} markers with a phase; "
          f"circular mean {np.degrees(np.angle(z.mean())):.1f} deg, resultant length {abs(z.mean()):.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
