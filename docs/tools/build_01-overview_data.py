"""
build_01-overview_data.py: data for the closed-loop widget of tutorial 01 (docs/assets/data/01-overview-deep.json).

    python docs/tools/build_01-overview_data.py DATASET_FOLDER

DATASET_FOLDER is the unzipped tutorial dataset (the folder holding <name>.csv and <name>_metadata.json).
Needs numpy, scipy, pandas and examples/processing/portiloop_analysis. Run build_assets.py first: the window is the
"deep" excerpt of docs/assets/data/excerpts.json (30 s starting at minute 123.0 of the tutorial night).

What it computes, for the detection channel (Cz, raw_ch6):
  device   the signal the slow-oscillation detector receives on the device. It is the filter chain of
           SlowOscillationFilter (portiloop/src/custom/custom_processors.py, at 250 Hz): a 20-tap Hamming FIR
           band-pass "0.5 to 30 Hz", the notch of the metadata (precomputed coefficients), the exponential DC
           removal (alpha = 0.005) and a 17-tap FIR band-pass "0.16 to 4 Hz", all causal, run from the first
           sample of the night with zero initial states (the same as the device and the offline simulator).
  slow     the 0.1 to 4 Hz band used in the analysis (portiloop_analysis.filter_band, zero phase, mains notch),
           computed on the window with 60 s of margin on each side.
  detections  the detections of a copy of SOUpstateDetector.run_detection (custom_detectors.py), run on "device"
           wherever stimulation_on is 1, with, for each one, the values of the fixed criteria at that sample:
           peak-to-peak amplitude, negative amplitude, time below and above zero, and the up-state delay estimate.
           The copy follows the software that recorded the tutorial night (branch main at commit 917f1f3): the
           running estimate of the up-state delay is truncated to an integer at each update (int(...),
           counter == estimate). Main keeps it as a float since commit e4959f1 (2026-11-26, compared with
           round(...)); set TRUNCATE = False to reproduce the fixed version.
  stims    the stimulations recorded by the device (stimulation column), as sample offsets in the window.
The script prints how many detections of the copy fall on the same sample as a recorded stimulation (with
TRUNCATE = True: all of them up to the end of the window), and the statistics of the up-state estimate over the
whole night.

Output: docs/assets/data/01-overview-deep.json (about 50 KB). Signals are base64 little-endian int16 in 0.1 uV units.
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import firwin, lfilter

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "examples" / "processing"))
from portiloop_analysis.preprocessing import filter_band  # noqa: E402

FS = 250
TRUNCATE = True  # software of the tutorial night (main at 917f1f3); False = main after the fix e4959f1
DATA = HERE.parents[1] / "assets" / "data"

# Notch coefficients of custom_processors.Notch at 250 Hz ("from the Portiloop PLOS ONE paper")
NOTCH = {60: (-0.12478308884588535, 0.98729186796473023, 0.99364593398236511, -0.12478308884588535, 0.99364593398236511),
         50: (-0.61410695998423581, 0.98729186796473023, 0.99364593398236511, -0.61410695998423581, 0.99364593398236511)}


def device_filter(x, power_line):
    """SlowOscillationFilter at 250 Hz, written with lfilter (same equations, zero initial states)."""
    c1, c2, c3, c4, c5 = NOTCH[power_line]
    alpha = 0.005
    y = lfilter(firwin(20, [0.5, 30.0], pass_zero=False, window="hamming", fs=FS), [1.0], x)
    y = lfilter([c3, c4, c5], [1.0, c1, c2], y)
    # DC: dc[n] = (1 - alpha) dc[n-1] + alpha x[n]; y = x - dc
    y = lfilter([1 - alpha, -(1 - alpha)], [1.0, -(1 - alpha)], y)
    return lfilter(firwin(17, [0.16, 4], fs=FS, pass_zero="bandpass"), [1.0], y)


class UpstateDetector:
    """Copy of SlowOscillationDetector.run_detection in SECOND_UPSTATE mode (custom_detectors.py, see TRUNCATE)."""
    th_PaP, th_Neg, min_tNe, max_tNe, max_tPo = 75, 40, 125, 1500, 1000

    def __init__(self):
        self.max_peak, self.min_peak, self.down, self.up = -1000, 1000, 0, 0
        self.prev, self.counter, self.t_up, self.est, self.alpha = None, None, None, None, 0.1

    def step(self, v):
        if self.counter is not None:
            self.counter += 1
        if v > self.max_peak:
            self.max_peak = v
            if self.counter is not None:
                self.t_up = self.counter
        if v < self.min_peak:
            self.min_peak = v
        if v >= 0:
            self.up += 1
        else:
            self.down += 1
        p2p, neg = abs(self.max_peak - self.min_peak), abs(self.min_peak)
        t_below, t_above = self.down / FS * 1000, self.up / FS * 1000
        so = p2p > self.th_PaP and neg > self.th_Neg and self.min_tNe < t_below < self.max_tNe and t_above < self.max_tPo
        target = None if self.est is None else (self.est if TRUNCATE else round(self.est))
        hit = so and self.est is not None and self.counter is not None and self.counter == target
        info = dict(p2p=p2p, neg=neg, below_ms=t_below, above_ms=t_above, est_ms=self.est * 1000 / FS,
                    since_up=self.counter, wave_len=self.down + self.up) if hit else None
        self.prev = self.prev if self.prev is not None else v
        if self.prev * v <= 0:
            if 0 < self.prev:
                if self.counter is not None and so:
                    if self.est is None:
                        self.est = self.t_up
                    else:
                        self.est = self.alpha * self.t_up + (1 - self.alpha) * self.est
                        if TRUNCATE:
                            self.est = int(self.est)
                self.max_peak, self.min_peak, self.down, self.up = -1000, 1000, 0, 0
                self.prev, self.counter = None, None
            else:
                self.counter, self.t_up = 0, 0
        self.prev = v
        return info


def b64_int16(x):
    y = np.clip(np.round(np.asarray(x, float) * 10), -32767, 32767)
    y[~np.isfinite(np.asarray(x, float))] = -32768
    return base64.b64encode(y.astype("<i2").tobytes()).decode()


def main(folder):
    folder = Path(folder)
    csv = next(f for f in folder.glob("*.csv") if not f.stem.endswith(("_nanosleepnet", "_simulated")))
    meta = json.loads((folder / f"{csv.stem}_metadata.json").read_text())
    power_line = meta["filter_settings"]["power_line"]
    channel = meta["channel_detection"]
    excerpts = json.loads((DATA / "excerpts.json").read_text())
    deep = next(e for e in excerpts["excerpts"] if e["key"] == "deep")
    start, n = int(round(deep["start_s"] * FS)), 30 * FS

    data = pd.read_csv(csv, usecols=[f"raw_ch{channel}", "stimulation", "stimulation_on"])
    raw = data[f"raw_ch{channel}"].to_numpy(float)
    on = data["stimulation_on"].to_numpy()
    stim = np.flatnonzero(data["stimulation"].to_numpy() == 1)

    device = device_filter(raw, power_line)
    det, infos = UpstateDetector(), {}
    for i in range(len(raw)):
        if on[i] == 1:
            info = det.step(device[i])
            if info is not None:
                infos[i] = info
    found = np.array(sorted(infos))
    print(f"TRUNCATE = {TRUNCATE}: detections of the copy: {len(found)}; recorded stimulations: {len(stim)}; "
          f"on the same sample: {len(np.intersect1d(found, stim))}")
    est = np.array([infos[i]["est_ms"] for i in found])
    print(f"up-state estimate at the detections (ms): median {np.median(est):.0f}, 5th to 95th percentile "
          f"{np.percentile(est, 5):.0f} to {np.percentile(est, 95):.0f}")

    margin = 60 * FS
    seg = raw[start - margin:start + n + margin].copy()
    seg[np.abs(seg) >= 187000] = np.nan
    slow = filter_band(seg, (0.1, 4.0), FS, notch_hz=power_line)[margin:margin + n]

    in_win = [i for i in found if start <= i < start + n]
    out = {
        "description": "Tutorial night, deep excerpt (Cz). device: input of the slow-oscillation detector "
                       "(SlowOscillationFilter, causal); slow: 0.1-4 Hz zero-phase band of the analysis; detections: "
                       "copy of SOUpstateDetector (branch main, truncated estimate) with its criteria values; "
                       "stims: recorded stimulations.",
        "truncated_estimate": TRUNCATE,
        "fs": FS, "start_s": start / FS, "seconds": 30, "channel": channel, "power_line": power_line,
        "encoding": "int16, 0.1 uV units, -32768 = missing",
        "device": b64_int16(device[start:start + n]), "slow": b64_int16(slow),
        "stims": [int(s - start) for s in stim if start <= s < start + n],
        "detections": [dict(sample=int(i - start), p2p=round(infos[i]["p2p"], 1), neg=round(infos[i]["neg"], 1),
                            below_ms=round(infos[i]["below_ms"]), above_ms=round(infos[i]["above_ms"]),
                            est_ms=round(infos[i]["est_ms"]), up_cross=int(i - start - infos[i]["since_up"]),
                            wave_start=int(i - start - infos[i]["wave_len"] + 1)) for i in in_win],
    }
    path = DATA / "01-overview-deep.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")
    for d in out["detections"]:
        print(d)
    print("recorded stims in window:", out["stims"])


if __name__ == "__main__":
    main(sys.argv[1])
