"""
build_replay_data.py: data for the widgets of tutorial 08 (docs/assets/data/08-replay.json).

    python docs/tools/build_replay_data.py DATASET_FOLDER SIMULATED_CSV [SIMULATED_FIXED_CSV]

DATASET_FOLDER is the unzipped tutorial dataset (the folder holding <name>.csv). SIMULATED_CSV is the output of the
simulator of the software version that recorded the tutorial night (portiloop-software commit 917f1f3) on that
recording; SIMULATED_FIXED_CSV, optional, the output of the simulator after commit e4959f1 ("Fixed the early
slow-oscillation stimulations": the up-state delay estimate is kept as a float instead of being truncated to an
integer at each update). With each version checked out:

    python portiloop/src/simulator.py -i DATASET_FOLDER/<name>.csv -o simulated.csv -q

Needs numpy and pandas. Reads only the columns it needs (each simulated file is about 1.7 GB for the tutorial night).

What the file contains (about 0.4 MB):
    recorded, on_recorded   sample index of every stimulation recorded by the device; [first, end) sample runs where
                            its stimulation_on is 1
    replays.recorded_version, replays.fixed   for each replay: "stimulations", "on" (stimulation_on runs), "lag" (for
                            every recorded stimulation, replayed minus recorded sample index of its pair, or null) and
                            "est_upstate" (the detector's estimate of the delay from the upward zero crossing to the
                            up-state peak, in samples, just after each of its stimulations)
    match_tolerance         25 samples (100 ms): recorded and replayed stimulations are paired one to one, closest
                            first, when they are at most this far apart
    excerpts                30 s windows of filtered_ch6 (the device-filtered Cz written by the simulator, uV, base64
                            little-endian float64, exact; identical in both versions), the recorded stimulations inside,
                            and for each replay its stimulations inside and the state of the slow-oscillation detector
                            at the first sample, so that the page can run the detection itself from there.

The script contains a Python copy of SOUpstateDetector.run_detection (portiloop/src/custom/custom_detectors.py, lines
214 to 288 at 917f1f3) in both versions (917f1f3: estimate truncated with int() and compared as is; e4959f1: float
estimate compared with round()) and of the 0.1 s spacing of SlowOscillationStimulator. It checks that each copy
reproduces its simulator's detection and stimulation columns sample for sample, and stops if it does not.
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FS = 250
TOLERANCE = 25
EXCERPTS = [  # (key, start minute, title)
    ("deep", 123.0, "Deep sleep: large slow waves, with stimulations on many of them"),
    ("artifact", 297.2, "A movement artifact: the criteria have no upper limit, so it triggers a stimulation"),
    ("after", 407.2, "After the electrodes came off: the device had stopped, the replay keeps stimulating"),
]
OUT = Path(__file__).resolve().parents[1] / "assets" / "data" / "08-replay.json"


class SODetector:
    """Copy of SlowOscillationDetector.run_detection in SECOND_UPSTATE mode (SOUpstateDetector)."""
    th_pap, th_neg, min_tne, max_tne, max_tpo, alpha = 75, 40, 125, 1500, 1000, 0.1

    def __init__(self, fixed=False):
        self.fixed = fixed  # False: commit 917f1f3 (recorded the night); True: after commit e4959f1
        self.max_peak, self.min_peak, self.down, self.up = -1000.0, 1000.0, 0, 0
        self.prev = self.counter = self.t_up = self.est = None

    def state(self):
        return {"max_peak": self.max_peak, "min_peak": self.min_peak, "down": self.down, "up": self.up,
                "prev": self.prev, "counter": self.counter, "t_up": self.t_up, "est": self.est}

    def step(self, x):
        if self.counter is not None:
            self.counter += 1
        if x > self.max_peak:
            self.max_peak = x
            if self.counter is not None:
                self.t_up = self.counter
        if x < self.min_peak:
            self.min_peak = x
        if x >= 0:
            self.up += 1
        else:
            self.down += 1
        tne, tpo = self.down / FS * 1000, self.up / FS * 1000
        so = (abs(self.max_peak - self.min_peak) > self.th_pap and abs(self.min_peak) > self.th_neg
              and self.min_tne < tne < self.max_tne and tpo < self.max_tpo)
        target = py_round(self.est) if self.fixed and self.est is not None else self.est
        upstate = so and self.est is not None and self.counter is not None and self.counter == target
        self.prev = self.prev if self.prev is not None else x
        if self.prev * x <= 0:
            if 0 < self.prev:
                if self.counter is not None and so:
                    if self.est is None:
                        self.est = self.t_up
                    else:
                        self.est = self.alpha * self.t_up + (1 - self.alpha) * self.est
                        if not self.fixed:  # 917f1f3 truncates the running average to an integer at every update
                            self.est = int(self.est)
                self.max_peak, self.min_peak, self.down, self.up = -1000.0, 1000.0, 0, 0
                self.prev = self.counter = None
            else:
                self.counter, self.t_up = 0, 0
        self.prev = x
        return upstate


def py_round(x):
    return int(round(x))  # Python rounds halves to even, as the detector does


def runs(on):
    d = np.diff(np.concatenate(([0], on.astype(int), [0])))
    return [[int(a), int(b)] for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def pair(rec_stim, sim_stim):
    """One-to-one pairing, closest first, within TOLERANCE samples: replayed minus recorded lag per recorded stim."""
    j = np.searchsorted(sim_stim, rec_stim)
    pairs = []
    for i, r in enumerate(rec_stim):
        for k in (j[i] - 1, j[i]):
            if 0 <= k < len(sim_stim) and abs(sim_stim[k] - r) <= TOLERANCE:
                pairs.append((abs(sim_stim[k] - r), i, k))
    pairs.sort()
    lag, used = [None] * len(rec_stim), set()
    for d, i, k in pairs:
        if lag[i] is None and k not in used:
            lag[i] = int(sim_stim[k] - rec_stim[i])
            used.add(k)
    return lag


def replay_detector(filtered, on, fixed, starts):
    """Run the detector copy over the night: detections, stimulations, estimate at each stimulation, snapshots."""
    detector, last_ts = SODetector(fixed), 0.0
    det, stim = np.zeros(len(filtered), np.int8), np.zeros(len(filtered), np.int8)
    snapshots, est_at_stim = {}, []
    xs, on = filtered.tolist(), on.tolist()
    for i in range(len(xs)):
        if i in starts:
            snapshots[starts[i]] = detector.state()
        if not on[i]:
            continue
        if detector.step(xs[i]):
            det[i] = 1
            if i / FS - last_ts > 0.1:
                stim[i] = 1
                est_at_stim.append(detector.est)
            last_ts = i / FS
    return det, stim, est_at_stim, snapshots


def main(folder, simulated):
    folder = Path(folder)
    csv = next(f for f in folder.glob("*.csv") if not f.stem.endswith(("_nanosleepnet", "_simulated")))
    rec = pd.read_csv(csv, usecols=["stimulation", "stimulation_on"])
    rec_stim = np.flatnonzero(rec["stimulation"].to_numpy() == 1)
    starts = {round(m * 60 * FS): key for key, m, _ in EXCERPTS}
    replays, filtered = {}, None
    for version, path in simulated.items():
        sim = pd.read_csv(path, usecols=["filtered_ch6", "detection", "stimulation", "stimulation_on"])
        if len(rec) != len(sim):
            raise SystemExit(f"{path}: {len(sim)} rows, the recording has {len(rec)}")
        f6 = sim["filtered_ch6"].to_numpy(np.float64)
        if filtered is None:
            filtered = f6
        elif not np.array_equal(filtered, f6):
            raise SystemExit("The two replays have different filtered signals; this script assumes they are identical")
        sim_on = sim["stimulation_on"].to_numpy()
        det, stim, est, snaps = replay_detector(f6, sim_on, version == "fixed", starts)
        same = np.array_equal(det, sim["detection"].to_numpy()) and np.array_equal(stim, sim["stimulation"].to_numpy())
        print(f"{version}: Python copy of the detector gives {det.sum()} detections, {stim.sum()} stimulations; "
              f"identical to the simulator: {same}")
        if not same:
            raise SystemExit("The Python copy of the detector does not reproduce the simulator; not writing the file.")
        sim_stim = np.flatnonzero(stim == 1)
        replays[version] = {"stims": sim_stim, "on": sim_on, "est": est, "snaps": snaps, "lag": pair(rec_stim, sim_stim)}

    excerpts = []
    for key, minute, title in EXCERPTS:
        s0 = round(minute * 60 * FS)
        s1 = s0 + 30 * FS
        ex = {"key": key, "title": title, "start": s0, "n": s1 - s0,
              "filtered": base64.b64encode(filtered[s0:s1].astype("<f8").tobytes()).decode(),
              "recorded": [int(s - s0) for s in rec_stim if s0 <= s < s1]}
        for version, r in replays.items():
            ex[version] = {"state": r["snaps"][key], "simulated": [int(s - s0) for s in r["stims"] if s0 <= s < s1],
                           "on": bool(r["on"][s0:s1].all())}
        excerpts.append(ex)
    out = {
        "description": "Tutorial night: stimulations recorded by the device and replayed by portiloop/src/simulator.py, "
                       "with the version that recorded the night (917f1f3) and after commit e4959f1",
        "fs": FS, "n_samples": int(len(rec)), "match_tolerance": TOLERANCE,
        "recorded": rec_stim.tolist(), "on_recorded": runs(rec["stimulation_on"].to_numpy()),
        "replays": {v: {"stimulations": r["stims"].tolist(), "on": runs(r["on"]), "lag": r["lag"],
                        "est_upstate": [round(float(e), 2) for e in r["est"]]} for v, r in replays.items()},
        "detector": {"th_pap_uv": 75, "th_neg_uv": 40, "min_tne_ms": 125, "max_tne_ms": 1500, "max_tpo_ms": 1000,
                     "alpha": 0.1, "stimulator_wait_s": 0.1},
        "excerpts": excerpts,
    }
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB); {len(rec_stim)} recorded stimulations")
    for version, r in replays.items():
        lags = [v for v in r["lag"] if v is not None]
        e = np.array(r["est"])
        last_on = runs(rec["stimulation_on"].to_numpy())[-1][1]
        print(f"{version}: {len(r['stims'])} stimulations ({int((r['stims'] < last_on).sum())} while the device's "
              f"stimulation_on was still 1), "
              f"{len(lags)} pairs within {TOLERANCE} samples, {int(np.isin(rec_stim, r['stims']).sum())} at the "
              f"identical sample; lags {pd.Series(lags).value_counts().sort_index().to_dict()}; estimate median "
              f"%.1f, 5th-95th percentile %.1f-%.1f" % (np.median(e), *np.percentile(e, [5, 95])))
    for x in excerpts:
        print(x["key"], x["start"] / FS / 60, "recorded", len(x["recorded"]),
              {v: (len(x[v]["simulated"]), x[v]["state"]["est"], x[v]["on"]) for v in replays})


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    paths = {"recorded_version": sys.argv[2]}
    if len(sys.argv) == 4:
        paths["fixed"] = sys.argv[3]
    main(sys.argv[1], paths)
