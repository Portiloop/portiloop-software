"""
build_assets.py: the data files the tutorial widgets load (docs/assets/data/*.json), from the tutorial night.

    python docs/tools/build_assets.py DATASET_FOLDER

DATASET_FOLDER is the unzipped tutorial dataset (the folder holding <name>.csv, <name>_metadata.json and
<name>_nanosleepnet.csv). Needs numpy, scipy, pandas and examples/processing/portiloop_analysis.

Outputs (arrays of numbers are stored as base64-encoded little-endian int16 or uint8, see "encoding" in each file):
    night.json     one value per 30 s epoch: automatic sleep stage, band powers, stimulations, Cz spectrogram
    excerpts.json  30 s windows of the 8 channels (0.3 to 35 Hz), chosen automatically: wake, N2, deep sleep with
                   stimulations, REM, and the moment the Cz electrode came off (raw signal)
    epochs.json    every stimulation: time, automatic stage, status in the analysis of tutorial 07, and the Cz
                   slow-band and sigma-envelope epochs from -4 to +4 s at 25 Hz
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt, welch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "examples" / "processing"))
from portiloop_analysis import EpochSettings, event_locked_responses, filter_recording, load_recording  # noqa: E402

FS = 250
EPOCH = 30 * FS
NAMES = ["ECG1", "ECG2", "EOG1", "EOG2", "EMG", "Cz", "Fz", "Fpz"]
RAIL = 187000.0
OUT = HERE.parents[1] / "assets" / "data"


def b64(array, dtype):
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode()


def int16_uv(x, scale=10.0):
    """Microvolts to int16 in units of 1/scale uV; NaN -> -32768."""
    y = np.round(np.asarray(x, dtype=float) * scale)
    y = np.clip(y, -32767, 32767)
    y[~np.isfinite(np.asarray(x, dtype=float))] = -32768
    return y.astype("<i2")


def main(folder):
    folder = Path(folder)
    csv = next(f for f in folder.glob("*.csv") if not f.stem.endswith(("_nanosleepnet", "_simulated")))
    stem = csv.stem
    data = pd.read_csv(csv, dtype=float)
    raw = data[[f"raw_ch{i}" for i in range(1, 9)]].to_numpy()
    stim = np.flatnonzero(data["stimulation"].to_numpy() == 1)
    stim_on = data["stimulation_on"].to_numpy()
    n = len(raw)
    n_epochs = n // EPOCH
    stages = pd.read_csv(folder / f"{stem}_nanosleepnet.csv")["stage"].to_numpy()[:n_epochs]
    metadata = json.loads((folder / f"{stem}_metadata.json").read_text())

    # Display version of every channel: band-pass 0.3 to 35 Hz, zero phase, rail samples set to NaN.
    sos = butter(2, [0.3, 35], btype="band", fs=FS, output="sos")
    railed = np.abs(raw) >= RAIL
    display = np.empty_like(raw)
    for c in range(8):
        x = raw[:, c] - np.median(raw[:, c])
        display[:, c] = sosfiltfilt(sos, np.where(railed[:, c], 0.0, x))
    display[railed] = np.nan

    # ---------------------------------------------------------------- night.json
    freqs = None
    spectrogram = []
    rows = []
    for e in range(n_epochs):
        sl = slice(e * EPOCH, (e + 1) * EPOCH)
        row = {"stage": str(stages[e]), "stims": int(((stim >= sl.start) & (stim < sl.stop)).sum()),
               "sound": round(float(stim_on[sl].mean()), 3), "cz_rail": round(float(railed[sl, 5].mean()), 3)}
        for c, name in enumerate(NAMES):
            x = np.nan_to_num(display[sl, c])
            if name in ("Cz", "EOG1", "EMG"):
                row[f"{name.lower()}_ptp"] = float(np.ptp(x))
            f, p = welch(x, FS, nperseg=4 * FS)
            power = lambda lo, hi: float(p[(f >= lo) & (f < hi)].sum() * (f[1] - f[0]))
            if name == "Cz":
                if freqs is None:
                    freqs = f[(f >= 0.5) & (f <= 25)]
                spectrogram.append(10 * np.log10(p[(f >= 0.5) & (f <= 25)] + 1e-3))
                row.update(cz_delta=power(0.5, 4), cz_theta=power(4, 8), cz_alpha=power(8, 12),
                           cz_sigma=power(11, 16), cz_beta=power(16, 30))
            if name == "EMG":
                row["emg_beta"] = power(16, 30)
            if name == "EOG1":
                row["eog_slow"] = power(0.3, 2)
        rows.append(row)
    table = pd.DataFrame(rows)
    spec = np.array(spectrogram)
    lo, hi = np.percentile(spec, 1), np.percentile(spec, 99.5)
    spec_u8 = np.clip(np.round((spec - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)
    night = {
        "description": "Tutorial night, one value per 30 s epoch. Powers in uV^2 (Welch, 4 s windows, after a 0.3-35 Hz "
                       "band-pass). stage: automatic staging by NanoSleepNet (not validated on Portiloop data). "
                       "sound: fraction of the epoch with detection/stimulation running (stimulation_on). "
                       "cz_rail: fraction of Cz samples at the ADS1299 limit (electrode off).",
        "recording": stem, "fs": FS, "epoch_s": 30, "n_epochs": n_epochs, "n_samples": n,
        "pipeline": metadata.get("pipeline"), "sleep_timeout_min": metadata.get("sleep_timeout"),
        "channel_detection": metadata.get("channel_detection"), "volume": metadata.get("volume"),
        "power_line": metadata.get("filter_settings", {}).get("power_line"),
        "impedances_kohm": metadata.get("impedances_kohm"),
        "stimulation_count": int(len(stim)),
        "columns": {k: [round(v, 2) if isinstance(v, float) else v for v in table[k].tolist()] for k in table.columns},
        "spectrogram": {"channel": "Cz", "freqs_hz": [round(float(v), 2) for v in freqs], "db_min": round(float(lo), 2),
                        "db_max": round(float(hi), 2), "encoding": "uint8, row-major [epoch][freq], value*(max-min)/255+min dB",
                        "data": b64(spec_u8, np.uint8)},
    }

    # ---------------------------------------------------------------- excerpts.json
    t_min = np.arange(n_epochs) * 0.5
    def pick(mask, score):
        idx = np.flatnonzero(mask)
        return int(idx[np.argmax(score[idx])]) if len(idx) else None
    # clean epochs: electrode on, no large deflection on Cz, the EOG or the chin EMG
    clean = ((table["cz_rail"] == 0) & (table["cz_ptp"] < 250) & (table["eog1_ptp"] < 500)
             & (table["emg_ptp"] < 300)).to_numpy()
    rel = lambda k: (table[k] / (table["cz_delta"] + table["cz_theta"] + table["cz_alpha"] + table["cz_beta"])).to_numpy()
    choices = [
        ("wake", "Awake, eyes closed, before sleep: alpha rhythm (8 to 12 Hz), no stimulation yet",
         pick(clean & (t_min < 25), rel("cz_alpha"))),
        ("n2", "N2 sleep: sleep spindles (11 to 16 Hz bursts) and K-complexes",
         pick(clean & (stages == "N2") & (table["stims"].to_numpy() <= 2), rel("cz_sigma"))),
        ("deep", "Deep sleep: large slow waves, and the sounds the Portiloop sent on their up-states",
         pick(clean & (table["stims"].to_numpy() >= 4), table["cz_delta"].to_numpy())),
        ("rem", "REM sleep (automatic staging): rapid eye movements on the EOG channels, low chin EMG",
         pick(clean & (stages == "REM") & (t_min > 60), table["eog_slow"].to_numpy() / (table["emg_beta"].to_numpy() + 1))),
        ("artifact", "A movement during sleep: large deflections on several channels; epochs like these are rejected",
         pick((table["cz_rail"] == 0).to_numpy() & (table["stims"].to_numpy() >= 3), table["cz_ptp"].to_numpy())),
    ]
    # start of the first period in which Cz stays at the rail most of the following minute
    run = np.convolve(railed[:, 5].astype(float), np.ones(60 * FS) / (60 * FS), mode="full")[60 * FS - 1:]
    long_off = np.flatnonzero(run > 0.5)
    off = None
    if len(long_off):
        first = long_off[0]
        off = int(np.flatnonzero(railed[first:first + 60 * FS, 5])[0] + first)
    excerpts = []
    for key, text, e in choices:
        if e is None:
            continue
        start = e * EPOCH
        excerpts.append({"key": key, "title": text, "start_s": start / FS, "stage": str(stages[e])})
    if off is not None:
        excerpts.append({"key": "leadoff", "title": "Near the end of the night a movement disturbs every channel and the "
                         "Cz electrode comes off: its raw signal stays at the limit of the amplifier, and the "
                         "Portiloop stops stimulating", "start_s": (off - 20 * FS) / FS,
                         "stage": str(stages[min(off // EPOCH, n_epochs - 1)]), "raw": True})
    for ex in excerpts:
        start = int(round(ex["start_s"] * FS))
        sl = slice(start, start + EPOCH)
        if ex.get("raw"):
            sig = raw[sl] - np.median(raw[start - 60 * FS:start], axis=0)
            ex["note"] = "raw signal minus its median over the previous minute; values beyond +/-3276 uV are clipped"
        else:
            sig = display[sl]
        ex["data"] = b64(int16_uv(sig.T), "<i2")
        ex["stims"] = [int(s - start) for s in stim if start <= s < start + EPOCH]
        ex["sound"] = [round(float(v), 2) for v in stim_on[sl][::FS]]
    excerpts_json = {"fs": FS, "seconds": 30, "channels": NAMES,
                     "encoding": "int16, row-major [channel][sample], 0.1 uV units, -32768 = missing",
                     "excerpts": excerpts}

    # ---------------------------------------------------------------- epochs.json
    recording = load_recording(csv, channel="Cz", montage="default", verbose=False)
    settings = EpochSettings(notch_hz=metadata.get("filter_settings", {}).get("power_line", 60))
    filtered = filter_recording(recording, settings)
    responses = event_locked_responses(recording, settings, filtered=filtered)
    status = responses.events.set_index("sample")["status"]
    half, step = 4 * FS, FS // 25
    offsets = np.arange(-half, half + 1, step)
    slow, sigma = [], []
    for s in stim:
        idx = s + offsets
        inside = (idx >= 0) & (idx < n)
        a = np.full(len(offsets), np.nan)
        b = np.full(len(offsets), np.nan)
        a[inside] = filtered["slow"][idx[inside]]
        piece = filtered["sigma"][max(0, s - half):min(n, s + half + 1)]
        if np.isfinite(piece).all() and len(piece) == 2 * half + 1:
            b = np.abs(hilbert(piece))[offsets + half]
        slow.append(a)
        sigma.append(b)
    epochs_json = {
        "fs": 25, "t0_s": -4.0, "n_times": len(offsets), "n_events": int(len(stim)),
        "settings": {"notch_hz": settings.notch_hz, "slow_band": settings.slow_band, "sigma_band": settings.sigma_band,
                     "max_ptp_uv": settings.max_ptp_uv, "jump_limit_uv": settings.jump_limit_uv},
        "events": {"minute": [round(s / FS / 60, 3) for s in stim],
                   "stage": [str(stages[min(s // EPOCH, n_epochs - 1)]) for s in stim],
                   "status": [str(status.get(int(s), "kept")) for s in stim]},
        "encoding": "int16, row-major [event][time], 0.1 uV units, -32768 = missing; sigma is the Hilbert envelope "
                    "of the 11-16 Hz band (no baseline), slow is the 0.1-4 Hz band (no baseline)",
        "slow": b64(int16_uv(np.array(slow)), "<i2"), "sigma": b64(int16_uv(np.array(sigma)), "<i2"),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    for name, obj in (("night.json", night), ("excerpts.json", excerpts_json), ("epochs.json", epochs_json)):
        (OUT / name).write_text(json.dumps(obj, separators=(",", ":")))
        print(f"wrote {OUT / name} ({(OUT / name).stat().st_size / 1024:.0f} KiB)")
    print("excerpts:", [(e["key"], round(e["start_s"] / 60, 1), e["stage"], len(e["stims"])) for e in excerpts])
    print("statuses:", pd.Series(epochs_json["events"]["status"]).value_counts().to_dict())


if __name__ == "__main__":
    main(sys.argv[1])
