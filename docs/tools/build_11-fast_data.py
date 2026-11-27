"""
build_11-fast_data.py: the /da/ syllable of the FFR pipeline, for the widgets of tutorial 11
(docs/assets/data/11-da.json).

    python docs/tools/build_11-fast_data.py PORTILOOP_SOFTWARE_CHECKOUT

PORTILOOP_SOFTWARE_CHECKOUT is a checkout of the code under review (branch fast-acquisition of
Portiloop/portiloop-software, or the head c528336 of pull request #12) that holds
portiloop/sounds/ffr/da_positive.wav (stored with Git LFS: run "git lfs pull" if the file is a small text pointer).
Needs numpy and scipy.

The file is a mono IEEE float WAV, 5292 samples at 44.1 kHz (120 ms), described by da_positive.json next to it
(a synthetic /da/ made with Praat's KlattGrid, F0 98 Hz). The script checks its SHA-256 against that JSON, then writes:
  - "x44": the samples at 44.1 kHz, rounded to 5 decimals (1.0 = digital full scale);
  - "x16": the same sound resampled to 16 kHz (scipy.signal.resample_poly(x, 160, 441)), the highest rate of the
    fast acquisition, rounded to 5 decimals: the widgets use it as the time base of the modelled responses;
  - "info": rate, number of samples, peak, RMS, the SHA-256, and the F0 estimated here by autocorrelation
    (40 to 110 ms, lags of 60 to 300 Hz).
Output: about 60 KB.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

HERE = Path(__file__).resolve()
OUT = HERE.parents[1] / "assets" / "data" / "11-da.json"


def main(checkout):
    folder = Path(checkout) / "portiloop" / "sounds" / "ffr"
    wav = folder / "da_positive.wav"
    raw = wav.read_bytes()
    if raw[:4] != b"RIFF":
        sys.exit(f"{wav} is not a WAV file (a Git LFS pointer? run 'git lfs pull').")
    sha = hashlib.sha256(raw).hexdigest()
    desc = json.loads((folder / "da_positive.json").read_text())
    if sha != desc["wav_sha256"]:
        sys.exit(f"SHA-256 mismatch: {sha} vs {desc['wav_sha256']} in da_positive.json")
    fs, x = wavfile.read(str(wav))
    x = x.astype(np.float64)
    assert fs == 44100 and x.ndim == 1 and len(x) == 5292, (fs, x.shape)
    x16 = resample_poly(x, 160, 441)

    seg = x[int(0.040 * fs):int(0.110 * fs)] - x[int(0.040 * fs):int(0.110 * fs)].mean()
    ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
    lo, hi = int(fs / 300), int(fs / 60)
    f0 = fs / (lo + int(np.argmax(ac[lo:hi])))

    out = {
        "source": "portiloop/sounds/ffr/da_positive.wav (code under review: branch fast-acquisition, PR #12)",
        "info": {"rate_hz": fs, "samples": len(x), "duration_ms": 1000 * len(x) / fs,
                 "peak": float(np.max(np.abs(x))), "rms": float(np.sqrt(np.mean(x ** 2))), "sha256": sha,
                 "f0_autocorr_hz": round(f0, 2), "f0_in_json_hz": desc["recipe"]["layers"][0]["sound"]["frequency_hz"],
                 "method": desc.get("speech_method")},
        "x44": [round(float(v), 5) for v in x],
        "rate16_hz": 16000,
        "x16": [round(float(v), 5) for v in x16],
    }
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"{OUT}: {OUT.stat().st_size / 1024:.1f} KB; peak {out['info']['peak']:.4f}, RMS {out['info']['rms']:.4f}, "
          f"F0 {f0:.2f} Hz, {len(x16)} samples at 16 kHz")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
