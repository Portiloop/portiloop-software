"""
stage_night.py: automatic sleep staging of a Portiloop recording, offline, with the NanoSleepNet model.

    python docs/tools/stage_night.py RECORDING.csv OUTPUT.csv [--fpz 8] [--cz 6]

Needs the code of the next Portiloop software release (branch rdkx5 of portiloop-software): the model and the
module portiloop/src/custom/sleep_staging.py are not part of the software of Portiloop v2.3 (branch main). Copy this
script to docs/tools/ of a checkout of that branch and run it there (it imports portiloop/src from two folders up). Nothing else in the tutorials depends on it: the stages of the tutorial night are
already in its <name>_nanosleepnet.csv file, which this script made.

It feeds the raw channels of a recording, 30 s epoch after 30 s epoch, to two classes of sleep_staging.py:
- Preprocessor: Fpz minus Cz (both referenced to the same electrode), first-order 0.5 Hz high-pass, 250 Hz to
  100 Hz, clipped to +/-192 uV, cut into 30 s epochs;
- NanoSleepNetStep: stages each epoch from it and the 9 epochs before it (causal).
Every epoch of the file is staged, including epochs recorded with an electrode off.

The output has one row per 30 s epoch: epoch (from 0), start_s (seconds from the start of the recording), stage
(W, N1, N2, N3 or REM: the largest logit) and p_W ... p_REM (softmax of the logits).

The model was trained on Sleep-EDF (Fpz-Cz) and has not been validated on Portiloop recordings (README of the
model): check its stages before using them.

Needs numpy, scipy and pandas, and that checkout with the model file portiloop/models/nanosleepnet/nanosleepnet.npz. Importing the Portiloop package imports the module
portilooplot (plots on the device); if it is not installed (pip install portilooplot), an empty stand-in is used,
since staging does not plot anything.
"""
import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

try:
    import portilooplot  # noqa: F401  (only needed for plots on the device)
except ImportError:
    for name in ("portilooplot", "portilooplot.jupyter_plot"):
        stand_in = types.ModuleType(name)
        stand_in.ProgressPlot = object
        sys.modules[name] = stand_in

from portiloop.src.custom.sleep_staging import STAGES, NanoSleepNetStep, Preprocessor  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv", type=Path, help="recording (.csv) with raw_ch1 ... raw_ch8 in microvolts")
    parser.add_argument("out", type=Path, help="output table (.csv), one row per 30 s epoch")
    parser.add_argument("--fpz", type=int, default=8, help="channel of Fpz, from 1 (default 8, CLAS protocol montage)")
    parser.add_argument("--cz", type=int, default=6, help="channel of Cz, from 1 (default 6, CLAS protocol montage)")
    args = parser.parse_args()

    columns = [f"raw_ch{i}" for i in range(1, 9)]
    preprocessor = Preprocessor(fpz=args.fpz - 1, cz=args.cz - 1)  # 0-based
    model = NanoSleepNetStep(NanoSleepNetStep.FOLDER)
    rows, start, epoch_number = [], time.time(), 0
    # 10 minutes of samples at a time: the results do not depend on the chunk size
    for chunk in pd.read_csv(args.csv, usecols=columns, dtype=float, chunksize=250 * 600):
        for epoch in preprocessor.add_datapoints(chunk[columns].to_numpy()):
            logits = model.step(epoch)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            rows.append({"epoch": epoch_number, "start_s": epoch_number * 30,
                         "stage": STAGES[int(np.argmax(logits))],
                         **{f"p_{stage}": float(value) for stage, value in zip(STAGES, p)}})
            epoch_number += 1
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"{epoch_number} epochs staged in {time.time() - start:.0f} s: {args.out}")


if __name__ == "__main__":
    main()
