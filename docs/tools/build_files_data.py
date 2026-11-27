"""
build_files_data.py: the data of the file explorer of tutorial 05 (docs/assets/data/05-files.json), from the tutorial
night.

    python docs/tools/build_files_data.py DATASET_FOLDER

DATASET_FOLDER is the unzipped tutorial dataset (the folder holding <name>.csv and <name>_metadata.json), a night
recorded with the software of Portiloop v2.3. Needs numpy and pandas only (it does not import the Portiloop software).

05-files.json holds:
    header      the header line of the CSV file
    contexts    a few 2 s stretches of the CSV file, copied line by line (500 rows each, exactly as in the file):
                the first rows, the moment detection starts, a stimulation, a repeated row (lost sample), the moment
                the Cz electrode reaches the amplifier limit and the moment detection stops
    metadata    the content of <name>_metadata.json
    and the recording name, the sampling frequency and the number of rows.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
OUT = HERE.parents[1] / "assets" / "data" / "05-files.json"
HALF = 250  # rows before the row of interest (1 s at 250 Hz); each context has 2 * HALF rows
RAIL = 187000.0


def main(folder):
    folder = Path(folder)
    csv = next(f for f in folder.glob("*.csv") if not f.stem.endswith(("_nanosleepnet", "_simulated")))
    stem = csv.stem
    metadata = json.loads((folder / f"{stem}_metadata.json").read_text())
    fs = int(metadata["frequency"])

    data = pd.read_csv(csv, dtype=float)
    n = len(data)
    raw = data[[f"raw_ch{i}" for i in range(1, 9)]]
    on = data["stimulation_on"].to_numpy()
    stim = np.flatnonzero(data["stimulation"].to_numpy() == 1)
    switches = np.flatnonzero(np.diff(on)) + 1
    repeated = np.flatnonzero((raw.diff() == 0).all(axis=1).to_numpy())
    cz_railed = np.abs(raw["raw_ch6"].to_numpy()) >= RAIL
    # first sample of the period in which Cz stays at the limit most of the following minute
    run = np.convolve(cz_railed.astype(float), np.ones(60 * fs) / (60 * fs), mode="full")[60 * fs - 1:]
    first_long = np.flatnonzero(run > 0.5)[0]
    cz_off = int(np.flatnonzero(cz_railed[first_long:first_long + 60 * fs])[0] + first_long)
    on_start = int(switches[on[switches] == 1][0])
    on_stop = int(switches[on[switches] == 0][-1])
    one_stim = int(stim[np.searchsorted(stim, 123 * 60 * fs)])  # first stimulation after minute 123
    one_repeat = int(repeated[np.searchsorted(repeated, 60 * 60 * fs)])  # first repeated row after minute 60

    targets = [
        ("start", "First rows of the file", 0, 0),
        ("on", "Detection starts (end of the \"Stimulation starts after\" delay)", on_start, on_start - HALF),
        ("stim", "A stimulation", one_stim, one_stim - HALF),
        ("repeat", "A row identical to the previous one (lost sample)", one_repeat, one_repeat - HALF),
        ("rail", "The Cz electrode reaches the amplifier limit", cz_off, cz_off - HALF),
        ("off", "Detection stops (lead-off protection)", on_stop, on_stop - HALF),
    ]
    wanted = {}
    for _, _, _, first in targets:
        for row in range(first, first + 2 * HALF):
            wanted[row] = None
    last = max(wanted)
    with open(csv) as f:
        header = f.readline().strip()
        for row, line in enumerate(f):
            if row in wanted:
                wanted[row] = line.strip()
            if row >= last:
                break
    contexts = [{"key": key, "title": title, "row": row, "first_row": first,
                 "lines": [wanted[r] for r in range(first, first + 2 * HALF)]}
                for key, title, row, first in targets]
    out = {"recording": stem, "fs": fs, "n_rows": n, "header": header, "contexts": contexts, "metadata": metadata,
           "counts": {"stimulations": int(len(stim)), "repeated_rows": int(len(repeated)),
                      "cz_railed_samples": int(cz_railed.sum())}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")
    print([(c["key"], c["row"], round(c["row"] / fs / 60, 3)) for c in contexts], out["counts"])


if __name__ == "__main__":
    main(sys.argv[1])
