"""
make_tutorial_dataset.py: turn a Portiloop night into the public dataset of the tutorials.

- renames the files (the recording name chosen at the bedside often contains the participant's name);
- removes the device path and the recording name from the metadata;
- writes the samples with 3 decimals (0.001 uV; the ADS1299 step at gain 24 and VREF 4.5 V is 0.022 uV, so no
  information is lost) to make the files smaller;
- packs the folder into a zip archive.

Usage:
    python docs/tools/make_tutorial_dataset.py ORIGINAL.csv OUTPUT_FOLDER --name clas-tutorial [--stages STAGES.csv]
"""
import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("csv", type=Path, help="original recording (.csv), with its _metadata.json next to it")
    p.add_argument("out", type=Path, help="output folder")
    p.add_argument("--name", default="clas-tutorial", help="new recording name (replaces the part before the date)")
    p.add_argument("--stages", type=Path, help="optional automatic sleep staging to include (CSV)")
    a = p.parse_args()

    stem = a.csv.stem
    match = re.search(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(_.*)?$", stem)
    if not match:
        raise SystemExit(f"No date and time in {a.csv.name}")
    new_stem = f"{a.name}_{match.group(1)}{match.group(2) or ''}"
    folder = a.out / new_stem
    folder.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(a.csv.with_name(f"{stem}_metadata.json").read_text())
    metadata["filename"] = f"{new_stem}/{new_stem}.csv"
    (folder / f"{new_stem}_metadata.json").write_text(json.dumps(metadata, indent=4))
    for suffix in ("leadoff", "stats"):
        source = a.csv.with_name(f"{stem}_{suffix}.json")
        if source.exists():
            shutil.copy(source, folder / f"{new_stem}_{suffix}.json")
    if a.stages:
        shutil.copy(a.stages, folder / f"{new_stem}_nanosleepnet.csv")

    out_csv = folder / f"{new_stem}.csv"
    first = True
    for chunk in pd.read_csv(a.csv, chunksize=500_000):
        chunk.to_csv(out_csv, mode="w" if first else "a", header=first, index=False, float_format="%.3f")
        first = False

    archive = a.out / f"{new_stem}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(folder.iterdir()):
            z.write(f, f"{new_stem}/{f.name}")
    print(f"wrote {folder} and {archive} ({archive.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
