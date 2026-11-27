"""
Loading Portiloop recordings.

A Portiloop recording made with the current software is a folder of files sharing one name:

    <name>.csv                one row per sample: raw_ch1 ... raw_ch8 (microvolts), detection, stimulation,
                              stimulation_on (some pipelines also write filtered_ch1 ... filtered_ch8 and timestamp).
                              stimulation is 1 when a sound was played (Cond B) or would have been played (Cond A,
                              sham); stimulation_on is 1 while detection was running (not during the delay before
                              stimulation, nor while an electrode, the reference or the bias was off).
    <name>_metadata.json      settings of the recording (sampling frequency, detection channel, mains frequency, ...)
    <name>_leadoff.json       electrodes that came off during the recording (written when lead-off detection is on)
    <name>_stats.json         timing statistics of the acquisition

The name ends with the start date and time, for example ``Par1_2026-10-15_13-55-36_CONDB``, which is used to
put several files of the same night in chronological order.

Older Portiloop software wrote other layouts; ``load_legacy_recording`` reads the two that appear in past studies.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

# Electrode connected to each ADS1299 input (raw_ch1 ... raw_ch8). Check them against the actual wiring of your cap.
MONTAGES: Dict[str, Tuple[str, ...]] = {
    "default": ("ECG1", "ECG2", "EOG1", "EOG2", "EMG", "Cz", "Fz", "Fpz"),
    "no_ecg": ("Fpz", "Cz", "Pz", "Fz", "EOG1", "EOG2", "EMG1", "EMG2"),
}
EEG_CHANNELS = ("Fpz", "Fz", "Cz", "Pz")


@dataclass
class Recording:
    """One channel of one night, possibly assembled from several files.

    Attributes:
        eeg: signal of the channel in microvolts, one value per sample; NaN where the data are missing or
            unusable (gaps between files, electrode, reference or bias off, samples without stimulation marker).
        markers: one integer per sample; 1 where a stimulation was sent, 0 elsewhere.
        fs: sampling frequency in Hz.
        channel: name of the electrode (for example "Cz").
        leadoff: True for the samples during which the electrode, the reference or the bias was off.
        breaks: first sample of every file after the first; signals are never filtered across these points.
        files: one row per file: file, first_sample (on the assembled timeline) and n_samples.
        stages: optional sleep stage of every sample (for example "N2"), "" where unknown.
        metadata: content of the _metadata.json file of the first file, if any.
    """
    eeg: np.ndarray
    markers: np.ndarray
    fs: float
    channel: str
    leadoff: np.ndarray
    breaks: List[int] = field(default_factory=list)
    files: pd.DataFrame = field(default_factory=pd.DataFrame)
    stages: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.eeg)

    @property
    def duration_h(self) -> float:
        return self.n_samples / self.fs / 3600

    @property
    def events(self) -> np.ndarray:
        """Sample index of every stimulation marker."""
        return np.flatnonzero(self.markers == 1)

    def summary(self) -> pd.Series:
        return pd.Series({
            "channel": self.channel,
            "files": len(self.files),
            "duration (h, with gaps)": round(self.duration_h, 3),
            "missing samples": int(np.count_nonzero(~np.isfinite(self.eeg))),
            "samples with electrode off": int(np.count_nonzero(self.leadoff)),
            "stimulation markers": int(len(self.events)),
        })


# ------------------------------------------------------------------------------------------- companion files

def companion_file(csv_file: PathLike, suffix: str) -> Path:
    """Path of a companion file: companion_file("a/rec.csv", "metadata") -> a/rec_metadata.json."""
    csv_file = Path(csv_file)
    return csv_file.with_name(f"{csv_file.stem}_{suffix}.json")


def read_json(path: PathLike) -> dict:
    """Read a JSON file written in UTF-8 (with or without byte order mark) or UTF-16 (files edited on Windows)."""
    raw = Path(path).read_bytes()
    try:
        return json.loads(raw.decode("utf-8-sig"))
    except UnicodeError:
        return json.loads(raw.decode("utf-16"))


def read_metadata(csv_file: PathLike) -> dict:
    """Settings of a recording (its _metadata.json), or an empty dict when the file does not exist."""
    path = companion_file(csv_file, "metadata")
    return read_json(path) if path.exists() else {}


def recording_start(csv_file: PathLike) -> datetime:
    """Start date and time encoded in the file name, for example ..._2026-10-15_13-55-36_CONDB.csv."""
    match = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", Path(csv_file).stem)
    if match is None:
        raise ValueError(f"No recording date and time (YYYY-MM-DD_HH-MM-SS) in the file name {Path(csv_file).name}")
    return datetime.strptime(match.group(), "%Y-%m-%d_%H-%M-%S")


def read_leadoff_periods(csv_file: PathLike, n_samples: int, channel_number: int) -> List[Tuple[int, int]]:
    """Periods (first sample, end sample) during which the data of a channel cannot be trusted.

    The _leadoff.json file lists events: a channel "disabled" (its electrode came off; it stays off until the end of
    the recording), "not_connected_at_start", "reference_or_bias_off" and "reference_or_bias_recovered". A period
    starts with an event that turns the selected channel, the reference or the bias off, and ends with the matching
    recovery event, or at the end of the recording.

    Args:
        csv_file: the recording .csv file (the _leadoff.json file is found next to it).
        n_samples: number of samples of the recording.
        channel_number: ADS1299 input of the channel, from 1 (raw_ch1) to 8.

    Returns:
        list of (first, end) sample pairs, end excluded. Empty when the file does not exist.
    """
    path = companion_file(csv_file, "leadoff")
    if not path.exists():
        return []
    log = read_json(path)
    open_periods: Dict[str, int] = {}
    periods = []
    for item in sorted(log.get("events", []), key=lambda event: int(event["sample"])):
        kind = item["event"].lower()
        sample = int(item["sample"])
        if not 0 <= sample <= n_samples:
            raise ValueError(f"Lead-off event outside the recording in {path.name}: sample {sample}")
        # Reference, bias and recording gaps affect every channel; other events only the listed channels.
        shared = next((key for key in ("reference", "bias", "gap") if key in kind), None)
        if shared is None and channel_number not in item.get("channels", []):
            continue
        key = shared or "selected_channel"
        starts = kind == "disabled" or kind.endswith("_off") or "not_connected" in kind
        ends = kind == "enabled" or kind.endswith("_recovered") or kind.endswith("_on")
        if starts:
            open_periods.setdefault(key, sample)
        elif ends and key in open_periods:
            periods.append((open_periods.pop(key), sample))
    periods.extend((start, n_samples) for start in open_periods.values())
    return periods


# ------------------------------------------------------------------------------------------------- loading

def channel_column(channel: str, montage: Union[str, Sequence[str]] = "default") -> Tuple[str, int]:
    """CSV column (raw_chN) and ADS1299 input number (N) of an electrode in a montage."""
    names = MONTAGES[montage] if isinstance(montage, str) else tuple(montage)
    if isinstance(montage, str) and montage not in MONTAGES:
        raise ValueError(f"Unknown montage {montage!r}; choose from {list(MONTAGES)} or give the 8 electrode names")
    if channel not in names:
        raise ValueError(f"{channel} is not in the montage {names}")
    number = names.index(channel) + 1
    return f"raw_ch{number}", number


def load_recording(files: Union[PathLike, Iterable[PathLike]], channel: str = "Cz",
                   montage: Union[str, Sequence[str]] = "default", fs: Optional[float] = None,
                   markers: str = "stimulation", verbose: bool = True) -> Recording:
    """Load one channel and the stimulation markers of one or several Portiloop recordings of the same night.

    Several files are put in chronological order (from the date and time in their names) and placed on a single
    timeline; the time between two files is kept as missing data (NaN), so that event times stay correct.
    Samples recorded while the electrode, the reference or the bias was off (from the _leadoff.json files) are
    flagged in ``Recording.leadoff`` and replaced by NaN, so that no filter runs through them.

    Args:
        files: one .csv file or a list of them.
        channel: electrode to analyse; it must be in the montage.
        montage: name of a montage in MONTAGES, or the 8 electrode names connected to raw_ch1 ... raw_ch8.
        fs: sampling frequency in Hz; by default read from the _metadata.json file (250 Hz if absent).
        markers: column giving the event times: "stimulation" (default), or "detection" for sham recordings
            made before 2026-10-13 (commit 9d598ac), which have no stimulation column. Check that detections
            and stimulations coincide in your version of the software before relying on "detection".
        verbose: print what was loaded.
    """
    files = [Path(files)] if isinstance(files, (str, Path)) else [Path(f) for f in files]
    files = [f.resolve() for f in files]
    if not files or len(set(files)) != len(files):
        raise ValueError("Give at least one file, without duplicates.")
    if len(files) > 1:
        files.sort(key=recording_start)
    column, number = channel_column(channel, montage)

    metadata = read_metadata(files[0])
    fs = float(fs or metadata.get("frequency", 250))
    blocks, info, leadoff_periods, breaks = [], [], [], []
    length = 0
    for file in files:
        file_fs = read_metadata(file).get("frequency", fs)
        if file_fs != fs:
            raise ValueError(f"{file.name} was recorded at {file_fs} Hz, not {fs} Hz")
        header = pd.read_csv(file, nrows=0).columns
        if markers not in header:
            raise ValueError(
                f"{file.name} has no '{markers}' column, so the times of the events are unknown. (The "
                "'stimulation_on' column only says whether detection was running.) Sham recordings made before "
                "2026-10-13 have no 'stimulation' column: use markers='detection' for them.")
        data = pd.read_csv(file, usecols=[column, markers], dtype=float).rename(columns={markers: "stimulation"})
        if data.empty:
            raise ValueError(f"No samples in {file.name}")
        # Samples without a stimulation value are not trusted: their EEG is treated as missing.
        missing = data["stimulation"].isna()
        if missing.any():
            data.loc[missing, column] = np.nan
            data.loc[missing, "stimulation"] = 0
        if not np.isin(data["stimulation"], [0, 1]).all():
            raise ValueError(f"Stimulation markers must be 0 or 1 in {file.name}")

        start = 0 if file == files[0] else round((recording_start(file) - recording_start(files[0])).total_seconds() * fs)
        if start < length:
            raise ValueError(f"{file.name} overlaps the previous file; check the files and their start times.")
        if start > length:
            blocks.append(pd.DataFrame({column: np.full(start - length, np.nan), "stimulation": 0.0}))
        if file != files[0]:
            breaks.append(start)
        periods = read_leadoff_periods(file, len(data), number)
        for first, end in periods:  # data recorded with the electrode off are treated as missing
            data.iloc[first:end, data.columns.get_loc(column)] = np.nan
        leadoff_periods += [(start + first, start + end) for first, end in periods]
        blocks.append(data)
        info.append({"file": str(file), "first_sample": start, "n_samples": len(data)})
        length = start + len(data)
        if verbose:
            leadoff_note = "no _leadoff.json" if not companion_file(file, "leadoff").exists() else \
                f"{len(periods)} lead-off period(s) on {channel}"
            print(f"{file.name}: {len(data)} samples ({len(data) / fs / 60:.1f} min), "
                  f"{int((data['stimulation'] == 1).sum())} stimulations, {leadoff_note}")

    data = pd.concat(blocks, ignore_index=True)
    leadoff = np.zeros(len(data), dtype=bool)
    for first, end in leadoff_periods:
        leadoff[first:end] = True
    recording = Recording(eeg=data[column].to_numpy(), markers=data["stimulation"].to_numpy().astype(int), fs=fs,
                          channel=channel, leadoff=leadoff, breaks=breaks, files=pd.DataFrame(info),
                          metadata=metadata)
    return recording


# ----------------------------------------------------------------------------------------- legacy formats

LEGACY_STAGE_CODES = {1: "N1", 2: "N2", 3: "N3", 4: "R"}


def load_legacy_recording(file: PathLike, channel: str = "Cz", fs: float = 250, condition: str = "stim") -> Recording:
    """Load a recording written by older Portiloop software.

    Two layouts are recognized from the header:

    - a header row followed by 7 columns: Fpz, Cz, Fz, Pz, two unused channels, and a stimulation column in which
      2 marks a stimulation and 1 a sham (a detection without sound);
    - columns named "Raw Signal Ch1" ... and "sorted_stim", in which a stimulation is marked by the sleep stage
      during which it happened (1 = N1, 2 = N2, 3 = N3, 4 = REM). Only Cz (Ch2) is known for this layout.

    Args:
        file: the .csv file.
        channel: electrode to load.
        fs: sampling frequency in Hz (not stored in these files).
        condition: "stim" to keep the stimulations, "sham" to keep the shams (first layout only).
    """
    file = Path(file)
    header = pd.read_csv(file, nrows=0, on_bad_lines="warn", encoding_errors="ignore").columns
    stages = None
    if "sorted_stim" in header:
        if channel != "Cz":
            raise ValueError("Only Cz (Raw Signal Ch2) is known for recordings with a sorted_stim column")
        data = pd.read_csv(file, usecols=["Raw Signal Ch2", "sorted_stim"], on_bad_lines="warn")
        eeg = pd.to_numeric(data["Raw Signal Ch2"], errors="coerce").to_numpy()
        codes = data["sorted_stim"].fillna(0).to_numpy().astype(int)
        markers = (codes > 0).astype(int)
        stages = np.array([LEGACY_STAGE_CODES.get(code, "") for code in codes], dtype=object)
    else:
        names = ("Fpz", "Cz", "Fz", "Pz", "unused1", "unused2", "stim")
        if channel not in names[:4]:
            raise ValueError(f"{channel} is not recorded in this layout ({names[:4]})")
        data = pd.read_csv(file, header=None, skiprows=1, usecols=[names.index(channel), 6],
                           on_bad_lines="warn", encoding_errors="ignore")
        data.columns = [channel, "stim"]
        eeg = pd.to_numeric(data[channel], errors="coerce").to_numpy()
        code = {"stim": 2, "sham": 1}[condition]
        markers = (pd.to_numeric(data["stim"], errors="coerce").to_numpy() == code).astype(int)
    return Recording(eeg=eeg, markers=markers, fs=float(fs), channel=channel, leadoff=np.zeros(len(eeg), dtype=bool),
                     files=pd.DataFrame([{"file": str(file), "first_sample": 0, "n_samples": len(eeg)}]),
                     stages=stages)


def load_hypnogram(path: PathLike, fs: float, n_samples: int) -> np.ndarray:
    """Sleep stage of every sample, from a scoring file with one row per scored period.

    The file is a CSV table with the columns "Annotation" (the stage, for example W, N1, N2, N3, R) and "Duration"
    (in seconds); periods follow each other from the start of the recording. Samples after the last period get "".
    """
    scoring = pd.read_csv(path)
    stages = np.repeat(scoring["Annotation"].astype(str).to_numpy(),
                       np.round(scoring["Duration"].to_numpy() * fs).astype(int))
    out = np.full(n_samples, "", dtype=object)
    out[:min(n_samples, len(stages))] = stages[:n_samples]
    return out


# ------------------------------------------------------------------------------------------------ LSL / XDF

def xdf_to_dataframe(xdf_file: PathLike, filtered_stream: str = "Portiloop Filtered",
                     raw_stream: str = "Portiloop Raw Data", marker_stream: str = "Portiloop_stimuli") -> pd.DataFrame:
    """Convert a recording made with LabRecorder (.xdf) from the Portiloop LSL streams into a table.

    The table has a time_stamps column, raw_data_<i> and filtered_data_<i> columns for each channel, and a stimuli
    column set to 1 at the sample closest to each marker. Needs the pyxdf package.
    """
    import pyxdf  # optional dependency, only needed for .xdf files

    streams, _ = pyxdf.load_xdf(str(xdf_file))
    by_name = {stream["info"]["name"][0]: stream for stream in streams}
    filtered, raw, markers = (by_name.get(name) for name in (filtered_stream, raw_stream, marker_stream))
    if filtered is None and raw is None:
        raise ValueError(f"Neither {filtered_stream!r} nor {raw_stream!r} is in {Path(xdf_file).name}")
    present = [s for s in (filtered, raw) if s is not None]
    n = min(len(s["time_stamps"]) for s in present)
    if len({s["time_series"].shape[1] for s in present}) != 1:
        raise ValueError("The raw and filtered streams have different numbers of channels")
    df = pd.DataFrame({"time_stamps": present[0]["time_stamps"][:n]})
    for i in range(present[0]["time_series"].shape[1]):
        if filtered is not None:
            df[f"filtered_data_{i}"] = filtered["time_series"][:n, i]
        if raw is not None:
            df[f"raw_data_{i}"] = raw["time_series"][:n, i]
    if markers is not None and len(markers["time_stamps"]):
        times = df["time_stamps"].to_numpy()
        if markers["time_stamps"].min() < times[0] or markers["time_stamps"].max() > times[-1]:
            raise ValueError("Some markers fall outside the data streams")
        index = np.searchsorted(times, markers["time_stamps"])
        index = np.clip(index, 1, len(times) - 1)
        closest = np.where(times[index] - markers["time_stamps"] < markers["time_stamps"] - times[index - 1],
                           index, index - 1)
        stimuli = np.zeros(len(times), dtype=int)
        stimuli[closest] = 1
        df["stimuli"] = stimuli
    return df
