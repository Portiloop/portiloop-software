"""Tests of portiloop_analysis. Run from examples/processing with: python -m pytest tests"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portiloop_analysis import (EpochSettings, bad_sections, compare_time_courses, deduplicate, detect_jumps,  # noqa: E402
                                event_locked_responses, evoked_slow_wave_amplitude, fdr_bh, filter_band,
                                load_hypnogram, load_legacy_recording, load_recording, make_demo_recording,
                                read_leadoff_periods, value_at, window_mean)

FS = 250


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    return make_demo_recording(tmp_path_factory.mktemp("demo"), duration_min=6, seed=1)


# ------------------------------------------------------------------------------------------------ preprocessing

def test_filter_band_keeps_the_band_and_skips_gaps():
    t = np.arange(120 * FS) / FS
    eeg = 50 * np.sin(2 * np.pi * 13 * t) + 50 * np.sin(2 * np.pi * 1 * t)
    eeg[60 * FS:61 * FS] = np.nan
    sigma = filter_band(eeg, (11, 16), FS, edge_s=10)
    assert np.isnan(sigma[:10 * FS]).all() and np.isnan(sigma[-10 * FS:]).all()
    assert np.isnan(sigma[50 * FS + 1:71 * FS - 1]).all()  # the gap and 10 s on both sides
    middle = sigma[20 * FS:40 * FS]
    assert np.isclose(middle.std() * np.sqrt(2), 50, rtol=0.05)  # 13 Hz kept, 1 Hz removed


def test_filter_band_splits_at_breaks():
    eeg = np.random.default_rng(0).standard_normal(100 * FS)
    split = filter_band(eeg, (0.5, 30), FS, edge_s=5, breaks=[50 * FS])
    assert np.isnan(split[45 * FS:55 * FS]).all()
    assert np.isfinite(split[10 * FS:40 * FS]).all()


def test_detect_jumps_ignores_file_starts():
    eeg = np.zeros(100)
    eeg[30:] += 2000
    eeg[60:] += 2000
    jumps = detect_jumps(eeg, 1000, breaks=[60])
    assert np.flatnonzero(jumps).tolist() == [30]


def test_bad_sections():
    fs = 10
    eeg = np.zeros(10000)
    eeg[1000:5000] = 200000         # 400 s above the off-head level: excluded
    eeg[6000:6050] = 200000         # 5 s only: not off-head, but the jumps around it split the good signal
    eeg[9500] = 500                 # a jump
    good = bad_sections(eeg, fs, offhead_uv=105600, jump_uv=222, min_good_s=300)
    assert not good[1000:5000].any()
    assert not good[:1000].any()     # 100 s of good signal before the off-head run: too short
    assert not good[5000:6051].any()  # 100 s between the off-head run and the short burst: too short
    assert good[6051:9500].all()     # 345 s between the burst and the jump: long enough
    assert not good[9500:].any()     # 50 s after the jump: too short
    eeg2 = np.zeros(10000)
    assert bad_sections(eeg2, fs).all()


# --------------------------------------------------------------------------------------------------- recording

def test_leadoff_periods(tmp_path):
    csv = tmp_path / "rec_2026-01-01_00-00-00_X.csv"
    csv.write_text("raw_ch1,stimulation\n")
    events = [
        {"event": "disabled", "channels": [6], "sample": 100},
        {"event": "disabled", "channels": [2], "sample": 150},
        {"event": "reference_or_bias_off", "channels": [1, 2], "sample": 300},
        {"event": "reference_or_bias_recovered", "channels": [1, 2], "sample": 400},
    ]
    (tmp_path / "rec_2026-01-01_00-00-00_X_leadoff.json").write_text(json.dumps({"events": events}))
    assert read_leadoff_periods(csv, 1000, 6) == [(300, 400), (100, 1000)]
    assert read_leadoff_periods(csv, 1000, 2) == [(300, 400), (150, 1000)]
    assert read_leadoff_periods(csv, 1000, 3) == [(300, 400)]


def test_load_recording_merges_files_with_gap(tmp_path):
    def write(name, n, stim_at):
        stim = np.zeros(n, dtype=int)
        stim[stim_at] = 1
        df = pd.DataFrame({f"raw_ch{i}": np.full(n, float(i)) for i in range(1, 9)})
        df["detection"], df["stimulation"], df["stimulation_on"] = stim, stim, 1
        path = tmp_path / name
        df.to_csv(path, index=False)
        return path
    second = write("p_2026-01-01_00-00-10_A.csv", 500, 100)   # starts 10 s (2500 samples) after the first
    first = write("p_2026-01-01_00-00-00_A.csv", 1000, 10)
    rec = load_recording([second, first], channel="Cz", montage="default", verbose=False)
    assert rec.n_samples == 3000 and rec.breaks == [2500]
    assert np.isnan(rec.eeg[1000:2500]).all() and (rec.eeg[:1000] == 6).all()
    assert rec.events.tolist() == [10, 2600]
    with pytest.raises(ValueError):
        load_recording(first, channel="Oz", verbose=False)


def test_load_demo_and_leadoff(demo):
    rec = load_recording(demo, channel="Cz", verbose=False)
    assert rec.fs == 250 and rec.leadoff.sum() == 60 * 250
    assert np.isnan(rec.eeg[rec.leadoff]).all()


def test_legacy_layouts(tmp_path):
    n = 1000
    stim = np.zeros(n, dtype=int)
    stim[[100, 500]] = [2, 1]
    pd.DataFrame({"Fpz": 1.0, "Cz": np.arange(n, dtype=float), "Fz": 3.0, "Pz": 4.0, "e1": 0, "e2": 0,
                  "Stim": stim}).to_csv(tmp_path / "a.csv", index=False)
    rec = load_legacy_recording(tmp_path / "a.csv", channel="Cz")
    assert rec.eeg[5] == 5 and rec.events.tolist() == [100]
    assert load_legacy_recording(tmp_path / "a.csv", condition="sham").events.tolist() == [500]
    sorted_stim = np.zeros(n)
    sorted_stim[[200, 300, 400]] = [2, 4, 3]
    pd.DataFrame({"Raw Signal Ch2": np.ones(n), "sorted_stim": sorted_stim}).to_csv(tmp_path / "b.csv", index=False)
    rec = load_legacy_recording(tmp_path / "b.csv")
    assert rec.events.tolist() == [200, 300, 400]
    assert rec.stages[[200, 300, 400]].tolist() == ["N2", "R", "N3"]


def test_hypnogram(tmp_path):
    pd.DataFrame({"Annotation": ["W", "N2"], "Duration": [1.0, 2.0]}).to_csv(tmp_path / "h.csv", index=False)
    stages = load_hypnogram(tmp_path / "h.csv", 10, 40)
    assert stages[:10].tolist() == ["W"] * 10 and stages[10:30].tolist() == ["N2"] * 20 and stages[30:].tolist() == [""] * 10


# ------------------------------------------------------------------------------------------------------ epochs

def test_deduplicate_compares_with_previous_marker():
    assert deduplicate(np.array([0, 100, 400, 600, 2000]), 500).tolist() == [0, 2000]
    assert deduplicate(np.array([], dtype=int), 500).tolist() == []


def test_event_locked_responses_on_demo(demo):
    rec = load_recording(demo, channel="Cz", verbose=False)
    responses = event_locked_responses(rec, EpochSettings())
    statuses = set(responses.events["status"])
    assert "kept" in statuses and statuses <= {"kept", "edge", "leadoff", "jump", "missing", "artifact"}
    assert len(responses.events) == len(rec.events)
    slow = responses.mean("slow")
    # the evoked slow wave: negative near 0.55 s, positive near 0.9 s
    assert value_at(responses.time, slow, 0.55) < -20 and value_at(responses.time, slow, 0.9) > 10
    sigma = responses.mean("sigma")
    assert window_mean(responses.time, sigma, (1.2, 1.8), baseline=(-1.5, -1.0)) > 2
    assert responses.epochs["slow"].shape == (responses.n_kept, len(responses.time))
    assert {"time_s", "slow_mean_uV", "sigma_sem_uV"} <= set(responses.averages().columns)


def test_stage_selection_and_duplicates(demo):
    rec = load_recording(demo, channel="Cz", verbose=False)
    rec.stages = np.where(np.arange(rec.n_samples) < rec.n_samples // 2, "N2", "W").astype(object)
    rec.markers[rec.events[5] + 10] = 1  # a duplicate marker 40 ms after another
    responses = event_locked_responses(rec, EpochSettings(stages=("N2",), min_interval_s=1.0))
    status = responses.events.set_index("sample")["status"]
    assert (status[status.index >= rec.n_samples // 2] == "stage").all()
    assert status[rec.events[5] + 10] == "duplicate"


def test_settings_validation(demo):
    rec = load_recording(demo, channel="Cz", verbose=False)
    with pytest.raises(ValueError):
        event_locked_responses(rec, EpochSettings(sigma_baseline=(-10, -9)))
    with pytest.raises(ValueError):
        event_locked_responses(rec, EpochSettings(stages=("N2",)))  # no stages in the recording


# ----------------------------------------------------------------------------------------------------- measures

def test_fdr_bh_matches_reference():
    p = np.array([0.01, 0.04, 0.03, 0.005, np.nan])
    expected = np.array([0.02, 0.04, 0.04, 0.02, np.nan])  # statsmodels multipletests(method="fdr_bh")
    assert np.allclose(fdr_bh(p), expected, equal_nan=True)


def test_compare_time_courses_window():
    rng = np.random.default_rng(0)
    time = np.linspace(-1, 1, 21)
    a = rng.standard_normal((20, 21)) + 2 * (time > 0)
    b = rng.standard_normal((20, 21))
    p, p_fdr = compare_time_courses(a, b, time, window=(0, 1), alternative="greater")
    assert np.isnan(p[time < 0]).all() and (p_fdr[time > 0] < 0.05).all()


def test_evoked_slow_wave_amplitude():
    time = np.linspace(0, 2, 201)
    wave = np.where((time >= 0.5) & (time <= 0.6), -10.0, 0.0) + np.where((time >= 0.8) & (time <= 1.0), 5.0, 0.0)
    assert np.isclose(evoked_slow_wave_amplitude(time, wave), 15.0)


def test_compare_time_courses_paired():
    rng = np.random.default_rng(1)
    time = np.linspace(-1, 1, 11)
    person = 5 * rng.standard_normal((12, 1))          # large differences between participants
    a = person + rng.standard_normal((12, 11)) + 1.0    # stimulation night: +1 everywhere
    b = person + rng.standard_normal((12, 11))          # sham night of the same participants
    p_paired, _ = compare_time_courses(a, b, time, paired=True)
    p_independent, _ = compare_time_courses(a, b, time)
    assert np.median(p_paired) < 0.05 < np.median(p_independent)
    with pytest.raises(ValueError):
        compare_time_courses(a, b[:5], time, paired=True)


def test_markers_from_detection_column(tmp_path):
    n = 2000
    det = np.zeros(n, dtype=int)
    det[[500, 1500]] = 1
    df = pd.DataFrame({f"raw_ch{i}": np.zeros(n) for i in range(1, 9)})
    df["detection"], df["stimulation_on"] = det, 1
    path = tmp_path / "old_2026-10-12_23-00-00_CONDA.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_recording(path, verbose=False)
    assert load_recording(path, markers="detection", verbose=False).events.tolist() == [500, 1500]
