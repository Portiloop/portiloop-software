# Offline analysis of Portiloop recordings

Examples of how to analyse the recordings made with a Portiloop: the average EEG response to the stimulations, in the slow-oscillation band and in the spindle (sigma) band, for one night or for groups of participants.

| File | What it does |
|---|---|
| `slow_and_sigma_response.ipynb` | One night (one or several files): epochs around every stimulation, artifact rejection, average slow-band response and sigma envelope with their SEM, figures and CSV outputs. |
| `group_comparison.ipynb` | Several participants: per-participant responses during N2 and N3 sleep, values at the moment of the stimulation, comparison between groups (younger and older adults), group averages. |
| `portiloop_analysis/` | The functions used by both notebooks (see below). |
| `tests/` | Tests of the functions (`python -m pytest tests`). |

Both notebooks run without data: with no file in their configuration, they write and analyse synthetic recordings (in `demo_data/`). The synthetic signal is not real EEG; it only shows what the analysis does.

## Installation

```bash
pip install -r requirements.txt
jupyter lab            # then open a notebook from this folder
```

The notebooks import `portiloop_analysis` from this folder, so start Jupyter here (or add this folder to `PYTHONPATH`).

## The recordings

A recording made with the current Portiloop software is a folder of files sharing one name, ending with the start date and time:

- `<name>.csv`: one row per sample (250 Hz): `raw_ch1` ... `raw_ch8` in microvolts, `detection`, `stimulation` (1 when a sound was played, in Cond B, or would have been played, in the sham Cond A) and `stimulation_on` (1 while detection was running: 0 during the delay before stimulation and while an electrode, the reference or the bias was off);
- `<name>_metadata.json`: the settings of the recording (sampling frequency, detection channel, mains frequency, ...);
- `<name>_leadoff.json`: the electrodes that came off during the recording; their data are treated as missing.

Which electrode is on which `raw_ch` column depends on the montage: `default` is ECG1, ECG2, EOG1, EOG2, EMG, Cz, Fz, Fpz; `no_ecg` is Fpz, Cz, Pz, Fz, EOG1, EOG2, EMG1, EMG2. Check them against the wiring of your cap. Recordings made with older Portiloop software can be read with `load_legacy_recording`.

## The `portiloop_analysis` functions

| Module | Main functions |
|---|---|
| `recording` | `load_recording` (current format, several files of one night), `load_legacy_recording`, `load_hypnogram` (sleep scoring), `read_leadoff_periods`, `read_metadata`, `xdf_to_dataframe` (LabRecorder files, needs `pyxdf`) |
| `preprocessing` | `filter_band` (zero-phase filtering that never runs across missing data), `detect_jumps`, `bad_sections` (whole-night cleaning) |
| `epochs` | `EpochSettings` (all the analysis parameters), `event_locked_responses` (epochs, rejection, baselines), `EventLockedResponses` (`mean`, `sem`, `averages`, `rejection_summary`) |
| `measures` | `value_at`, `window_mean`, `window_rms`, `evoked_slow_wave_amplitude` (P900 minus N550), `compare_time_courses` (independent or paired t-tests with FDR correction), `fdr_bh` |
| `plotting` | `plot_responses`, `plot_groups` (box plots with one dot per participant), `plot_correlation` |
| `demo` | `make_demo_recording` |

A minimal analysis:

```python
from portiloop_analysis import load_recording, EpochSettings, event_locked_responses, plot_responses

recording = load_recording("night1/night1_2026-10-15_22-02-52_CONDB.csv", channel="Cz", montage="default")
responses = event_locked_responses(recording, EpochSettings(notch_hz=60, window_s=5))
print(responses.rejection_summary())
plot_responses(responses)
```

## What changed from the original scripts

These examples replace `portiloop_slow_and_sigma_analysis.ipynb`, `Portiloop_OA.ipynb` and `my_bsp_functions.py`.

- `slow_and_sigma_response.ipynb` gives exactly the same results as `portiloop_slow_and_sigma_analysis.ipynb` (same epochs kept, same averages to the last bit, checked on two real recordings), and also saves the reason why each epoch was rejected.
- `group_comparison.ipynb` follows `Portiloop_OA.ipynb`, with these corrections:
  - the 2.64/4.5 amplitude correction is gone: it compensated a wrong VREF in older versions of the Portiloop software;
  - "N2/N3 only" keeps N2 and N3; the original kept every stage except W and N1, so REM as well;
  - the off-head detection of `bad_section_detection_v4_corrected` never triggered (its counter was reset at every sample); it now removes runs of at least 5 minutes above the off-head level;
  - missing data are no longer bridged by linear interpolation before filtering: each stretch of signal is filtered on its own, and epochs touching missing data are rejected;
  - participants are listed in a table, with their group, instead of being assigned to groups from the letters of their identifiers (which put the OP participants in the wrong group);
  - results are saved as `.npz` files (readable without Python's pickle).
  On a synthetic recording in the old format, the corrected analysis kept the same 324 epochs as the original; the sigma and broadband averages were identical, and the slow-band average differed by at most 0.5 µV (out of 51 µV).
- From `my_bsp_functions.py`, the five copies of the epoch extraction (which differed only in their amplitude limit and printed messages) became one function with settings. The time-frequency epochs it built with MNE were not used by any notebook and were built at the wrong sample positions; they were dropped, together with the MNE, seaborn and statsmodels dependencies (the FDR correction is in `measures.fdr_bh`). Statistics, slow-wave amplitude (N550/P900), RMS and correlation plots are in `measures` and `plotting`.
