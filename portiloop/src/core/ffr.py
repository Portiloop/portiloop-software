"""
FFR paradigm (frequency-following response), the "FFR (fast)" pipeline: a fast acquisition (core/fast_capture.py)
during which the Portiloop plays the stimuli of the FFR protocol (FFR_Timeline_and_parameters.pdf), with a trigger
code in the recording at the onset of each one. There is no detection or stimulation in closed loop.

Stimuli (portiloop/sounds/ffr, prepared at the start of each run):
    - /da/: da_positive.wav (120 ms at 44.1 kHz); -da = -1 x da.
    - Noise tokens N1 to N4: noise1.wav to noise4.wav (120 ms), each scaled to +10 dB SNR against /da/ (RMS over the
      120 ms).
    - The 4 conditions and their trigger codes: + (/da/, 1), - (-da, 2), +N (/da/ + noise, 3), -N (-da + noise, 4).
      /da/ has the same level in all of them (the noise is added to it), the level calibrated at 80 dB SPL.
    - Resampled to 48 kHz (scipy.signal.resample_poly, 160/147), the rate of the Portiloop's sound output (dmix), so
      that ALSA does not resample them; the same sound in both ears.

Run (one recording, about 3.3 min; a session is 4 runs, so 4 recordings):
    - 1000 trials: 250 blocks of the 4 conditions, shuffled within each block (a condition can repeat back to back
      across blocks).
    - SOA (onset to onset) uniformly random between 195 and 205 ms, rounded to the frame (1/48 ms); for the noise
      conditions, a noise token drawn at random among N1 to N4.
    - The random parameters come from a seed drawn for each run, saved in the metadata. The trials are also listed in
      <name>_trials.csv.
    - LEAD_IN_S of silence before the first trial, TAIL_S after the last one.

The whole run is rendered as one sound, played by ffrplay (portiloop/fast_capture/ffrplay.c) through "softvol" (the
GUI's volume applies), so that the SOAs are exact to the frame. ffrplay sends the time of each onset (CLOCK_MONOTONIC,
from the playback position of ALSA) to adscapture, which writes the trigger code in the row of the conversion nearest
to it, in the "trigger" column of the CSV, and lists it in <name>_triggers.csv with the offset between the onset and
the time of that row. These times are the times at which the audio DMA reads the sound: the latency of the codec,
amplifier and earphones (and the delay of the ADS1299's filter) are constant, not included.

With the stimulation off ("Cond A"), the run is the same, with triggers, in silence.
"""

import csv
import hashlib
import json
import logging
import os
import secrets
import signal
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

STIMULI_DIR = Path(__file__).resolve().parents[2] / "sounds" / "ffr"
FFRPLAY = Path(__file__).resolve().parents[2] / "fast_capture" / "ffrplay"
SPEECH_FILE = "da_positive.wav"
NOISE_FILES = ("noise1.wav", "noise2.wav", "noise3.wav", "noise4.wav")

STIMULUS_RATE = 44100  # Hz, rate of the stimulus files
TOKEN_SAMPLES = 5292  # 120 ms at 44.1 kHz
AUDIO_RATE = 48000  # Hz, rate of the sound output (dmix in /etc/asound.conf): no resampling by ALSA
RESAMPLE_UP, RESAMPLE_DOWN = 160, 147  # 44100 * 160 / 147 = 48000
SNR_DB = 10.0  # /da/ against each noise token, RMS over the 120 ms

CONDITIONS = (("+", 1), ("-", 2), ("+N", 3), ("-N", 4))  # (condition, trigger code)
CONDITION_NAMES = {
    "+": "positive /da/",
    "-": "negative /da/",
    "+N": "positive /da/ + noise token",
    "-N": "negative /da/ + noise token",
}
N_BLOCKS = 250  # x 4 conditions: 1000 trials per run
SOA_MS = (195.0, 205.0)  # uniformly random, onset to onset
LEAD_IN_S = 1.0  # silence before the first trial (also lets ffrplay fit the clock of the sound output)
TAIL_S = 0.5  # silence after the last trial
TEST_SOUND_S = 5.0  # "Test Sound": a train of positive /da/ at 200 ms, to check the level

CHANNELS = 2  # the same sound in both ears
ALSA_DEVICE = "softvol"  # as the stimulators: the volume of the GUI (SoftMaster) applies
ALSA_LATENCY_US = 100000  # onsets are sent to adscapture about this long before they are played
PLAYER_RT_PRIORITY = 5  # below adscapture's reader (10), within LimitRTPRIO of simplegui.service
PLAYER_EXCLUDED_CPU = 3  # adscapture's reader (CAPTURE_CPU)
FULL_SCALE = 32767  # 16-bit samples: 1.0 in the stimulus files
TMP_DIR = "/dev/shm" if os.path.isdir("/dev/shm") else None  # the rendered run (about 38 MB) stays in memory


def ffr_status():
    """
    Returns:
        None if the FFR paradigm can run (besides the fast acquisition), otherwise the reason why not (str)
    """
    if not FFRPLAY.is_file():
        return f"{FFRPLAY} is not built (run portiloop/fast_capture/install.sh)"
    for name in (SPEECH_FILE,) + NOISE_FILES:
        if not (STIMULI_DIR / name).is_file():
            return f"{STIMULI_DIR / name} is missing"
    return None


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _read_token(path):
    from scipy.io import wavfile
    rate, x = wavfile.read(str(path))
    if rate != STIMULUS_RATE or x.ndim != 1 or len(x) != TOKEN_SAMPLES:
        raise ValueError(f"{path}: expected a mono sound of {TOKEN_SAMPLES} samples at {STIMULUS_RATE} Hz, "
                         f"got {x.shape} at {rate} Hz")
    if x.dtype.kind != "f":
        raise ValueError(f"{path}: expected floating-point samples, got {x.dtype}")
    return x.astype(np.float64)


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def make_conditions():
    """
    Prepares the stimuli of the 4 conditions from the files of STIMULI_DIR.

    Returns:
        stimuli: {code: [int16 array at AUDIO_RATE]}, one array for codes 1 and 2, one per noise token for 3 and 4
        info: dict for the metadata
    """
    from scipy.signal import resample_poly

    da = _read_token(STIMULI_DIR / SPEECH_FILE)
    noises = [_read_token(STIMULI_DIR / name) for name in NOISE_FILES]
    gains = [_rms(da) / _rms(n) / 10 ** (SNR_DB / 20) for n in noises]

    def to_output(x):
        return resample_poly(x, RESAMPLE_UP, RESAMPLE_DOWN)

    da_out = to_output(da)
    noises_out = [to_output(g * n) for g, n in zip(gains, noises)]
    float_stimuli = {
        1: [da_out],
        2: [-da_out],  # resampling is linear: exactly -1 x the positive /da/ (also once rounded)
        3: [da_out + n for n in noises_out],
        4: [-da_out + n for n in noises_out],
    }
    peak = max(float(np.max(np.abs(x))) for xs in float_stimuli.values() for x in xs)
    if peak >= 1.0:
        raise ValueError(f"The stimuli clip (peak {peak:.3f} of the full scale).")
    stimuli = {code: [np.round(x * FULL_SCALE).astype(np.int16) for x in xs] for code, xs in float_stimuli.items()}

    info = {
        "stimuli_folder": str(STIMULI_DIR),
        "stimulus_files_sha256": {name: _sha256(STIMULI_DIR / name) for name in (SPEECH_FILE,) + NOISE_FILES},
        "stimulus_rate": STIMULUS_RATE,
        "token_ms": 1000 * TOKEN_SAMPLES / STIMULUS_RATE,
        "da_rms": _rms(da),
        "snr_db": SNR_DB,
        "noise_gains": gains,
        "audio_rate": AUDIO_RATE,
        "resampling": f"scipy.signal.resample_poly(x, {RESAMPLE_UP}, {RESAMPLE_DOWN}) of each token",
        "token_frames": int(len(da_out)),
        "peak_full_scale": peak,
        "sample_format": f"16-bit, 1.0 in the files = {FULL_SCALE}",
    }
    return stimuli, info


def make_run(seed, n_blocks=N_BLOCKS):
    """
    The trials of a run, drawn from the seed: for each block, rng.permutation(4) of CONDITIONS; then the SOAs after
    each trial, rng.uniform(195, 205, n) ms (the last one is not used); then the noise tokens, rng.integers(1, 5, n)
    (used by the noise conditions only), with rng = numpy.random.default_rng(seed).

    Returns:
        list of dicts: trial (from 1), block (from 1), condition, code, noise (1 to 4, 0 without noise),
        onset_frame (in the rendered run, at AUDIO_RATE), soa_ms (to the next onset, None for the last trial)
    """
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(len(CONDITIONS)) for _ in range(n_blocks)])
    n = len(order)
    soa_frames = np.round(rng.uniform(SOA_MS[0], SOA_MS[1], n) * AUDIO_RATE / 1000).astype(np.int64)
    noise = rng.integers(1, len(NOISE_FILES) + 1, n)
    onsets = int(round(LEAD_IN_S * AUDIO_RATE)) + np.concatenate([[0], np.cumsum(soa_frames[:-1])])
    trials = []
    for i in range(n):
        condition, code = CONDITIONS[order[i]]
        trials.append({
            "trial": i + 1,
            "block": i // len(CONDITIONS) + 1,
            "condition": condition,
            "code": code,
            "noise": int(noise[i]) if code in (3, 4) else 0,
            "onset_frame": int(onsets[i]),
            "soa_ms": 1000 * int(soa_frames[i]) / AUDIO_RATE if i < n - 1 else None,
        })
    return trials


def render(trials, stimuli, sound=True):
    """
    The sound of the run (int16 frames, CHANNELS channels): silence, with the stimulus of each trial at its onset.
    Without sound, silence of the same length.
    """
    length = len(stimuli[1][0])
    nframes = trials[-1]["onset_frame"] + length + int(round(TAIL_S * AUDIO_RATE))
    mono = np.zeros(nframes, dtype=np.int16)
    if sound:
        for t in trials:
            x = stimuli[t["code"]][max(t["noise"], 1) - 1]
            mono[t["onset_frame"]:t["onset_frame"] + length] = x
    return np.repeat(mono[:, None], CHANNELS, axis=1)


def _player_args(audio_path, schedule_path, stats_path=None):
    args = [str(FFRPLAY), "-D", ALSA_DEVICE, "-r", str(AUDIO_RATE), "-c", str(CHANNELS), "-l", str(ALSA_LATENCY_US),
            "-p", str(PLAYER_RT_PRIORITY), "-u", str(PLAYER_EXCLUDED_CPU)]
    if stats_path is not None:
        args += ["-s", stats_path]
    return args + [audio_path, schedule_path]


def _log_stream(stream):
    for line in iter(stream.readline, b''):
        logger.info(line.decode(errors='replace').rstrip())


class FFRRun:
    """
    One run of the FFR paradigm, for start_fast_capture(paradigm=FFRRun): prepares the stimuli and the trials, and
    plays them during the capture.
    """
    trigger_column = "trigger"
    label = "FFR"  # in the messages
    metadata_key = "ffr"  # its section in _metadata.json; "<metadata_key>_playback" in _stats.json
    trial_fields = ["trial", "block", "condition", "code", "noise", "onset_frame", "soa_ms"]

    def __init__(self, config_dict, n_blocks=N_BLOCKS, seed=None):
        self.sound = bool(config_dict.get('stimulate', True))
        self.volume = config_dict.get('volume')
        self.seed = secrets.randbits(32) if seed is None else int(seed)
        self.stimuli, self.stimuli_info = make_conditions()
        self.trials = make_run(self.seed, n_blocks)
        audio = render(self.trials, self.stimuli, self.sound)
        self.nframes = len(audio)
        self.duration_s = self.nframes / AUDIO_RATE
        self._tmp = []
        self.audio_path = self._tmpfile("_ffr_audio.raw")
        audio.astype('<i2').tofile(self.audio_path)
        self.schedule_path = self._tmpfile("_ffr_schedule.txt")
        with open(self.schedule_path, "w") as f:
            for t in self.trials:
                f.write(f"{t['trial']} {t['code']} {t['onset_frame']}\n")
        self.player_stats_path = self._tmpfile("_ffr_player.json")
        self.player = None
        self._stopped = False
        logger.info("FFR run: %d trials, %.1f s, seed %d, %s.", len(self.trials), self.duration_s, self.seed,
                    "with sound" if self.sound else "without sound (Cond A)")

    def _tmpfile(self, suffix):
        fd, path = tempfile.mkstemp(suffix=suffix, dir=TMP_DIR)
        os.close(fd)
        self._tmp.append(path)
        return path

    def metadata(self):
        """The parameters of the run, for the metadata (JSON)."""
        counts = {c: sum(t["condition"] == c for t in self.trials) for c, _ in CONDITIONS}
        return dict(
            self.stimuli_info,
            paradigm="FFR_Timeline_and_parameters.pdf",
            seed=self.seed,
            random_draws="rng = numpy.random.default_rng(seed); for each block rng.permutation(4) of the conditions; "
                         "then rng.uniform(195, 205, n) ms (SOA after each trial); "
                         "then rng.integers(1, 5, n) (noise token, noise conditions only)",
            numpy_version=np.__version__,
            n_trials=len(self.trials),
            n_blocks=len(self.trials) // len(CONDITIONS),
            trials_per_condition=counts,
            trigger_codes={str(code): f"{c}: {CONDITION_NAMES[c]}" for c, code in CONDITIONS},
            soa_ms=list(SOA_MS),
            lead_in_s=LEAD_IN_S,
            tail_s=TAIL_S,
            duration_s=self.duration_s,
            sound=self.sound,
            volume=self.volume,
            alsa_device=ALSA_DEVICE,
            channels=f"{CHANNELS} (the same sound in both ears)",
            trigger_column=self.trigger_column,
            trials_file="<name>_trials.csv: the trials (design), with the row of their trigger after the recording",
            triggers_file="<name>_triggers.csv: the triggers as written by the capture program",
            onset_time="CLOCK_MONOTONIC time at which the audio DMA reads the first frame of the token (fitted on the "
                       "playback position of ALSA); the latency of the codec, amplifier and earphones is not included",
            trigger_row="the row whose conversion (DRDY) is nearest to the onset time; offset_us = onset time - time "
                        "of that row (within half a sample period); rows count from 0 at the first data row",
        )

    def write_trials(self, path, triggers=None):
        """
        Writes the trials (CSV). With the triggers of the capture program ({trial: dict}), adds their row, offset and
        whether they are in the CSV.
        """
        fields = list(self.trial_fields)
        if triggers is not None:
            fields += ["row", "offset_us", "in_csv"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for t in self.trials:
                row = dict(t, soa_ms="" if t["soa_ms"] is None else f"{t['soa_ms']:.6f}")
                if triggers is not None:
                    trig = triggers.get(t["trial"], {})
                    row.update(row=trig.get("row", ""), offset_us=trig.get("offset_us", ""),
                               in_csv=trig.get("in_csv", 0))
                w.writerow(row)

    def start(self, events_out):
        """Starts playing the run; the onsets are written to events_out (adscapture's stdin)."""
        args = _player_args(self.audio_path, self.schedule_path, self.player_stats_path)
        logger.info("Starting the %s player: %s", self.label, " ".join(args))
        self.player = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=events_out, stderr=subprocess.PIPE)
        threading.Thread(target=_log_stream, args=(self.player.stderr,), daemon=True).start()

    def ended(self):
        """Whether the player has exited (the whole run was played, or it was stopped or failed)."""
        return self.player is not None and self.player.poll() is not None

    def stop(self):
        """Stops the sound (the GUI's Stop)."""
        self._stopped = True
        if self.player is not None and self.player.poll() is None:
            self.player.send_signal(signal.SIGTERM)

    def finish(self, prefix, stats_path):
        """
        After the capture: adds the statistics of the player to the capture's statistics, and the rows of the
        triggers to the trials file. Raises an error if the player failed.
        """
        exitcode = None
        if self.player is not None:
            try:
                exitcode = self.player.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.player.kill()
                exitcode = self.player.wait()
        try:
            with open(self.player_stats_path) as f:
                player_stats = json.load(f)
        except (OSError, ValueError) as e:
            player_stats = {"error": f"no statistics from the player ({e})"}
        player_stats["exit_code"] = exitcode
        try:
            with open(stats_path) as f:
                stats = json.load(f)
            stats[f"{self.metadata_key}_playback"] = player_stats
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=4)
            triggers = stats.get("fast_capture", {}).get("triggers", {})
            logger.info("%s: %s onsets played (%s underruns); triggers: %s.", self.label,
                        player_stats.get("onsets_sent"), player_stats.get("underruns"), triggers)
            if player_stats.get("underruns") or triggers.get("late") or triggers.get("same_row"):
                logger.warning("%s: some triggers may be wrong (see %s).", self.label, stats_path)
        except (OSError, ValueError) as e:
            logger.error("Could not add the player's statistics to %s (%s).", stats_path, e)

        if prefix is not None:
            try:
                with open(f"{prefix}_triggers.csv") as f:
                    triggers = {int(r["trial"]): r for r in csv.DictReader(f)}
                self.write_trials(f"{prefix}_trials.csv", triggers)
            except (OSError, ValueError, KeyError) as e:
                logger.error("Could not add the rows of the triggers to the trials (%s).", e)

        if exitcode not in (0, None) and not self._stopped:
            raise RuntimeError(f"The {self.label} player failed (exit code {exitcode}).")

    def close(self):
        if self.player is not None and self.player.poll() is None:
            self.player.kill()
            self.player.wait()
        for path in self._tmp:
            try:
                os.remove(path)
            except OSError:
                pass
        self._tmp = []


def start_ffr_capture(config_dict, q_msg, q_display, pause_value=None):
    """The "FFR (fast)" pipeline: one run (see the module), arguments as start_fast_capture()."""
    from portiloop.src.core.fast_capture import start_fast_capture
    start_fast_capture(config_dict, q_msg, q_display, pause_value, paradigm=FFRRun)


def test_sound():
    """
    Plays TEST_SOUND_S of positive /da/ every 200 ms (to check the sound or its level), without waiting.
    """
    stimuli, _ = make_conditions()
    n = int(TEST_SOUND_S * 1000 / 200)
    trials = [{"code": 1, "noise": 0, "onset_frame": int(i * 0.2 * AUDIO_RATE)} for i in range(n)]
    audio = render(trials, stimuli)
    paths = []
    for suffix in ("_ffr_test.raw", "_ffr_test.txt"):
        fd, path = tempfile.mkstemp(suffix=suffix, dir=TMP_DIR)
        os.close(fd)
        paths.append(path)
    audio.astype('<i2').tofile(paths[0])  # the schedule stays empty: no onsets to send
    proc = subprocess.Popen(_player_args(*paths), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)

    def wait():
        _log_stream(proc.stderr)
        proc.wait()
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass

    threading.Thread(target=wait, daemon=True).start()
    return proc
