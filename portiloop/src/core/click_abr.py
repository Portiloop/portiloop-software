"""
Click ABR paradigm, the "Click ABR (fast)" pipeline: a fast acquisition (core/fast_capture.py) during which the
Portiloop plays a train of clicks in alternating polarity, with a trigger code in the recording at the onset of each
one. It works as the FFR paradigm (core/ffr.py), whose player (ffrplay) and triggers it uses.

Stimulus (made at the start of each run, no file):
    - A square burst of CLICK_FRAMES frames at the rate of the Portiloop's sound output (48 kHz): 5 frames = 104.2 us,
      the nearest to 100 us (the same burst at 44.1 kHz, burst.wav, would need resampling, which makes a square ring).
    - Amplitude CLICK_AMPLITUDE of the full scale (as burst.wav: 0.1), + or - (burst-.wav = -1 x burst.wav).
    - Trigger codes: 5 for +, 6 for -. The same sound in both ears.

Run (one recording, about 92 s):
    - N_TRIALS clicks (2000: 1000 +, 1000 -), in strictly alternating polarity starting with + (+ - + - ...).
    - SOA (onset to onset) uniformly random in SOA_MS (40 to 50 ms, mean 45 ms, about 22 Hz), rounded to the frame.
      The jitter keeps periodic activity of the Portiloop (the audio periods of 1024 frames, 21.3 ms, the CPU and
      the line noise) from being time-locked to the clicks: with a fixed SOA of 32 ms (1.5 periods), it made two
      artifacts in the average, at a fixed latency after the + and after the - clicks.
    - LEAD_IN_S of silence before the first click, TAIL_S after the last one. The SOAs come from a seed drawn for
      each run, saved in the metadata. The clicks are listed in <name>_trials.csv.

With the stimulation off ("Cond A"), the run is the same, with triggers, in silence.
"""

import logging
import os
import secrets
import subprocess
import tempfile
import threading

import numpy as np

from portiloop.src.core.ffr import (FFRRun, FFRPLAY, AUDIO_RATE, CHANNELS, FULL_SCALE, LEAD_IN_S, TAIL_S, TMP_DIR,
                                    ALSA_DEVICE, _player_args, _log_stream)

logger = logging.getLogger(__name__)

CLICK_FRAMES = 5  # at AUDIO_RATE: 104.2 us
CLICK_AMPLITUDE = 0.1  # of the full scale, as burst.wav
POLARITIES = (("+", 5), ("-", 6))  # (polarity, trigger code), alternating from the first one
POLARITY_NAMES = {"+": "positive click (condensation)", "-": "negative click (rarefaction)"}
N_TRIALS = 2000  # 1000 of each polarity
SOA_MS = (40.0, 50.0)  # uniformly random, onset to onset (mean 45 ms: about 22 Hz)
TEST_SOUND_S = 5.0  # "Test Sound": the clicks of a run (alternating, jittered SOA) for this long, to check the level


def click_abr_status():
    """
    Returns:
        None if the click ABR paradigm can run (besides the fast acquisition), otherwise the reason why not (str)
    """
    if not FFRPLAY.is_file():
        return f"{FFRPLAY} is not built (run portiloop/fast_capture/install.sh)"
    return None


def make_clicks():
    """
    Returns:
        stimuli: {code: int16 array at AUDIO_RATE}, the click of each polarity
    """
    click = np.full(CLICK_FRAMES, CLICK_AMPLITUDE)
    return {code: np.round(sign * click * FULL_SCALE).astype(np.int16)
            for sign, (_, code) in zip((1, -1), POLARITIES)}


def make_run(seed, n_trials=N_TRIALS):
    """
    The trials of a run, drawn from the seed: the SOAs after each trial, rng.uniform(40, 50, n) ms rounded to the frame
    (the last one is not used), with rng = numpy.random.default_rng(seed).

    Returns:
        list of dicts: trial (from 1), polarity, code, onset_frame (in the rendered run, at AUDIO_RATE), soa_ms (to
        the next onset, None for the last trial)
    """
    rng = np.random.default_rng(seed)
    soa_frames = np.round(rng.uniform(SOA_MS[0], SOA_MS[1], n_trials) * AUDIO_RATE / 1000).astype(np.int64)
    onsets = int(round(LEAD_IN_S * AUDIO_RATE)) + np.concatenate([[0], np.cumsum(soa_frames[:-1])])
    trials = []
    for i in range(n_trials):
        polarity, code = POLARITIES[i % len(POLARITIES)]
        trials.append({
            "trial": i + 1,
            "polarity": polarity,
            "code": code,
            "onset_frame": int(onsets[i]),
            "soa_ms": 1000 * int(soa_frames[i]) / AUDIO_RATE if i < n_trials - 1 else None,
        })
    return trials


def render(trials, stimuli, sound=True):
    """
    The sound of the run (int16 frames, CHANNELS channels): silence, with the click of each trial at its onset.
    Without sound, silence of the same length.
    """
    nframes = trials[-1]["onset_frame"] + CLICK_FRAMES + int(round(TAIL_S * AUDIO_RATE))
    mono = np.zeros(nframes, dtype=np.int16)
    if sound:
        for t in trials:
            mono[t["onset_frame"]:t["onset_frame"] + CLICK_FRAMES] = stimuli[t["code"]]
    return np.repeat(mono[:, None], CHANNELS, axis=1)


class ClickABRRun(FFRRun):
    """
    One run of the click ABR paradigm, for start_fast_capture(paradigm=ClickABRRun): the clicks are played and
    their triggers written as in FFRRun.
    """
    label = "Click ABR"
    metadata_key = "click_abr"
    trial_fields = ["trial", "polarity", "code", "onset_frame", "soa_ms"]

    def __init__(self, config_dict, n_trials=N_TRIALS, seed=None):
        self.seed = secrets.randbits(32) if seed is None else int(seed)
        self.sound = bool(config_dict.get('stimulate', True))
        self.volume = config_dict.get('volume')
        self.stimuli = make_clicks()
        self.trials = make_run(self.seed, n_trials)
        audio = render(self.trials, self.stimuli, self.sound)
        self.nframes = len(audio)
        self.duration_s = self.nframes / AUDIO_RATE
        self._tmp = []
        self.audio_path = self._tmpfile("_click_audio.raw")
        audio.astype('<i2').tofile(self.audio_path)
        self.schedule_path = self._tmpfile("_click_schedule.txt")
        with open(self.schedule_path, "w") as f:
            for t in self.trials:
                f.write(f"{t['trial']} {t['code']} {t['onset_frame']}\n")
        self.player_stats_path = self._tmpfile("_click_player.json")
        self.player = None
        self._stopped = False
        logger.info("Click ABR run: %d clicks, %.1f s, seed %d, %s.", len(self.trials), self.duration_s, self.seed,
                    "with sound" if self.sound else "without sound (Cond A)")

    def metadata(self):
        """The parameters of the run, for the metadata (JSON)."""
        counts = {p: sum(t["polarity"] == p for t in self.trials) for p, _ in POLARITIES}
        return dict(
            paradigm="click ABR: square clicks of about 100 us in alternating polarity, jittered SOA of 40 to 50 ms",
            click=f"square of {CLICK_FRAMES} frames at {AUDIO_RATE} Hz",
            click_us=1e6 * CLICK_FRAMES / AUDIO_RATE,
            click_amplitude_full_scale=CLICK_AMPLITUDE,
            click_samples={str(code): self.stimuli[code].tolist() for _, code in POLARITIES},
            audio_rate=AUDIO_RATE,
            sample_format=f"16-bit, 1.0 = {FULL_SCALE}",
            n_trials=len(self.trials),
            trials_per_polarity=counts,
            order="alternating, starting with +",
            trigger_codes={str(code): f"{p}: {POLARITY_NAMES[p]}" for p, code in POLARITIES},
            soa_ms=list(SOA_MS),
            soa="uniformly random in soa_ms (onset to onset), rounded to the frame; each trial's in <name>_trials.csv",
            mean_rate_hz=1000 / np.mean(SOA_MS),
            seed=self.seed,
            random_draws="rng = numpy.random.default_rng(seed); soa_ms = rng.uniform(40, 50, n_trials), rounded to "
                         "the frame at audio_rate (the last one is not used)",
            lead_in_s=LEAD_IN_S,
            tail_s=TAIL_S,
            duration_s=self.duration_s,
            sound=self.sound,
            volume=self.volume,
            alsa_device=ALSA_DEVICE,
            channels=f"{CHANNELS} (the same sound in both ears)",
            trigger_column=self.trigger_column,
            trials_file="<name>_trials.csv: the clicks (design), with the row of their trigger after the recording",
            triggers_file="<name>_triggers.csv: the triggers as written by the capture program",
            onset_time="CLOCK_MONOTONIC time at which the audio DMA reads the first frame of the click (fitted on the "
                       "playback position of ALSA); the latency of the codec, amplifier and earphones is not included",
            trigger_row="the row whose conversion (DRDY) is nearest to the onset time; offset_us = onset time - time "
                        "of that row (within half a sample period); rows count from 0 at the first data row",
        )


def start_click_abr_capture(config_dict, q_msg, q_display, pause_value=None):
    """The "Click ABR (fast)" pipeline: one run (see the module), arguments as start_fast_capture()."""
    from portiloop.src.core.fast_capture import start_fast_capture
    start_fast_capture(config_dict, q_msg, q_display, pause_value, paradigm=ClickABRRun)


def test_sound():
    """
    Plays TEST_SOUND_S of the clicks of a run (alternating polarity, jittered SOA), to check the sound or its level,
    without waiting.
    """
    n = int(TEST_SOUND_S * 1000 / np.mean(SOA_MS))
    lead_in = int(round(LEAD_IN_S * AUDIO_RATE))
    trials = [dict(t, onset_frame=t["onset_frame"] - lead_in) for t in make_run(secrets.randbits(32), n)]
    audio = render(trials, make_clicks())
    paths = []
    for suffix in ("_click_test.raw", "_click_test.txt"):
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
