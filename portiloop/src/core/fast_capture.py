"""
Fast acquisition (2 to 16 kSPS), the "Acquisition only (fast)" pipeline.

Python cannot read the ADS1299 faster than about 1 kSPS, so the capture runs in a C program
(portiloop/fast_capture/adscapture.c) that reads the ADS through the spimap kernel module. It writes the recording
itself, in the format of the "Acquisition only" pipeline (raw_ch1..raw_ch8 in uV, stimulation_on always 0), in parts
of at most PART_ROWS rows, and the statistics (_stats.json). It also sends a decimated copy of the signal (means of
blocks of samples, DISPLAY_RATE Hz) to this process, for the GUI display and LSL.

There is no lead-off detection during these recordings (its checks need register commands, which would interrupt the
reads); the electrodes OFF at the last impedance check are disabled as in the other pipelines.
"""

import json
import logging
import os
import queue
import select
import signal
import socket
import subprocess
import tempfile
import threading
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from portiloop.src.core.hardware.config_hardware import ADS_LSB, BACKEND_CONFIG, mod_config

logger = logging.getLogger(__name__)

FAST_RATES = (2000, 4000, 8000, 16000)
DISPLAY_RATE = 250  # Hz, decimated signal for the display and LSL
PART_ROWS = 28_800_000  # rows per CSV file, at most about 2.8 GB (FAT32 files are limited to 4 GB): 1 h at 8 kSPS
MIN_FREE_MB = 200  # the recording stops when the free space falls below this
CAPTURE_CPU = 3  # the reader polls DRDY on this CPU
CAPTURE_RT_PRIORITY = 10  # allowed by LimitRTPRIO in simplegui.service
STOP_TIMEOUT = 30  # s, for the capture program to write its last rows and exit

FAST_CAPTURE_DIR = Path(__file__).resolve().parents[2] / "fast_capture"
ADSCAPTURE = FAST_CAPTURE_DIR / "adscapture"
SPIMAP = Path("/dev/spimap")
SPI_POWER_CONTROL = Path("/sys/devices/platform/soc/1100c000.spi/power/control")
PORTILOOP_ID = f"{socket.gethostname()}-portiloop"


def fast_capture_status():
    """
    Returns:
        None if the fast acquisition can run on this Portiloop, otherwise the reason why not (str)
    """
    install = "run portiloop/fast_capture/install.sh"
    if not ADSCAPTURE.is_file():
        return f"{ADSCAPTURE} is not built ({install})"
    if not SPIMAP.exists():
        return f"{SPIMAP} is missing: the spimap module is not loaded ({install})"
    if not os.access(str(SPIMAP), os.R_OK | os.W_OK):
        return f"{SPIMAP} is not accessible ({install})"
    try:
        control = SPI_POWER_CONTROL.read_text().strip()
    except OSError as e:
        return f"cannot read {SPI_POWER_CONTROL} ({e})"
    if control != "on":
        return f"{SPI_POWER_CONTROL} is '{control}', not 'on' ({install})"
    return None


def _log_stderr(stream):
    for line in iter(stream.readline, b''):
        logger.info(line.decode(errors='replace').rstrip())


def _stop_ads():
    """
    Stops the ADS1299 through the kernel driver, after the capture program has exited. Its kernel transfers also put
    the SPI controller back in the state the kernel driver expects, if the program could not do it (killed).
    """
    from portiloop.src.core.hardware.backend import Backend
    from portiloop.src.core.utils import get_hardware_version
    backend = Backend(get_hardware_version())
    try:
        backend.stop_conversion()
    finally:
        backend.close()


def start_fast_capture(config_dict, q_msg, q_display, pause_value=None):
    """
    Runs a fast acquisition until 'STOP' is received on q_msg (or the duration is reached).
    Same arguments as start_capture() (without the pipeline classes): config_dict['frequency'] must be in FAST_RATES.
    """
    from portiloop.src.core.hardware.leds import Color, LEDs

    leds = None
    proc = None
    stats_tmp = None
    try:
        leds = LEDs()
        leds.led1(Color.CYAN if config_dict['stimulate'] else Color.PURPLE)

        status = fast_capture_status()
        if status is not None:
            raise RuntimeError(f"Fast acquisition unavailable: {status}")
        rate = int(config_dict['frequency'])
        if rate not in FAST_RATES:
            raise ValueError(f"The fast acquisition runs at {FAST_RATES} Hz, not {rate} Hz.")

        nb_channels = int(config_dict['nb_channels'])
        ads_config = mod_config(deepcopy(BACKEND_CONFIG), rate, config_dict['channel_states'])
        scale = 1e6 * config_dict['vref'] * ADS_LSB  # uV per code, as bin_to_microvolt()
        decimation = rate // DISPLAY_RATE
        duration = config_dict['duration'] if config_dict['duration'] > 0 else 0
        record = config_dict['record']

        config_dict = dict(config_dict,
                           python_clock=False,
                           leadoff_detection=False,
                           fast_capture={
                               "csv_rows_per_file": PART_ROWS,
                               "csv_files": "<name>.csv, then <name>_part2.csv, ... (each with a header)",
                               "decimated_rate": rate / decimation,
                               "decimated_signal": f"means of {decimation} samples (GUI display and LSL only)",
                           })

        if record:
            prefix = os.path.splitext(config_dict['filename'])[0]
            Path(prefix).parent.mkdir(parents=True, exist_ok=True)
            stats_path = f"{prefix}_stats.json"
            with open(f"{prefix}_metadata.json", "w") as f:
                json.dump(config_dict, f, indent=4)
        else:
            prefix = None
            fd, stats_tmp = tempfile.mkstemp(suffix="_stats.json")
            os.close(fd)
            stats_path = stats_tmp

        outlet = None
        if config_dict['lsl']:
            from pylsl import StreamInfo, StreamOutlet
            info = StreamInfo(name='Portiloop Raw Data', type='eeg', channel_count=nb_channels,
                              nominal_srate=rate / decimation, channel_format='float32', source_id=PORTILOOP_ID)
            outlet = StreamOutlet(info)

        args = [str(ADSCAPTURE), "-r", str(rate), "-c", str(nb_channels), "-k", repr(scale), "-n", str(PART_ROWS),
                "-d", str(decimation), "-t", str(duration), "-m", str(MIN_FREE_MB), "-s", stats_path,
                "-u", str(CAPTURE_CPU), "-p", str(CAPTURE_RT_PRIORITY)]
        if prefix is not None:
            args += ["-o", prefix]
        args += [f"{b:02x}" for b in ads_config[1:]]
        logger.info("Starting the fast acquisition at %s Hz: %s", rate, " ".join(args))
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=_log_stderr, args=(proc.stderr,), daemon=True).start()

        record_size = 4 * nb_channels
        out_fd = proc.stdout.fileno()
        pending = bytearray()
        nb_points = 0
        stop_sent = None  # time of the SIGTERM
        while True:
            try:
                while True:
                    if q_msg.get_nowait() == 'STOP' and not stop_sent:
                        proc.send_signal(signal.SIGTERM)
                        stop_sent = time.monotonic()
            except queue.Empty:
                pass
            if stop_sent and time.monotonic() - stop_sent > STOP_TIMEOUT:
                break  # killed below

            ready, _, _ = select.select([out_fd], [], [], 0.1)
            if not ready:
                continue
            chunk = os.read(out_fd, 1 << 16)
            if not chunk:  # the capture program has exited
                break
            pending += chunk
            n = len(pending) // record_size
            if n == 0:
                continue
            points = np.frombuffer(bytes(pending[:n * record_size]), dtype='<f4').reshape(n, nb_channels)
            del pending[:n * record_size]
            if outlet is not None:
                outlet.push_chunk(points.tolist())
            if q_display is not None:
                for point in points.tolist():
                    q_display.put([nb_points / DISPLAY_RATE, [point], [point]])
                    nb_points += 1

        try:
            exitcode = proc.wait(timeout=STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.error("The capture program did not exit: killing it.")
            proc.kill()
            exitcode = proc.wait()

        try:
            with open(stats_path) as f:
                stats = json.load(f)
            fast = stats.get("fast_capture", {})
            logger.info("Captured %s samples in %.1f s (%.3f Hz as per the OS clock): %s.", fast.get("rows"),
                        stats.get("os_time", 0), stats.get("os_sampling_frequency", 0), fast.get("stop_reason"))
            if stats.get("nb_lost_samples"):
                logger.warning("%s lost samples (repeating the previous sample).", stats["nb_lost_samples"])
        except (OSError, ValueError) as e:
            logger.error("No statistics from the capture program (%s).", e)

        if exitcode != 0:
            raise RuntimeError(f"The capture program failed (exit code {exitcode}).")

    except Exception:
        logger.exception("start_fast_capture crashed.")
        raise

    finally:
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            try:
                _stop_ads()
            except Exception:
                logger.exception("Could not stop the ADS1299.")
        if stats_tmp is not None:
            try:
                os.remove(stats_tmp)
            except OSError:
                pass
        if leds is not None:
            leds.led1(Color.YELLOW)
            leds.close()
