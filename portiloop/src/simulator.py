#!/usr/bin/env python3
"""
Portiloop Command-Line Simulator.

Replays recorded Portiloop CSV data through Portiloop processors, detectors,
and stimulators faster than real time (as fast as possible), generating a new
CSV with updated detection and stimulation markers, and optionally streaming via LSL.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import types
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("portiloop-simulator")

# ---------------------------------------------------------------------------
# Resolve repository root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Hardware & Platform Fallbacks (for offline simulation on any machine)
# ---------------------------------------------------------------------------
# 1. Mock 'periphery' if not installed (used by backend.py on Coral)
if 'periphery' not in sys.modules:
    try:
        import periphery
    except ImportError:
        dummy_periphery = types.ModuleType('periphery')
        class DummyGPIO:
            def __init__(self, *args, **kwargs): pass
            def close(self): pass
        dummy_periphery.GPIO = DummyGPIO
        sys.modules['periphery'] = dummy_periphery

# 2. Mock 'spidev' if not installed
if 'spidev' not in sys.modules:
    try:
        import spidev
    except ImportError:
        dummy_spidev = types.ModuleType('spidev')
        class DummySpiDev:
            def __init__(self, *args, **kwargs): pass
            def open(self, *args, **kwargs): pass
            def close(self): pass
        dummy_spidev.SpiDev = DummySpiDev
        sys.modules['spidev'] = dummy_spidev

# 3. Mock 'alsaaudio' if not installed or to avoid physical sound playback
if 'alsaaudio' not in sys.modules:
    try:
        import alsaaudio
    except ImportError:
        dummy_alsa = types.ModuleType('alsaaudio')
        class DummyPCM:
            def __init__(self, *args, **kwargs): pass
            def write(self, data): pass
            def close(self): pass
        class ALSAAudioError(Exception): pass
        dummy_alsa.PCM = DummyPCM
        dummy_alsa.ALSAAudioError = ALSAAudioError
        dummy_alsa.PCM_FORMAT_U8 = 1
        dummy_alsa.PCM_FORMAT_S16_LE = 2
        dummy_alsa.PCM_FORMAT_S24_3LE = 3
        dummy_alsa.PCM_FORMAT_S32_LE = 4
        sys.modules['alsaaudio'] = dummy_alsa

# 4. Fallback from tensorflow to tflite_runtime or placeholder dummy
if 'tensorflow' not in sys.modules:
    try:
        import tensorflow
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            dummy_tf = types.ModuleType('tensorflow')
            dummy_tf.lite = types.ModuleType('lite')
            dummy_tf.lite.Interpreter = tflite.Interpreter
            sys.modules['tensorflow'] = dummy_tf
        except ImportError:
            dummy_tf = types.ModuleType('tensorflow')
            dummy_tf.lite = types.ModuleType('lite')
            class UnavailableInterpreter:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("TensorFlow or tflite_runtime is required to run SleepSpindleRealTimeDetector.")
            dummy_tf.lite.Interpreter = UnavailableInterpreter
            sys.modules['tensorflow'] = dummy_tf

# 5. Mock backend to avoid failure when hardware is not physically present
try:
    import portiloop.src.core.hardware.backend as hb
    class MockBackend:
        def __init__(self, *args, **kwargs): pass
        def get_nb_channels(self): return 8
        def close(self): pass
    hb.Backend = MockBackend
except Exception:
    pass

# 6. Patch Path.exists during constants import if default folders are elsewhere
_orig_exists = Path.exists
def _patched_exists(self):
    if 'portiloop-software' in str(self):
        return True
    return _orig_exists(self)

Path.exists = _patched_exists
try:
    import portiloop.src.core.constants as constants
finally:
    Path.exists = _orig_exists

# Ensure constants points to local paths if needed
if not constants.SIGNAL_SAMPLES_FOLDER.exists():
    constants.SIGNAL_SAMPLES_FOLDER = REPO_ROOT / 'portiloop' / 'signal_samples'
if not constants.SOUNDS_FOLDER.exists():
    constants.SOUNDS_FOLDER = REPO_ROOT / 'portiloop' / 'sounds'
if not constants.DEFAULT_MODEL_PATH.exists():
    constants.DEFAULT_MODEL_PATH = REPO_ROOT / 'portiloop' / 'models' / 'portiloop_model_quant.tflite'

# Import base utilities
from portiloop.src.core.constants import DEFAULT_CONFIG_DICT, NB_CHANNELS
from portiloop.src.core.utils import Dummy

# Pipeline name alias mappings
PIPELINE_ALIASES = {
    "spindles": "Sleep spindles",
    "spindle": "Sleep spindles",
    "sleep_spindles": "Sleep spindles",
    "sleep spindles": "Sleep spindles",
    "so": "Sleep slow oscillations",
    "slow_oscillations": "Sleep slow oscillations",
    "sleep slow oscillations": "Sleep slow oscillations",
    "upstate": "Sleep slow oscillations (upstate)",
    "so_upstate": "Sleep slow oscillations (upstate)",
    "sleep slow oscillations (upstate)": "Sleep slow oscillations (upstate)",
    "none": "Acquisition only",
    "acquisition only": "Acquisition only",
    "acquisition": "Acquisition only",
}

KNOWN_PIPELINES = [
    "Sleep spindles",
    "Sleep slow oscillations",
    "Sleep slow oscillations (upstate)",
    "Acquisition only",
]


# ---------------------------------------------------------------------------
# Virtual Simulated Clock
# ---------------------------------------------------------------------------
class SimulatedClock:
    """Provides virtual simulated time matching recording timestamps."""
    def __init__(self, initial_time: float = 0.0):
        self._time = initial_time

    def time(self) -> float:
        return self._time

    def set_time(self, t: float):
        self._time = t

    def advance(self, dt: float):
        self._time += dt


# ---------------------------------------------------------------------------
# Simulated CSV Recorder (matches Portiloop CSV format)
# ---------------------------------------------------------------------------
class SimulatedCSVRecorder:
    """
    Buffers and writes simulation output in the exact same format as CSVRecorder,
    optimized for fast streaming batch writes without real-time delays.
    """
    def __init__(self, filepath: Path, nb_channels: int, write_batch_size: int = 5000):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.nb_channels = nb_channels
        self.write_batch_size = write_batch_size

        self.file = open(self.filepath, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        self.detection_buffer: List[float] = []
        self.stimulation_buffer: List[float] = []
        self.writing_buffer: List[List[Any]] = []

        # Write header
        header = [f"raw_ch{i + 1}" for i in range(self.nb_channels)]
        header += [f"filtered_ch{i + 1}" for i in range(self.nb_channels)]
        header += ["detection", "stimulation", "stimulation_on", "timestamp"]
        self.writer.writerow(header)

    def append_detection_signal_buffer(self, buffer: List[float]):
        self.detection_buffer.extend(buffer)

    def append_stimulation_signal_buffer(self, buffer: List[float]):
        self.stimulation_buffer.extend(buffer)

    def add_row(self, raw_point: List[float], filtered_point: List[float], stim_on: int, timestamp: float):
        det_val = self.detection_buffer.pop(0) if self.detection_buffer else 0
        stim_val = self.stimulation_buffer.pop(0) if self.stimulation_buffer else 0

        row = list(raw_point) + list(filtered_point) + [det_val, stim_val, stim_on, f"{timestamp:.6f}"]
        self.writing_buffer.append(row)

        if len(self.writing_buffer) >= self.write_batch_size:
            self.flush()

    def flush(self):
        if self.writing_buffer:
            self.writer.writerows(self.writing_buffer)
            self.writing_buffer.clear()
            self.file.flush()

    def close(self):
        self.flush()
        if self.file and not self.file.closed:
            self.file.close()


# ---------------------------------------------------------------------------
# Simulator Core
# ---------------------------------------------------------------------------
class PortiloopSimulator:
    def __init__(
        self,
        input_csv: str,
        output_csv: Optional[str] = None,
        pipeline_name: str = "Sleep spindles",
        metadata_file: Optional[str] = None,
        channel_detection: Optional[int] = None,
        frequency: Optional[float] = None,
        stimulate: bool = True,
        threshold: Optional[float] = None,
        power_line: Optional[int] = None,
        delay_mode: Optional[str] = None,
        min_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        inter_stim_delay: Optional[float] = None,
        lsl: bool = False,
        quiet: bool = False,
    ):
        self.input_path = Path(input_csv).resolve()
        if not self.input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        if output_csv is not None:
            self.output_path = Path(output_csv).resolve()
        else:
            self.output_path = self.input_path.parent / f"{self.input_path.stem}_simulated.csv"

        # Load metadata if available
        self.config = self._build_config(
            metadata_file=metadata_file,
            channel_detection=channel_detection,
            frequency=frequency,
            stimulate=stimulate,
            threshold=threshold,
            power_line=power_line,
            delay_mode=delay_mode,
            min_delay=min_delay,
            max_delay=max_delay,
            inter_stim_delay=inter_stim_delay,
            lsl=lsl,
        )

        # Resolve pipeline (use metadata pipeline if CLI argument was default and metadata specifies one)
        pipeline_from_meta = self.config.get("pipeline_name") or self.config.get("pipeline")
        if pipeline_name == "Sleep spindles" and pipeline_from_meta:
            pipeline_name = pipeline_from_meta

        key_lookup = str(pipeline_name).lower().strip()
        self.pipeline_canonical_name = PIPELINE_ALIASES.get(key_lookup, pipeline_name)

        self.lsl = lsl
        self.quiet = quiet
        self.sim_clock = SimulatedClock(0.0)

    def _build_config(
        self,
        metadata_file: Optional[str],
        channel_detection: Optional[int],
        frequency: Optional[float],
        stimulate: bool,
        threshold: Optional[float],
        power_line: Optional[int],
        delay_mode: Optional[str],
        min_delay: Optional[float],
        max_delay: Optional[float],
        inter_stim_delay: Optional[float],
        lsl: bool,
    ) -> Dict[str, Any]:
        config = dict(DEFAULT_CONFIG_DICT)

        # Look for companion _metadata.json
        meta_path = None
        if metadata_file is not None:
            meta_path = Path(metadata_file)
        else:
            candidate = self.input_path.parent / f"{self.input_path.stem}_metadata.json"
            if candidate.is_file():
                meta_path = candidate

        if meta_path and meta_path.is_file():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_dict = json.load(f)
                config.update(meta_dict)
                logger.info("Loaded configuration from metadata: %s", meta_path)
            except Exception as e:
                logger.warning("Could not read metadata file %s: %s", meta_path, e)

        # CLI overrides
        if frequency is not None:
            config["frequency"] = float(frequency)
        if channel_detection is not None:
            config["channel_detection"] = int(channel_detection)
        config["stimulate"] = bool(stimulate)
        if threshold is not None:
            config["threshold"] = float(threshold)
        if power_line is not None:
            config.setdefault("filter_settings", {})["power_line"] = int(power_line)
        if delay_mode is not None:
            config["stim_delay_mode"] = delay_mode
        if min_delay is not None:
            config["min_delay"] = float(min_delay) / 1000.0 if float(min_delay) > 1.0 else float(min_delay)
        if max_delay is not None:
            config["max_delay"] = float(max_delay) / 1000.0 if float(max_delay) > 1.0 else float(max_delay)
        if inter_stim_delay is not None:
            config["inter_stim_delay"] = float(inter_stim_delay) / 1000.0 if float(inter_stim_delay) > 1.0 else float(inter_stim_delay)
        config["lsl"] = bool(lsl)
        config["filter"] = True
        config["detect"] = True
        config["record"] = True

        return config

    def _get_pipeline(self):
        from portiloop.src.custom.custom_pipelines import PIPELINES
        if self.pipeline_canonical_name not in PIPELINES:
            valid_keys = list(PIPELINES.keys()) + list(PIPELINE_ALIASES.keys())
            raise ValueError(f"Unknown pipeline: '{self.pipeline_canonical_name}'. Valid choices: {valid_keys}")
        return PIPELINES[self.pipeline_canonical_name]

    def _parse_input_file(self) -> Tuple[List[str], Dict[int, int], Optional[int], bool]:
        """
        Inspects CSV header to find raw channel mapping and timestamp column.
        Returns:
            headers: list of column names
            channel_col_map: dict mapping channel_index (0..nb_channels-1) -> csv_column_index
            timestamp_col_idx: column index of timestamp, if present
            is_header: whether the first row was a text header
        """
        with open(self.input_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            first_row = next(reader, None)

        if not first_row:
            raise ValueError(f"Input CSV is empty: {self.input_path}")

        # Check if first row is a text header
        is_header = False
        try:
            float(first_row[0].strip())
        except ValueError:
            is_header = True

        channel_col_map: Dict[int, int] = {}
        timestamp_col_idx: Optional[int] = None
        nb_channels = self.config["nb_channels"]
        det_channel = self.config["channel_detection"]

        if is_header:
            headers = [h.strip() for h in first_row]
            for idx, col in enumerate(headers):
                lower_col = col.lower()
                if lower_col.startswith("raw_ch"):
                    try:
                        ch_num = int(lower_col[6:])
                        if 1 <= ch_num <= nb_channels:
                            channel_col_map[ch_num - 1] = idx
                    except ValueError:
                        pass
                elif lower_col == "timestamp":
                    timestamp_col_idx = idx

            # If no 'raw_ch' found, look for 'ch1', 'ch2'
            if not channel_col_map:
                for idx, col in enumerate(headers):
                    lower_col = col.lower()
                    if lower_col.startswith("ch") and lower_col[2:].isdigit():
                        try:
                            ch_num = int(lower_col[2:])
                            if 1 <= ch_num <= nb_channels:
                                channel_col_map[ch_num - 1] = idx
                        except ValueError:
                            pass

            # If still nothing matched, default first column to detection channel
            if not channel_col_map:
                channel_col_map[det_channel - 1] = 0
        else:
            headers = [f"col_{i}" for i in range(len(first_row))]
            num_cols = len(first_row)
            if num_cols < nb_channels:
                # E.g. test samples with 1 raw channel + metadata columns:
                # Column 0 is the raw signal, assigned to the detection channel
                channel_col_map[det_channel - 1] = 0
            else:
                for i in range(nb_channels):
                    channel_col_map[i] = i

        return headers, channel_col_map, timestamp_col_idx, is_header

    def run(self):
        """Executes the simulation as fast as possible."""
        t_start_real = time.perf_counter()

        pipeline = self._get_pipeline()
        headers, channel_col_map, timestamp_col_idx, is_header = self._parse_input_file()
        nb_channels = self.config["nb_channels"]
        det_channel = self.config["channel_detection"]
        frequency = self.config["frequency"]
        dt = 1.0 / frequency if frequency > 0 else 0.004

        if not self.quiet:
            logger.info("==================================================")
            logger.info("Portiloop Simulator")
            logger.info("Input CSV:           %s", self.input_path)
            logger.info("Output CSV:          %s", self.output_path)
            logger.info("Pipeline:            %s", self.pipeline_canonical_name)
            logger.info("Detection Channel:   %d", det_channel)
            cond_str = "Condition B (STIM)" if self.config["stimulate"] else "Condition A (NOSTIM)"
            logger.info("Condition:           %s", cond_str)
            logger.info("LSL Streaming:       %s", self.lsl)
            logger.info("Mapped channels:     %s", {f"ch{k+1}": f"col_{v}" for k, v in channel_col_map.items()})
            logger.info("==================================================")

        # -------------------------------------------------------------------
        # Initialize LSL Streamer if requested
        # -------------------------------------------------------------------
        if self.lsl:
            from portiloop.src.core.output import LSLStreamer
            streams = {'filtered': True, 'markers': True}
            lsl_streamer = LSLStreamer(streams, nb_channels, frequency, id="portiloop-sim")
        else:
            lsl_streamer = Dummy()

        # -------------------------------------------------------------------
        # Initialize Output Recorder
        # -------------------------------------------------------------------
        csv_recorder = SimulatedCSVRecorder(self.output_path, nb_channels)

        # -------------------------------------------------------------------
        # Monkey-patch time.time and sound playback for the simulation loop
        # -------------------------------------------------------------------
        import portiloop.src.custom.custom_stimulators as cs_module
        if not hasattr(cs_module, 'alsaaudio') or cs_module.alsaaudio is None:
            cs_module.alsaaudio = sys.modules.get('alsaaudio')

        orig_time = time.time
        orig_t_sound = getattr(cs_module.SleepSpindleRealTimeStimulator, "_t_sound", None)
        orig_play_sound = getattr(cs_module.SleepSpindleRealTimeStimulator, "play_sound", None)

        # Disable sound card playback to run at maximum computational speed
        cs_module.SleepSpindleRealTimeStimulator._t_sound = lambda self: None
        cs_module.SleepSpindleRealTimeStimulator.play_sound = lambda self: None

        time.time = self.sim_clock.time

        processor = None
        detector = None
        stimulator = None

        total_samples = 0
        total_detections = 0

        try:
            # Instantiate pipeline components
            processor_cls = pipeline.get("processor")
            detector_cls = pipeline.get("detector")
            stimulator_cls = pipeline.get("stimulator")

            if processor_cls is not None:
                processor = processor_cls(self.config, lsl_streamer, csv_recorder)

            if detector_cls is not None:
                detector = detector_cls(self.config, lsl_streamer, csv_recorder)

            if stimulator_cls is not None and self.config["stimulate"]:
                stimulator = stimulator_cls(self.config, lsl_streamer, csv_recorder)

            # Replay data loop
            with open(self.input_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                if is_header:
                    next(reader, None)  # skip header row

                last_progress_t = time.perf_counter()
                stim_on_flag = 1 if self.config["stimulate"] else 0

                for row in reader:
                    if not row:
                        continue

                    # Extract timestamp
                    if timestamp_col_idx is not None and len(row) > timestamp_col_idx:
                        try:
                            sim_t = float(row[timestamp_col_idx].strip())
                        except ValueError:
                            sim_t = total_samples * dt
                    else:
                        sim_t = total_samples * dt

                    self.sim_clock.set_time(sim_t)

                    # Extract raw channels
                    raw_point = np.zeros(nb_channels, dtype=np.float64)
                    for ch_idx, col_idx in channel_col_map.items():
                        if col_idx < len(row):
                            try:
                                raw_point[ch_idx] = float(row[col_idx].strip())
                            except ValueError:
                                pass

                    # 1. Processing / Filtering
                    raw_points_batch = np.array([raw_point])
                    if processor is not None:
                        filtered_points_batch = processor.filter(raw_points_batch.copy())
                    else:
                        filtered_points_batch = raw_points_batch.copy()

                    filtered_list = filtered_points_batch.tolist()
                    raw_list = raw_points_batch.tolist()

                    # Push to LSL
                    if self.lsl:
                        lsl_streamer.push_raw(raw_list[0])
                        if processor is not None:
                            lsl_streamer.push_filtered(filtered_list[0])

                    # 2. Detection
                    det_signal = None
                    if detector is not None:
                        det_signal = detector.detect(filtered_list)
                        # Check detection result count
                        if isinstance(det_signal, tuple) and len(det_signal) > 0:
                            res_list = det_signal[0]
                            if any(res_list):
                                total_detections += 1

                    # 3. Stimulation
                    if stimulator is not None and det_signal is not None:
                        stimulator.stimulate(det_signal)

                    # 4. Record to CSV
                    csv_recorder.add_row(
                        raw_point=raw_list[0],
                        filtered_point=filtered_list[0],
                        stim_on=stim_on_flag,
                        timestamp=sim_t
                    )

                    total_samples += 1

                    # Progress display every 1.0 wall-clock seconds
                    if not self.quiet:
                        now_real = time.perf_counter()
                        if now_real - last_progress_t >= 1.0:
                            last_progress_t = now_real
                            elapsed_real = now_real - t_start_real
                            sim_sec = total_samples * dt
                            speedup = (sim_sec / elapsed_real) if elapsed_real > 0 else 0
                            print(
                                f"\r[Replay] Samples: {total_samples:,} | "
                                f"Simulated time: {sim_sec:.1f}s | "
                                f"Speedup: {speedup:.1f}x real-time | "
                                f"Detections: {total_detections}",
                                end="",
                                flush=True
                            )

        finally:
            # Restore original time function and stimulator methods
            time.time = orig_time
            if orig_t_sound is not None:
                cs_module.SleepSpindleRealTimeStimulator._t_sound = orig_t_sound
            if orig_play_sound is not None:
                cs_module.SleepSpindleRealTimeStimulator.play_sound = orig_play_sound

            csv_recorder.close()

            # Dump metadata JSON for the simulated recording
            try:
                out_meta_path = self.output_path.parent / f"{self.output_path.stem}_metadata.json"
                out_config = dict(self.config)
                out_config['filename'] = str(self.output_path)
                out_config['pipeline_name'] = self.pipeline_canonical_name
                out_config['pipeline'] = self.pipeline_canonical_name
                with open(out_meta_path, "w") as f:
                    json.dump(out_config, f, indent=4)
                meta_saved = True
            except Exception as e:
                meta_saved = False
                logger.warning("Could not write simulated metadata: %s", e)

        t_end_real = time.perf_counter()
        real_duration = max(t_end_real - t_start_real, 1e-6)
        simulated_duration = total_samples * dt
        overall_speedup = simulated_duration / real_duration

        if not self.quiet:
            print()  # newline after progress
            logger.info("==================================================")
            logger.info("Simulation Complete!")
            logger.info("Total samples processed: %d", total_samples)
            logger.info("Simulated recording time: %.2f seconds (%.2f minutes)", simulated_duration, simulated_duration / 60)
            logger.info("Real computation time:    %.2f seconds", real_duration)
            logger.info("Overall replay speedup:   %.1fx real-time", overall_speedup)
            logger.info("Total detections:         %d", total_detections)
            logger.info("Output written to:        %s", self.output_path)
            if meta_saved:
                logger.info("Metadata written to:      %s", out_meta_path)
            logger.info("==================================================")


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Portiloop Command-Line Simulator: Fast offline replay of recorded CSV data through detectors and stimulators.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-i", "--input", required=True, help="Path to input CSV file recorded by Portiloop.")
    parser.add_argument("-o", "--output", default=None, help="Path for the generated output CSV. Defaults to <input>_simulated.csv.")
    parser.add_argument(
        "-p", "--pipeline",
        default="Sleep spindles",
        choices=KNOWN_PIPELINES + list(PIPELINE_ALIASES.keys()),
        help="Pipeline to run: 'Sleep spindles', 'Sleep slow oscillations', 'Sleep slow oscillations (upstate)', or 'Acquisition only'."
    )
    parser.add_argument("-c", "--channel", type=int, default=None, help="Detection channel (1-8). Defaults to metadata or 6.")
    parser.add_argument("-f", "--frequency", type=float, default=None, help="Sampling frequency in Hz. Defaults to metadata or 250.")
    parser.add_argument("-m", "--metadata", default=None, help="Path to _metadata.json. If omitted, automatically checks for companion JSON.")
    parser.add_argument("--threshold", type=float, default=None, help="Detection threshold (e.g. 0.75 for spindles).")
    parser.add_argument("--power-line", type=int, choices=[50, 60], default=None, help="Notch filter frequency (50 or 60 Hz).")
    parser.add_argument(
        "-C", "--condition",
        choices=["A", "B", "a", "b"],
        default=None,
        help="Experimental condition: 'B' for STIM, 'A' for NOSTIM (overrides --stimulate/--no-stimulate)."
    )
    parser.add_argument("--stimulate", dest="stimulate", action="store_true", default=None, help="Enable stimulation (Condition B).")
    parser.add_argument("--no-stimulate", dest="stimulate", action="store_false", default=None, help="Disable stimulation (Condition A).")
    parser.add_argument("--delay-mode", choices=["Time", "Peak", "Valley"], default=None, help="Stimulation delay mode.")
    parser.add_argument("--min-delay", type=float, default=None, help="Minimum stimulation delay in ms.")
    parser.add_argument("--max-delay", type=float, default=None, help="Maximum stimulation delay in ms.")
    parser.add_argument("--inter-stim-delay", type=float, default=None, help="Inter-stimulation delay in ms.")
    parser.add_argument("--lsl", action="store_true", default=False, help="Enable streaming raw, filtered, and markers over LSL.")
    parser.add_argument("-q", "--quiet", action="store_true", default=False, help="Suppress real-time progress updates.")

    args = parser.parse_args()

    stimulate_flag = True
    if args.condition is not None:
        stimulate_flag = (args.condition.upper() == "B")
    elif args.stimulate is not None:
        stimulate_flag = args.stimulate

    simulator = PortiloopSimulator(
        input_csv=args.input,
        output_csv=args.output,
        pipeline_name=args.pipeline,
        metadata_file=args.metadata,
        channel_detection=args.channel,
        frequency=args.frequency,
        stimulate=stimulate_flag,
        threshold=args.threshold,
        power_line=args.power_line,
        delay_mode=args.delay_mode,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        inter_stim_delay=args.inter_stim_delay,
        lsl=args.lsl,
        quiet=args.quiet,
    )
    simulator.run()


if __name__ == "__main__":
    main()
