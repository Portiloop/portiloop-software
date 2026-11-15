"""
Generates calibration datasets for Horizon / D-Robotics RDK X5 BPU (`hb_mapper makertbin`, march='bayes-e')
from the exported Portiloop ONNX models (`portiloop_spindle.onnx`, `tinysleepnet.onnx`, `nanosleepnet.onnx`).

Because all three Portiloop models are stateful across time steps (`h`, `c`, `past_tokens`),
this script runs multi-step inference trajectories with `onnxruntime` on realistic EEG signals
so that calibration inputs for both signal and recurrent state tensors match real inference distributions.
"""
import argparse
from pathlib import Path
import numpy as np
import onnxruntime as ort


def _synthetic_sleep_eeg(n_samples: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Generates band-limited sleep-like EEG (slow waves + alpha/spindle bursts + 1/f background) in uV."""
    t = np.arange(n_samples, dtype=np.float64) / fs
    sig = (
        25.0 * np.sin(2.0 * np.pi * 0.8 * t + rng.uniform(0, 2 * np.pi))
        + 12.0 * np.sin(2.0 * np.pi * 2.5 * t + rng.uniform(0, 2 * np.pi))
        + 8.0 * np.sin(2.0 * np.pi * 13.0 * t + rng.uniform(0, 2 * np.pi)) * (0.5 + 0.5 * np.sin(2.0 * np.pi * 0.2 * t))
        + rng.normal(0.0, 10.0, size=n_samples)
    )
    return np.clip(sig, -192.0, 192.0).astype(np.float32)


def generate_spindle_calibration(onnx_path: Path, cal_dir: Path, num_samples: int = 64, seed: int = 42):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x_dir = cal_dir / "x"
    h_dir = cal_dir / "h"
    x_dir.mkdir(parents=True, exist_ok=True)
    h_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    h = np.zeros((1, 7), dtype=np.float32)
    for idx in range(num_samples):
        if idx % 16 == 0:
            h = np.zeros((1, 7), dtype=np.float32)
        window = (_synthetic_sleep_eeg(54, 250.0, rng) / 20.0).reshape(1, 1, 1, 54).astype(np.float32)
        window.tofile(x_dir / f"sample_{idx:04d}.bin")
        h.tofile(h_dir / f"sample_{idx:04d}.bin")
        _, h = sess.run(None, {"x": window, "h": h})


def generate_tinysleepnet_calibration(onnx_path: Path, probe_path: Path, cal_dir: Path,
                                      num_samples: int = 32, seed: int = 42):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x_dir = cal_dir / "x"
    c_dir = cal_dir / "c"
    h_dir = cal_dir / "h"
    for d in (x_dir, c_dir, h_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    c = np.zeros((1, 128), dtype=np.float32)
    h = np.zeros((1, 128), dtype=np.float32)

    probe_epoch = None
    if probe_path.is_file():
        with np.load(probe_path) as probe:
            probe_epoch = probe["x"].astype(np.float32).reshape(1, 1, 1, 3000)

    for idx in range(num_samples):
        if idx % 10 == 0:
            c = np.zeros((1, 128), dtype=np.float32)
            h = np.zeros((1, 128), dtype=np.float32)
        if idx == 0 and probe_epoch is not None:
            epoch = probe_epoch
        else:
            epoch = _synthetic_sleep_eeg(3000, 100.0, rng).reshape(1, 1, 1, 3000)
        epoch.tofile(x_dir / f"sample_{idx:04d}.bin")
        c.tofile(c_dir / f"sample_{idx:04d}.bin")
        h.tofile(h_dir / f"sample_{idx:04d}.bin")
        _, c, h = sess.run(None, {"x": epoch, "c": c, "h": h})


def generate_nanosleepnet_calibration(onnx_path: Path, probe_path: Path, cal_dir: Path,
                                      num_samples: int = 32, seed: int = 42):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x_dir = cal_dir / "x"
    past_dir = cal_dir / "past_tokens"
    for d in (x_dir, past_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    past = np.zeros((1, 64, 1, 9), dtype=np.float32)

    probe_epoch = None
    if probe_path.is_file():
        with np.load(probe_path) as probe:
            probe_epoch = probe["x"].astype(np.float32).reshape(1, 1, 1, 3000)

    for idx in range(num_samples):
        if idx % 12 == 0:
            past = np.zeros((1, 64, 1, 9), dtype=np.float32)
        if idx == 0 and probe_epoch is not None:
            epoch = probe_epoch
        else:
            epoch = _synthetic_sleep_eeg(3000, 100.0, rng).reshape(1, 1, 1, 3000)
        epoch.tofile(x_dir / f"sample_{idx:04d}.bin")
        past.tofile(past_dir / f"sample_{idx:04d}.bin")
        _, tok = sess.run(None, {"x": epoch, "past_tokens": past})
        past = np.concatenate([past[:, :, :, 1:], tok.reshape(1, 64, 1, 1)], axis=3)


def main():
    parser = argparse.ArgumentParser(description="Generate BPU PTQ calibration data for Portiloop ONNX models")
    models_root = Path(__file__).resolve().parents[1] / "models"
    default_cal_root = Path(__file__).resolve().parent / "cal_data"
    parser.add_argument("--models-dir", type=Path, default=models_root)
    parser.add_argument("--cal-dir", type=Path, default=default_cal_root)
    args = parser.parse_args()

    spindle_onnx = args.models_dir / "portiloop_spindle.onnx"
    tsn_onnx = args.models_dir / "tinysleepnet" / "tinysleepnet.onnx"
    nsn_onnx = args.models_dir / "nanosleepnet" / "nanosleepnet.onnx"
    probe_npz = args.models_dir / "tinysleepnet" / "probe.npz"

    if spindle_onnx.is_file():
        generate_spindle_calibration(spindle_onnx, args.cal_dir / "spindle")
        print(f"Generated spindle calibration data in {args.cal_dir / 'spindle'}")

    if tsn_onnx.is_file():
        generate_tinysleepnet_calibration(tsn_onnx, probe_npz, args.cal_dir / "tinysleepnet")
        print(f"Generated TinySleepNet calibration data in {args.cal_dir / 'tinysleepnet'}")

    if nsn_onnx.is_file():
        generate_nanosleepnet_calibration(nsn_onnx, probe_npz, args.cal_dir / "nanosleepnet")
        print(f"Generated NanoSleepNet calibration data in {args.cal_dir / 'nanosleepnet'}")


if __name__ == "__main__":
    main()
