# Portiloop ONNX -> RDK X5 BPU (`bayes-e`) Conversion Pipeline

This directory contains the conversion and quantization pipeline to compile Portiloop neural network models for the **D-Robotics RDK X5 BPU** (Horizon Sunrise X5, BPU architecture `bayes-e`, 10 TOPS INT8).

## Supported Models

| Pipeline | ONNX Model (`opset 11`) | Inputs (4D NCHW + state) | Outputs | Compiled BPU Binary |
|---|---|---|---|---|
| **Sleep Spindles** (`SleepSpindleRealTimeDetector`) | `portiloop/models/portiloop_spindle.onnx` | `x: [1,1,1,54]`, `h: [1,7]` | `y: [1,1]`, `h_out: [1,7]` | `portiloop/models/portiloop_model_bpu.bin` |
| **Sleep Staging (`TinySleepNet`)** (`SleepStagingDetector`) | `portiloop/models/tinysleepnet/tinysleepnet.onnx` | `x: [1,1,1,3000]`, `c: [1,128]`, `h: [1,128]` | `logits: [1,5]`, `c_out: [1,128]`, `h_out: [1,128]` | `portiloop/models/tinysleepnet/tinysleepnet_bpu.bin` |
| **Sleep Staging (`NanoSleepNet`)** (`NanoSleepNetStagingDetector`) | `portiloop/models/nanosleepnet/nanosleepnet.onnx` | `x: [1,1,1,3000]`, `past_tokens: [1,64,1,9]` | `logits: [1,5]`, `token: [1,64]` | `portiloop/models/nanosleepnet/nanosleepnet_bpu.bin` |

## 1. Exporting ONNX Models from `portiloop-training`

`portiloop-training` (`portiloop_software.portiloop_python.ANN.onnx_export`) exports trained PyTorch checkpoints (as well as `TinySleepNet` and `NanoSleepNet`) directly to BPU-compatible ONNX (`opset 11`), lowering 1D convolutions/pooling to 4D `NCHW` 2D convolutions/pooling and unrolling single-step recurrent cells (`GRU` / peephole `LSTM`) into explicit matrix multiplications and activations:

```bash
cd ~/Projects/portiloop-training
python3 -m portiloop_software.portiloop_python.ANN.onnx_export \
    --output-dir ~/Projects/portiloop-software/portiloop/models \
    --tinysleepnet-tflite ~/Projects/portiloop-software/portiloop/models/tinysleepnet/tinysleepnet_float.tflite \
    --nanosleepnet-npz ~/Projects/portiloop-software/portiloop/models/nanosleepnet/nanosleepnet.npz
```

During training (`portiloop_detector_training.py` with `save_model=True`), `portiloop-training` also exports `<experiment_name>.onnx` automatically whenever a new best checkpoint is saved.

## 2. Generating PTQ Calibration Data & Compiling `.bin` for BPU

To compile the `.onnx` models into `.bin` BPU binaries on an x86_64 host with the D-Robotics OpenExplorer X5 toolchain (`hb_mapper` or Docker image `openexplorer/ai_toolchain_ubuntu_20_x5_cpu`):

```bash
./portiloop/bpu/compile_bpu.sh
```

What `compile_bpu.sh` does:
1. Runs `portiloop/bpu/prepare_calibration_data.py` using `onnxruntime` to generate multi-step trajectory calibration datasets (`portiloop/bpu/cal_data/{spindle,tinysleepnet,nanosleepnet}/`) so that both input EEG windows/epochs and recurrent hidden states (`h`, `c`, `past_tokens`) are calibrated on realistic activation distributions.
2. Runs `hb_mapper makertbin` (`march: 'bayes-e'`) using the YAML configs in `portiloop/bpu/configs/`:
   - `configs/spindle_bpu.yaml`
   - `configs/tinysleepnet_bpu.yaml`
   - `configs/nanosleepnet_bpu.yaml`
3. Copies the compiled `.bin` files into `portiloop/models/`.

## 3. Runtime Selection on the RDK X5 (`portiloop/src/core/bpu.py`)

At runtime on the RDK X5 (`IS_RDKX5 == True`), `SleepSpindleRealTimeDetector`, `TinySleepNetStep`, and `NanoSleepNetStep` automatically check for compiled BPU models:
1. **BPU (`hobot_dnn.pyeasy_dnn`)**: Used automatically whenever `<model>_bpu.bin` is present in `portiloop/models/` and `hobot_dnn` is installed.
2. **ONNX Runtime (`onnxruntime`)**: Used if `.onnx` is selected or when `tflite_runtime` is unavailable.
3. **TFLite / NumPy fallback (`tflite_runtime` / `numpy`)**: Used when `.bin` is not present, ensuring all pipelines work out-of-the-box.
