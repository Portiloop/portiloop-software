#!/usr/bin/env bash
set -euo pipefail

# End-to-end compilation script for Portiloop ONNX models -> D-Robotics RDK X5 BPU (.bin, march='bayes-e').
# Uses `hb_mapper` directly if installed on PATH, or runs inside the D-Robotics OpenExplorer Docker container
# (OPENEXPLORER_IMAGE, default: openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8).

BPU_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BPU_DIR}/../.." && pwd)"
MODELS_DIR="${REPO_ROOT}/portiloop/models"
OPENEXPLORER_IMAGE="${OPENEXPLORER_IMAGE:-openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8}"

cd "${BPU_DIR}"

echo "[1/3] Preparing PTQ calibration data..."
if python3 -c "import onnxruntime" >/dev/null 2>&1; then
    python3 "${BPU_DIR}/prepare_calibration_data.py" --models-dir "${MODELS_DIR}" --cal-dir "${BPU_DIR}/cal_data"
elif command -v uv >/dev/null 2>&1; then
    uv run --with onnxruntime --with numpy python3 "${BPU_DIR}/prepare_calibration_data.py" --models-dir "${MODELS_DIR}" --cal-dir "${BPU_DIR}/cal_data"
else
    echo "ERROR: 'onnxruntime' is not installed for python3 and 'uv' was not found." >&2
    exit 1
fi

run_hb_mapper() {
    if command -v hb_mapper >/dev/null 2>&1; then
        hb_mapper "$@"
    elif command -v docker >/dev/null 2>&1; then
        local uid_gid
        uid_gid="$(id -u):$(id -g)"
        docker run --rm \
            -v "${REPO_ROOT}:${REPO_ROOT}" \
            -w "${BPU_DIR}" \
            "${OPENEXPLORER_IMAGE}" \
            bash -c "hb_mapper \"\$@\"; ret=\$?; chown -R ${uid_gid} \"${BPU_DIR}\"; exit \$ret" -- "$@"
    else
        echo "ERROR: Neither 'hb_mapper' nor 'docker' (for ${OPENEXPLORER_IMAGE}) was found." >&2
        exit 1
    fi
}

echo "[2/3] Compiling ONNX models for RDK X5 BPU (march=bayes-e)..."
run_hb_mapper makertbin --config "${BPU_DIR}/configs/spindle_bpu.yaml" --model-type onnx
run_hb_mapper makertbin --config "${BPU_DIR}/configs/tinysleepnet_bpu.yaml" --model-type onnx
run_hb_mapper makertbin --config "${BPU_DIR}/configs/nanosleepnet_bpu.yaml" --model-type onnx

echo "[3/3] Installing compiled .bin BPU models into ${MODELS_DIR}..."
cp -f "${BPU_DIR}/build_spindle/portiloop_model_bpu.bin" "${MODELS_DIR}/portiloop_model_bpu.bin"
cp -f "${BPU_DIR}/build_tinysleepnet/tinysleepnet_bpu.bin" "${MODELS_DIR}/tinysleepnet/tinysleepnet_bpu.bin"
cp -f "${BPU_DIR}/build_nanosleepnet/nanosleepnet_bpu.bin" "${MODELS_DIR}/nanosleepnet/nanosleepnet_bpu.bin"

echo "Done! Compiled BPU models:"
ls -lh \
    "${MODELS_DIR}/portiloop_model_bpu.bin" \
    "${MODELS_DIR}/tinysleepnet/tinysleepnet_bpu.bin" \
    "${MODELS_DIR}/nanosleepnet/nanosleepnet_bpu.bin"
