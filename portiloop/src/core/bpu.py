"""
D-Robotics RDK X5 BPU (`hbm_runtime.HB_HBMRuntime`, with `hobot_dnn.pyeasy_dnn` fallback)
and ONNX Runtime (`onnxruntime`) inference wrappers for Portiloop models.
"""
import logging
from pathlib import Path
import numpy as np

from portiloop.src import IS_RDKX5

logger = logging.getLogger(__name__)

_hbm_runtime = None
_dnn = None
if IS_RDKX5:
    try:
        import hbm_runtime as _hbm_runtime
    except ImportError:
        _hbm_runtime = None
    if _hbm_runtime is None:
        try:
            from hobot_dnn import pyeasy_dnn as _dnn
        except ImportError:
            try:
                import hobot_dnn as _dnn
            except ImportError:
                _dnn = None

try:
    import onnxruntime as _ort
except ImportError:
    _ort = None


def bpu_available() -> bool:
    """True when running on an RDK X5 with `hbm_runtime` or `hobot_dnn` installed."""
    return _hbm_runtime is not None or _dnn is not None


def onnx_available() -> bool:
    """True when `onnxruntime` is installed."""
    return _ort is not None


class BPUSession:
    """
    Thin wrapper around D-Robotics `hbm_runtime.HB_HBMRuntime` (or legacy `pyeasy_dnn`)
    with named input/output lookup.
    """
    def __init__(self, bin_path):
        self.bin_path = Path(bin_path)
        if _hbm_runtime is not None:
            self._use_hbm = True
            self.model = _hbm_runtime.HB_HBMRuntime(str(self.bin_path))
            self.model_name = self.model.model_names[0]
            self.input_names = list(self.model.input_names[self.model_name])
            self.input_shapes = [tuple(self.model.input_shapes[self.model_name][n]) for n in self.input_names]
            self.output_names = list(self.model.output_names[self.model_name])
            self.output_shapes = [tuple(self.model.output_shapes[self.model_name][n]) for n in self.output_names]
        elif _dnn is not None:
            self._use_hbm = False
            models = _dnn.load(str(self.bin_path))
            self.model = models[0]
            self.model_name = getattr(self.model, "name", self.bin_path.stem)
            self.input_names = [inp.name for inp in self.model.inputs]
            self.input_shapes = [tuple(inp.properties.shape) for inp in self.model.inputs]
            self.output_names = [out.name for out in self.model.outputs]
            self.output_shapes = [tuple(out.properties.shape) for out in self.model.outputs]
        else:
            raise RuntimeError("Neither hbm_runtime nor hobot_dnn is available on this system")

    def run(self, feed_dict: dict) -> dict:
        if self._use_hbm:
            model_inputs = {}
            for name, shape in zip(self.input_names, self.input_shapes):
                arr = np.asarray(feed_dict[name], dtype=np.float32)
                if arr.shape != shape and arr.size == int(np.prod(shape)):
                    arr = arr.reshape(shape)
                model_inputs[name] = np.ascontiguousarray(arr)
            raw_outs = self.model.run({self.model_name: model_inputs})[self.model_name]
            return {
                name: np.asarray(raw_outs[name], dtype=np.float32)
                for name in self.output_names
            }

        ordered_inputs = []
        for name, shape in zip(self.input_names, self.input_shapes):
            arr = np.asarray(feed_dict[name], dtype=np.float32)
            if arr.shape != shape and arr.size == int(np.prod(shape)):
                arr = arr.reshape(shape)
            ordered_inputs.append(np.ascontiguousarray(arr))
        if len(ordered_inputs) == 1:
            raw_outs = self.model.forward(ordered_inputs[0])
        else:
            raw_outs = self.model.forward(ordered_inputs)
        if not isinstance(raw_outs, (list, tuple)):
            raw_outs = [raw_outs]
        result = {}
        for name, out_tensor in zip(self.output_names, raw_outs):
            buf = getattr(out_tensor, "buffer", out_tensor)
            result[name] = np.array(buf, dtype=np.float32, copy=True)
        return result


class ONNXSession:
    """
    Thin wrapper around `onnxruntime.InferenceSession` with the same `.run(feed_dict)` interface.
    """
    def __init__(self, onnx_path):
        if _ort is None:
            raise RuntimeError("onnxruntime is not installed")
        self.onnx_path = Path(onnx_path)
        sess_opts = _ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        self.session = _ort.InferenceSession(
            str(self.onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def run(self, feed_dict: dict) -> dict:
        inputs = {k: np.ascontiguousarray(feed_dict[k], dtype=np.float32) for k in self.input_names}
        outs = self.session.run(self.output_names, inputs)
        return dict(zip(self.output_names, outs))
