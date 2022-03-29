from pycoral.utils import edgetpu
import time
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

DEFAULT_MODEL_PATH = str(Path(__file__).parent / "models/portiloop_model_quant.tflite")
print(DEFAULT_MODEL_PATH)

class AbstractQuantizedModelForInference(ABC):
    @abstractmethod
    def add_datapoints(self, input_float):
        return NotImplemented

class QuantizedModelForInference(AbstractQuantizedModelForInference):
    def __init__(self, num_models_parallel=8, window_size=54, seq_stride=42, model_path=None, verbose=False, channel=2):
        model_path = DEFAULT_MODEL_PATH if model_path is None else model_path
        self.verbose = verbose
        self.channel = channel
        self.num_models_parallel = num_models_parallel
        
        self.interpreters = []
        for i in range(self.num_models_parallel):
            self.interpreters.append(edgetpu.make_interpreter(model_path))
            self.interpreters[i].allocate_tensors()
        self.interpreter_counter = 0
        
        self.input_details = self.interpreters[0].get_input_details()
        self.output_details = self.interpreters[0].get_output_details()
        
        self.buffer = []
        self.seq_stride = seq_stride
        self.window_size = window_size 
        
        self.stride_counters = [np.floor((self.seq_stride / self.num_models_parallel) * i) for i in range(self.num_models_parallel)]
        for idx, i in enumerate(self.stride_counters[1:]):
            self.stride_counters[idx+1] = i - self.stride_counters[idx]
        self.current_stride_counter = self.stride_counters[0] - 1
        
        
    def add_datapoints(self, inputs_float):
        res = []
        for inp in inputs_float:
            result = self.add_datapoint(inp)
            if result is not None:
                res.append(result)
        return res
    
        
    def add_datapoint(self, input_float):
        input_float = input_float[self.channel-1]
        result = None
        self.buffer.append(input_float)
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[1:]
            self.current_stride_counter += 1
            if self.current_stride_counter == self.stride_counter[self.interpreter_counter]:
                result = self.call_model(self.interpreter_counter, self.buffer)
                self.interpreter_counter += 1
                self.interpreter_counter %= self.num_model_parallel
                self.current_stride_counter = 0
        return result
            
                
        
    def call_model(self, idx, input_float=None):
        if input_float is None:
            # For debuggin purposes
            input_shape = input_details[0]['shape']
            input = np.array(np.random.random_sample(input_shape), dtype=np.int8)
        else:
            # Convert float input to Int
            input_scale, input_zero_point = input_details[0]["quantization"]
            input = np.asarray(input_float) / input_scale + input_zero_point
            input = input.astype(input_details[0]["dtype"])

        interpreter.set_tensor(input_details[0]['index'], input)
        if self.verbose:
            start_time = time.time()

        interpreter.invoke()

        if self.verbose:
            end_time = time.time()

        output = interpreter.get_tensor(output_details[0]['index'])
        output_scale, output_zero_point = input_details[0]["quantization"]
        output = float(output - output_zero_point) * output_scale

        if self.verbose:
            print(f"Computed output {output} in {end_time - start_time} seconds")

        return output
    
    