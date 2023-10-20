from multiprocessing import Process, Queue, Value
import time
from nicegui import ui
from nicegui.events import ValueChangeEventArguments
from datetime import datetime
from portiloop.src.capture import start_capture
from portiloop.src.detection import SleepSpindleRealTimeDetector
from portiloop.src.stimulation import SleepSpindleRealTimeStimulator
from portiloop.src.hardware.leds import Color, LEDs
import socket
import os
from portiloop.src.utils import get_portiloop_version
from portiloop.src.hardware.frontend import Frontend

WORKSPACE_DIR = "/home/mendel/workspace/edf_recording/"

try:
    version = get_portiloop_version()
    frontend = Frontend(version)
    nb_channels = frontend.get_version()
finally:
    frontend.close()
    del frontend
# version = 2
# nb_channels = 6
portiloop_ID = socket.gethostname()

RUN_SETTINGS = {
    "version": version,
    "nb_channels": nb_channels,
    "frequency": 250,
    "duration": 36000,
    "filter": True,
    "record": True,
    "detect": False,
    "stimulate": False,
    "lsl": False,
    "display": False,
    "threshold": 0.82,
    "signal_input": "ADS",
    "python_clock": True,
    "signal_labels": [f"ch{i+1}" for i in range(nb_channels)],
    "channel_states": ["simple"] * nb_channels,
    "channel_detection": 2,
    "detection_sound": "stimul_15ms.wav",
    "spindle_detection_mode": "Fast",
    "spindle_freq": 10,
    "stim_delay": 0.0,
    "inter_stim_delay": 0.0,
    "volume": 100,
    "filter_settings": {
        "power_line": 60,
        "custom_fir": False,
        "custom_fir_order": 20,
        "custom_fir_cutoff": 30,
        "polyak_mean": 0.1,
        "polyak_std": 0.001,
        "epsilon": 1e-06,
        "filter_args": [
            True,
            True,
            True
        ]
    },
    "width_display": 1250,
    "filename": "/home/mendel/workspace/edf_recording/recording_test1.edf"
}


class ExperimentState:
    def __init__(self):
        self.started = False
        self.time_started = datetime.now()
        self.q_msg = Queue()
        self.detector_cls = SleepSpindleRealTimeDetector
        self.stimulator_cls = SleepSpindleRealTimeStimulator
        self.run_dict = RUN_SETTINGS
        self.pause_value = Value('b', False)
        self._t_capture = None
        self.stim_on = False
        self.exp_name = ""

    def start(self):
        # Set the variables for the experiment
        self.time_started = datetime.now()
        stim_str = "STIMON" if self.stim_on else "STIMOFF"
        time_str = self.time_started.strftime('%Y-%m-%d_%H-%M-%S')
        self.exp_name = f"{portiloop_ID}_{time_str}_{stim_str}.edf"

        print(f"Starting recording {self.exp_name.split('.')[0]}")

        if self.stim_on:
            self.run_dict['detect'] = True
            self.run_dict['stimulate'] = True
        else:
            self.run_dict['detect'] = False
            self.run_dict['stimulate'] = False

        self.run_dict['filename'] = os.path.join(WORKSPACE_DIR, self.exp_name)

        self._t_capture = Process(target=start_capture,
                                     args=(self.detector_cls,
                                           self.stimulator_cls,
                                           self.run_dict,
                                           self.q_msg,
                                           self.pause_value,))
        self._t_capture.start()
        print(f"PID start process: {self._t_capture.pid}. Kill this process if program crashes before end of execution.")
        
    def stop(self):
        print("Stopping recording...")
        self.q_msg.put('STOP')
        assert self._t_capture is not None
        self._t_capture.join()
        self._t_capture = None
        print("Done.")

    def toggle_stim(self):
        self.stim_on = not self.stim_on

exp_state = ExperimentState()

# exp_state.start()
# time.sleep(15)
# exp_state.stop()

def start():
    exp_state.start()
    start_button.enabled = False

def stop():
    exp_state.stop()
    start_button.enabled = True

def test_sound():
    stimulator_class = exp_state.stimulator_cls(soundname=RUN_SETTINGS['detection_sound'])
    stimulator_class.test_stimulus()
    del stimulator_class

ui.markdown('''## Portiloop Experiment''')
ui.label(f"Running on Portiloop {portiloop_ID} (v{version}) with {nb_channels} channels.")
ui.separator()

test_sound_button = ui.button('Test Sound 🔊', on_click=test_sound)

stim_toggle = ui.toggle(['Stim Off', 'Stim On'], value='Stim Off', on_change=lambda: exp_state.toggle_stim())

with ui.row():
    start_button = ui.button('Start ▶', on_click=start, color='green')
    stop_button = ui.button('Stop', on_click=stop, color='orange')
    start_button.bind_enabled_to(stop_button, forward=lambda x: not x)
    start_button.bind_enabled_to(stim_toggle)

time_label = ui.label()
save_file_label = ui.label().bind_text_from(
    exp_state, 
    "exp_name", 
    backward=lambda x: f"Current experiment {x.split('.')[0]}")

timer = ui.timer(1.0, lambda: time_label.set_text(f'Timer: {str(datetime.now() - exp_state.time_started).split(".")[0]}'))
start_button.bind_enabled_to(timer, 'active', forward=lambda x: not x)

ui.run(
    host='192.168.4.1', 
    port=8080,
    title='Portiloop Experiment',
    dark=True,
    favicon='🧠',
    reload=False)
