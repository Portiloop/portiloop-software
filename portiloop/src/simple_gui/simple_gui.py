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

WORKSPACE_DIR = "/home/mendel/workspace/edf_recording/"

RUN_SETTINGS = {
    "version": 2,
    "nb_channels": 4,
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
    "signal_labels": [
        "ch1",
        "ch2",
        "ch3",
        "ch4"
    ],
    "channel_states": [
        "simple",
        "simple",
        "simple",
        "simple"
    ],
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

    def start(self):
        # Set the variables for the experiment
        self.time_started = datetime.now()
        stim_str = "STIMON" if self.stim_on else "STIMOFF"
        time_str = self.time_started.strftime('%Y-%m-%d_%H-%M-%S')
        exp_name = f"{socket.gethostname()}_{time_str}_{stim_str}.edf"
        if self.stim_on:
            self.run_dict['detect'] = True
            self.run_dict['stimulate'] = True
        else:
            self.run_dict['detect'] = False
            self.run_dict['stimulate'] = False

        self.run_dict['filename'] = os.path.join(WORKSPACE_DIR, exp_name)

        self._t_capture = Process(target=start_capture,
                                     args=(self.detector_cls,
                                           self.stimulator_cls,
                                           self.run_dict,
                                           self.q_msg,
                                           self.pause_value,))
        self._t_capture.start()
        print(f"PID start process: {self._t_capture.pid}. Kill this process if program crashes before end of execution.")
        
    def stop(self):
        print("Pressed Stop Button")
        self.q_msg.put('STOP')
        assert self._t_capture is not None
        self._t_capture.join()
        self._t_capture = None

    def toggle_stim(self):
        self.stim_on = not self.stim_on

exp_state = ExperimentState()

def start():
    exp_state.start()
    start_button.enabled = False

def stop():
    exp_state.stop()
    start_button.enabled = True

ui.markdown('''## Portiloop Experiment''')
ui.separator()

stim_toggle = ui.toggle(['Stim Off', 'Stim On'], value='Stim Off', on_change=lambda: exp_state.toggle_stim())

with ui.row():
    start_button = ui.button('Start', on_click=start)
    stop_button = ui.button('Stop', on_click=stop)
    start_button.bind_enabled_to(stop_button, forward=lambda x: not x)
    start_button.bind_enabled_to(stim_toggle)

time_label = ui.label()

timer = ui.timer(1.0, lambda: time_label.set_text(f'Timer: {str(datetime.now() - exp_state.time_started).split(".")[0]}'))
start_button.bind_enabled_to(timer, 'active', forward=lambda x: not x)

ui.run(
    host='192.168.4.1', 
    port=8080,
    title='Portiloop Experiment',
    dark=True,
    favicon='🧠',
    reload=False)