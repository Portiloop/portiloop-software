from multiprocessing import Process, Queue, Value
import time
from nicegui import ui
from nicegui.events import ValueChangeEventArguments
from datetime import datetime
from portiloop.src.capture import start_capture
from portiloop.src.detection import SleepSpindleRealTimeDetector
from portiloop.src.stimulation import SleepSpindleRealTimeStimulator, AlternatingStimulator
from portiloop.src.hardware.leds import Color, LEDs
import socket
import os
from portiloop.src.utils import get_portiloop_version
from portiloop.src.hardware.frontend import Frontend
import alsaaudio
from alsaaudio import ALSAAudioError

# This line is to start something which seems to be necessary to make sure the sound works properly. Not sure why
os.system('aplay /home/mendel/portiloop-software/portiloop/sounds/sample1.wav')

WORKSPACE_DIR_SD = "/media/sd_card/workspace/edf_recordings/"
WORKSPACE_DIR_IN = "/home/mendel/workspace/edf_recordings/"

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
    "detect": True,
    "stimulate": False,
    "lsl": False,
    "display": False,
    "threshold": 0.75,
    "signal_input": "ADS",
    "python_clock": True,
    "signal_labels": [f"ch{i+1}" for i in range(nb_channels)],
    "channel_states": [
        "simple",
        "simple",
        "simple",
        "simple",
        "disabled",
        "disabled"],
    "channel_detection": 2,
    "detection_sound": "15msPN_48kHz_norm_stereo.wav",
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
    "filename": "/home/mendel/workspace/edf_recording/recording_test1.csv"
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
        self.display_q = Queue()
        self.check_sd_card()
        self.lsl = False
        self.save_local = True
        self.stimulator_type = 'Spindle'
        self.display_rate = 0
        self.last_time_display = 0.0

    def start(self):
        # Set the variables for the experiment
        self.time_started = datetime.now()
        stim_str = "STIMON" if self.stim_on else "STIMOFF"
        time_str = self.time_started.strftime('%Y-%m-%d_%H-%M-%S')
        self.exp_name = f"{portiloop_ID}_{time_str}_{stim_str}.csv"
        print(f"Starting recording {self.exp_name.split('.')[0]}")

        try:
            mixers = alsaaudio.mixers()
            if len(mixers) <= 0:
                print(f"No ALSA mixer found.")
                mixer = DummyAlsaMixer()
            else:
                mixer = alsaaudio.Mixer(control='SoftMaster', device='dmixer')
                # mixer = alsaaudio.Mixer()
        except ALSAAudioError as e:
            print(e)
            print(f"No ALSA mixer found. Volume control will not be available from notebook.\nAvailable mixers were:\n{mixers}")
            mixer = DummyAlsaMixer()
            
        volume = mixer.getvolume()[0]  # we will set the same volume on all channels
        self.run_dict['volume'] = volume

        if self.stim_on:
            self.run_dict['stimulate'] = True
            if self.stimulator_type == 'Spindle':
                self.stimulator_cls = SleepSpindleRealTimeStimulator
            elif self.stimulator_type == 'Interval':
                self.stimulator_cls = AlternatingStimulator
                self.run_dict['detect'] = False
        else:
            self.run_dict['stimulate'] = False
            self.stimulator_cls = None

        if self.lsl:
            self.run_dict['lsl'] = True
        else:
            self.run_dict['lsl'] = False

        if self.save_local:
            self.run_dict['record'] = True
        else:
            self.run_dict['record'] = False

        if self.sd_card:
            workspace_dir = WORKSPACE_DIR_SD
        else:
            workspace_dir = WORKSPACE_DIR_IN

        self.run_dict['filename'] = os.path.join(workspace_dir, self.exp_name)

        self._t_capture = Process(target=start_capture,
                                     args=(self.detector_cls,
                                           self.stimulator_cls,
                                           self.run_dict,
                                           self.q_msg,
                                           self.display_q,
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

    def check_sd_card(self):
        self.sd_card = os.path.exists("/media/sd_card/workspace/edf_recordings")


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

def update_line_plot():
    now = datetime.now()
    x = now.timestamp()
    try:
        # empty the queue
        x = []
        y = [[]] * nb_channels
        while not exp_state.display_q.empty():
            point = exp_state.display_q.get(block=False)
            time, point = point
            x.append(time)
            for channel in range(nb_channels):
                y[channel].append(point[0][channel])
    except Exception:
        print("AAAAAAAAAAAAAAAAAAH")
    
    # exp_state.display_rate += len(y)
    # current_time = time.time()
    # if current_time - exp_state.last_time_display >= 1:
    #     # print(f"Display rate: {exp_state.display_rate}")
    #     exp_state.last_time_display = current_time
    #     exp_state.display_rate = 0
    print(len(y))
    print(len(y[0]))
    print(y)
    if len(x) > 0 and len(y[0]) > 0:
        line_plot.push(x, y)

ui.markdown('''## Portiloop Control Center''')
ui.label(f"Running on Portiloop {portiloop_ID} (v{version}) with {nb_channels} channels.")
ui.separator()

with ui.column().classes('w-full items-center'):
    sd_card_checker = ui.checkbox('SD Card').classes('w-full justify-center').bind_value_from(
        exp_state,
        'sd_card'
    ).disable()

    test_sound_button = ui.button('Test Sound 🔊', on_click=test_sound).classes('w-half justify-center')

    stim_toggle = ui.toggle(['Stim Off', 'Stim On'], value='Stim Off', on_change=lambda: exp_state.toggle_stim()).classes('w-full justify-center')

def disable_stim_toggle_callback(caller):
    if caller.value == 'Interval':
        stim_toggle.disable()
        stim_toggle.value = 'Stim On'
    else:
        stim_toggle.enable()

with ui.expansion('Advanced Options', icon='settings').classes('w-full items-center'):
    ui.label("If you are a subject in an experiment, do not change any of these options!")
    lsl_checker = ui.checkbox('Stream LSL').bind_value_to(exp_state, 'lsl')
    save_checker = ui.checkbox('Save Local', value=True).bind_value_to(exp_state, 'save_local')
    select_stimulator = ui.select(['Spindle', 'Interval'], value='Spindle', on_change=disable_stim_toggle_callback).bind_value_to(exp_state, 'stimulator_type')

ui.separator()

line_timer = ui.timer(1/25, update_line_plot, active=False)

with ui.row().classes('w-full justify-center'):
    start_button = ui.button('Start ▶', on_click=start, color='green').classes('w-full justify-center')
    stop_button = ui.button('Stop', on_click=stop, color='orange').classes('w-full justify-center')
    start_button.bind_enabled_to(stop_button, forward=lambda x: not x)
    start_button.bind_enabled_to(stim_toggle)
    start_button.bind_enabled_to(lsl_checker)
    start_button.bind_enabled_to(save_checker)
    start_button.bind_enabled_to(select_stimulator)
    start_button.bind_enabled_to(line_timer, 'active', forward=lambda x: not x)

time_label = ui.label()
save_file_label = ui.label().classes('w-full justify-center').bind_text_from(
    exp_state, 
    "exp_name", 
    backward=lambda x: f"Current experiment {x.split('.')[0]}")

timer = ui.timer(1.0, lambda: time_label.set_text(f'Timer: {str(datetime.now() - exp_state.time_started).split(".")[0]}'))
sd_card_timer = ui.timer(0.5, exp_state.check_sd_card)
start_button.bind_enabled_to(timer, 'active', forward=lambda x: not x)

line_plot = ui.line_plot(n=nb_channels, limit=250 * 5, figsize=(3, 2), update_every=5)
line_plot.bind_visibility_from(start_button, 'enabled', backward=lambda x: not x)

ui.run(
    host='192.168.4.1', 
    port=8081,
    title='Portiloop Experiment',
    dark=True,
    favicon='🧠',
    reload=False)
