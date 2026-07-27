"""
Example of a custom UI exposing custom_pipeline_example.py.

This UI reads sound_sets.json and enables the user to select one of the defined sound sets.
Then, it randomly plays a sound sampled from this set when a spindle is detected.

Custom variables may be passed through the configuration dictionary derived from RUN_SETTINGS.
For instance, in this UI, we pass the list of sounds in the selected set to MyCustomStimulator
via this dictionary in the following line:

exp_state.run_dict["sound_files"]

Note that all entries in this dictionary must be pickle-serializable.
"""


from multiprocessing import Process, Queue, Value
import time
import os
import socket
from datetime import datetime
from pathlib import Path
import pickle as pkl
import json

import alsaaudio
from alsaaudio import ALSAAudioError
import psutil
from nicegui import ui

from portiloop import __version__
from portiloop.src.core.capture import start_capture
from portiloop.src.core.utils import DummyAlsaMixer
from portiloop.src.core.constants import CSV_PATH, SD_CARD_DETECTED, STATE_PATH, NB_CHANNELS
from portiloop.src.custom.config import RUN_SETTINGS
from portiloop.src.custom.custom_processors import SpindleFilter
from portiloop.src.custom.custom_detectors import SleepSpindleRealTimeDetector

from custom_pipeline_example import MyCustomStimulator


portiloop_ID = socket.gethostname()


ENABLE_DISPLAY = True
LINE_PLOT_WINDOW = 5  # (window in seconds)
LINE_PLOT_UPDATE_EVERY = 2  # plot every N x TIMER_READ_DISPLAY_QUEUE s
LINE_PLOT_FIGSIZE = (3, 2)
LINE_PLOT_STRIDE = 4  # plot only 1 in N datapoints
TIMER_READ_DISPLAY_QUEUE = 1.0

TIMER_SD_CARD = 5.0


class ExperimentState:
    def __init__(self):
        self.point_index = 0
        self.len_plot = int(RUN_SETTINGS['frequency'] * LINE_PLOT_WINDOW / LINE_PLOT_STRIDE)
        self.started = False
        self.time_started = datetime.now()
        self.q_msg = Queue()
        self.run_dict = RUN_SETTINGS
        # enable all channels:
        self.run_dict["channel_states"] = ["simple"] * self.run_dict["nb_channels"]
        self.pause_value = Value('b', False)
        self._t_capture = None
        self.stim_on = False
        self.exp_name = ""
        self.display_q = Queue() if ENABLE_DISPLAY else None
        self.sd_card = False
        self.check_sd_card()
        self.lsl = False
        self.save_local = True
        self.display_rate = 0
        self.last_time_display = 0.0
        self.selected_channel = 'Channel 2'
        self.display_data = 'Raw'
        self.disk_str = f"Disk Usage:"
        self.min_delay = 0
        self.sleep_timeout = 0
        self.select_freq = 250
        self.power_line = 60
        self.persistent_file_name = STATE_PATH / "example_gui_state.pkl"

    def save(self):
        state = {
            "run_dict": self.run_dict,
            "lsl": self.lsl,
            "save_local": self.save_local,
            "selected_channel": self.selected_channel,
            "display_data": self.display_data,
            "min_delay": self.min_delay,
            "sleep_timeout": self.sleep_timeout,
            "select_freq": self.select_freq,
            "power_line": self.power_line,
        }
        with open(self.persistent_file_name, 'wb') as f:
            pkl.dump(state, f)

    def load(self):
        if self.persistent_file_name.is_file():
            with open(self.persistent_file_name, 'rb') as f:
                state = pkl.load(f)
            
            # check whether the previous state should be ignored (e.g., version change)
            run_dict = state["run_dict"]
            if run_dict["nb_channels"] != NB_CHANNELS or run_dict["software_version"] != __version__:
                return

            self.run_dict = run_dict
            self.lsl = state["lsl"]
            self.save_local = state["save_local"]
            self.display_data = state["display_data"]
            self.min_delay = state["min_delay"]
            self.sleep_timeout = state["sleep_timeout"]
            self.select_freq = state["select_freq"]
            self.power_line = state["power_line"]

            self.selected_channel = state["selected_channel"]

    def start(self):
        self.save()
        print(f"Frequency: {self.select_freq}, Sleep_timeout: {self.sleep_timeout}")
        # Set the variables for the experiment
        self.time_started = datetime.now()
        stim_str = "STIMON" if self.stim_on else "STIMOFF"
        time_str = self.time_started.strftime('%Y-%m-%d_%H-%M-%S')
        self.exp_name = f"{portiloop_ID}_{time_str}_{stim_str}.csv"
        print(f"Starting recording {self.exp_name.split('.')[0]}")

        print(f"STIMON = {self.stim_on}")

        self.run_dict['frequency'] = self.select_freq
        self.run_dict["filter_settings"]["power_line"] = self.power_line

        self.point_index = 0
        self.len_plot = int(self.run_dict['frequency'] * LINE_PLOT_WINDOW / LINE_PLOT_STRIDE)

        # Calculating how much time to pause in seconds
        if self.sleep_timeout > 0:
            self.time_unpause = self.time_started.timestamp() + self.sleep_timeout * 60
            self.pause_value.value = True
            print(f"Currently: {self.time_started.timestamp()}, Pausing until: {self.time_unpause}")

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
        else:
            self.run_dict['stimulate'] = False

        if self.min_delay != 0:
            self.run_dict['min_delay'] = int(self.min_delay) / 1000

        if self.lsl:
            self.run_dict['lsl'] = True
        else:
            self.run_dict['lsl'] = False

        if self.save_local:
            self.run_dict['record'] = True
        else:
            self.run_dict['record'] = False

        workspace_dir = CSV_PATH
        self.run_dict['filename'] = os.path.join(workspace_dir, self.exp_name.split('.')[0], self.exp_name)

        self._t_capture = Process(target=start_capture,
                                  args=(SpindleFilter,
                                        SleepSpindleRealTimeDetector,
                                        MyCustomStimulator,
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
        if ENABLE_DISPLAY:
            # flush display queue
            while self._t_capture.is_alive():
                while not self.display_q.empty():
                    try:
                        self.display_q.get_nowait()
                    except Exception:
                        break
                time.sleep(0.05)  # avoid busy loop
        self._t_capture.join()
        if ENABLE_DISPLAY:
            # drain remaining
            while not self.display_q.empty():
                try:
                    self.display_q.get_nowait()
                except Exception:
                    break
        self._t_capture = None
        print("Done.")

    def toggle_stim(self):
        self.stim_on = not self.stim_on

    def check_sd_card(self):
        if SD_CARD_DETECTED:
            self.sd_card = True
            self.disk_str = f"Disk Usage: {psutil.disk_usage(os.path.abspath('/media/sd_card/')).percent}%"
        else:
            self.sd_card = False
            self.disk_str = f"Disk Usage: {psutil.disk_usage(os.getcwd()).percent}%"

    def check_sleep_timeout(self):
        if self.pause_value.value:
            current_time = time.time()
            if current_time > self.time_unpause:
                self.pause_value.value = False


class CustomUI:
    def __init__(self):
        path_dir = Path(__file__).resolve().parent
        path_json = path_dir / "sound_sets.json"
        self.path_sounds = path_dir / "sounds"
        with open(path_json, "r") as f:
            self.sound_sets_dict = json.load(f)
        self.sound_set_ids = list(self.sound_sets_dict.keys())
        self.ui_values = {"id": self.sound_set_ids[0]}
    
    def get_sound_list(self):
        res = [str(self.path_sounds / f) for f in self.sound_sets_dict[self.ui_values["id"]]]
        return res

    def run(self,
            host='192.168.4.1',
            port=8083,
            title='Portiloop Control Center',
            dark=True,
            favicon='🧠',
            reload=False):

        exp_state = ExperimentState()
        try:
            exp_state.load()  # load persistent state
        except Exception as e:
            print(f"WARNING: Caught exception while loading persistent state: {e}")

        def start():
            exp_state.start()
            start_button.enabled = False

        def stop():
            exp_state.stop()
            start_button.enabled = True

        def test_sound():
            config_dict = RUN_SETTINGS
            config_dict["sound_files"] = self.get_sound_list()
            stimulator = MyCustomStimulator(config_dict)
            stimulator.test_stimulus()
            del stimulator

        def update_line_plot():

            if not ENABLE_DISPLAY:
                return

            x = []
            y = []

            # empty the display queue
            try:
                while not exp_state.display_q.empty():
                    channel = int(exp_state.selected_channel[-1]) - 1
                    point = exp_state.display_q.get(block=False)
                    time, raw_point, filtered_point = point
                    if exp_state.display_data == 'Raw':
                        point = raw_point[0][channel]
                    elif exp_state.display_data == 'Filter':
                        point = filtered_point[0][channel]
                    else:
                        point = 0.0
                    exp_state.point_index += 1
                    if exp_state.point_index % LINE_PLOT_STRIDE == 0:
                        x.append(time)
                        y.append(point)
            except Exception as e:
                print(f"Caught exception: {e}")

            # update the actual plot 
            if len(x) > 0 and len(y) > 0:
                line_plot.push(x, [y])
        
        def update_stimulator():
            sound_list = self.get_sound_list()
            exp_state.run_dict["sound_files"] = sound_list

        def _update_stimulator(caller):
            update_stimulator()

        update_stimulator()  # initialize the stimulator config to the first set of sounds

        ui.label('Portiloop NEST 🧠').classes('text-4xl font-mono')
        ui.label('Control Center').classes('text-2xl font-mono')

        ui.html(f"Connected to: <strong>{portiloop_ID}</strong> (v{RUN_SETTINGS['hardware_version']} - {RUN_SETTINGS['nb_channels']} channels)")
        ui.separator()

        with ui.tabs().classes('w-full') as tabs:
            control_tab = ui.tab('Control', icon='home')
            output_tab = ui.tab('Output', icon='timeline')
            advanced_tab = ui.tab('Advanced', icon='settings')

        with ui.tab_panels(tabs, value=control_tab).classes('w-full'):
            ############### First Tab ##################
            with ui.tab_panel(control_tab).classes('w-full items-center'):
                ################ Simple Options ################
                with ui.column().classes('w-full items-center'):
                    sd_card_checker = ui.checkbox('SD Card').bind_value_from(
                        exp_state,
                        'sd_card'
                    ).disable()

                    test_sound_button = ui.button('Test Sound 🔊', on_click=test_sound)

                    stim_toggle = ui.toggle(['Stim Off', 'Stim On'], value='Stim Off', on_change=lambda: exp_state.toggle_stim())

                    ui.separator()

                    ################ Recording Controls ##################
                    with ui.row():
                        start_button = ui.button('Start ▶', on_click=start, color='green').classes('text-2xl')
                        stop_button = ui.button('Stop', on_click=stop, color='orange').classes('text-2xl')
                    start_button.bind_enabled_to(stop_button, forward=lambda x: not x)
                    start_button.bind_enabled_to(stim_toggle)

                    ################# Control Display ##################
                    time_label = ui.label().classes('text-2xl')
                    save_file_label = ui.label().bind_text_from(
                        exp_state,
                        "exp_name",
                        backward=lambda x: f"Current experiment {x.split('.')[0]}")
                    timer = ui.timer(1.0, lambda: time_label.set_text(f'Timer: {str(datetime.now() - exp_state.time_started).split(".")[0]}'))
                    sd_card_timer = ui.timer(TIMER_SD_CARD, exp_state.check_sd_card)
                    start_button.bind_enabled_to(timer, 'active', forward=lambda x: not x)

            ############### Output Tab ####################
            with ui.tab_panel(output_tab).classes('w-full items-center'):
                ############# Line Plot stuff ################
                if ENABLE_DISPLAY:
                    line_timer = ui.timer(TIMER_READ_DISPLAY_QUEUE, update_line_plot, active=False)
                    start_button.bind_enabled_to(line_timer, 'active', forward=lambda x: not x)
                    line_plot = ui.line_plot(n=1, limit=exp_state.len_plot, update_every=LINE_PLOT_UPDATE_EVERY, figsize=LINE_PLOT_FIGSIZE, layout='tight')

                ui.separator()
                ############# Display Control ###############
                with ui.column().classes('w-full items-center'):
                    available_channels = [f"Channel {i+1}" for i in range(RUN_SETTINGS['nb_channels'])]
                    val = exp_state.selected_channel if exp_state.selected_channel in available_channels else available_channels[1]
                    select_channel_display = ui.select(available_channels, value=val, label="Display Channel")
                    select_channel_display.bind_value_to(exp_state, 'selected_channel').classes('w-1/2')

                    filtered_toggle = ui.toggle(['Raw', 'Filter'], value=exp_state.display_data)
                    filtered_toggle.bind_value_to(exp_state, 'display_data')

            ############### Advanced Tab #############
            with ui.tab_panel(advanced_tab).classes('w-full items-center'):
                ################ Advanced Options ###################
                with ui.column().classes('w-full items-center'):
                    ui.label("If you are a subject in an experiment, do not change any of these options unless explicitly prompted to!").classes('text-1.5xl').style('color:#d9a011')
                    ui.separator()
                    space_label = ui.label(f"Disk Usage: {psutil.disk_usage(os.getcwd())}%").bind_text_from(
                        exp_state,
                        'disk_str'
                    ).classes('text-2xl')
                    possible_freqs = [50, 100, 250, 500, 1000]
                    select_freq = ui.select(
                        possible_freqs,
                        value=exp_state.select_freq,
                        label="Sample Frequency (Hz)").bind_value_to(exp_state, 'select_freq').classes('w-3/4')
                    ui.separator().classes('w-2/3')
                    possible_notches = [60, 50]
                    select_notch = ui.select(
                        possible_notches,
                        value=exp_state.power_line,
                        label="Notch filter frequency (Hz)").bind_value_to(exp_state, 'power_line').classes('w-3/4')
                    ui.separator().classes('w-2/3')
                    sleep_timeout = ui.slider(min=0, max=40, value=exp_state.sleep_timeout).bind_value_to(exp_state, 'sleep_timeout').classes('w-3/4') #.props('label-always')
                    ui.label().bind_text_from(sleep_timeout, 'value', backward=lambda x: f"Sleep Timeout: {x} minutes")
                    sleep_timeout_timer = ui.timer(10, exp_state.check_sleep_timeout)
                    ui.separator().classes('w-2/3')
                    lsl_checker = ui.checkbox('Stream LSL', value=exp_state.lsl).bind_value_to(exp_state, 'lsl')
                    save_checker = ui.checkbox('Save Recording Locally', value=exp_state.save_local).bind_value_to(exp_state, 'save_local')
                    min_delay = ui.number(value=exp_state.min_delay, label='Stimulation Delay (in ms)').bind_value_to(exp_state, 'min_delay')
                    select_id = ui.select(self.sound_set_ids, value=self.sound_set_ids[0], on_change=_update_stimulator, label="ID").bind_value_to(self.ui_values, 'id')
                    select_id.classes('w-1/2')
                    start_button.bind_enabled_to(lsl_checker)
                    start_button.bind_enabled_to(save_checker)
                    start_button.bind_enabled_to(select_id)
                    start_button.bind_enabled_to(min_delay)
                    start_button.bind_enabled_to(select_freq)
                    start_button.bind_enabled_to(select_notch)
                    start_button.bind_enabled_to(sleep_timeout)
                    start_button.bind_enabled_to(sleep_timeout_timer, 'active', forward=lambda x: not x)

        if ENABLE_DISPLAY:
            line_plot.bind_visibility_from(start_button, 'enabled', backward=lambda x: not x)

        ui.run(
            host=host,
            port=port,
            title=title,
            dark=dark,
            favicon=favicon,
            reload=reload
            )


if __name__ == "__main__":
    gui = CustomUI()
    gui.run()
