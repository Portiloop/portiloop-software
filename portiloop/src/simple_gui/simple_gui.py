from multiprocessing import Process, Queue, Value
import time
import os
import socket
from datetime import datetime
import pickle as pkl

import alsaaudio
from alsaaudio import ALSAAudioError
import psutil
from nicegui import ui

from portiloop import __version__
from portiloop.src.core.capture import start_capture
from portiloop.src.core.utils import DummyAlsaMixer
from portiloop.src.core.constants import CSV_PATH, SD_CARD_DETECTED, STATE_PATH, NB_CHANNELS
from portiloop.src.custom.config import RUN_SETTINGS
from portiloop.src.custom.custom_pipelines import PIPELINES

portiloop_ID = socket.gethostname()


class ExperimentState:
    def __init__(self, pipelines = PIPELINES):
        self._pipelines = pipelines
        self.pipeline_keys = list(self._pipelines.keys())
        self.pipeline_key = self.pipeline_keys[0]
        # self.processor_cls = pipeline["processor"]
        # self.detector_cls = pipeline["detector"]
        # self.stimulator_cls = pipeline["stimulator"]

        self.started = False
        self.time_started = datetime.now()
        self.q_msg = Queue()
        # self.processor_cls = SpindleFilter
        # self.detector_cls = SleepSpindleRealTimeDetector
        # self.stimulator_cls = SleepSpindleRealTimeStimulator
        self.run_dict = RUN_SETTINGS
        # enable all channels:
        self.run_dict["channel_states"] = ["simple"] * self.run_dict["nb_channels"]
        self.pause_value = Value('b', False)
        self._t_capture = None
        self.stim_on = False
        self.exp_name = ""
        self.display_q = Queue()
        self.sd_card = False
        self.check_sd_card()
        self.lsl = False
        self.save_local = True
        self.display_rate = 0
        self.last_time_display = 0.0
        self.selected_channel = 'Channel 2'
        self.display_data = 'Raw'
        self.disk_str = f"Disk Usage:"
        self.stim_delay = 0
        self.sleep_timeout = 0
        self.select_freq = 250
        self.power_line = 60
        self.persistent_file_name = STATE_PATH / "simple_gui_state.pkl"

    def save(self):
        state = {
            "run_dict": self.run_dict,
            "lsl": self.lsl,
            "save_local": self.save_local,
            "selected_channel": self.selected_channel,
            "display_data": self.display_data,
            "stim_delay": self.stim_delay,
            "sleep_timeout": self.sleep_timeout,
            "select_freq": self.select_freq,
            "power_line": self.power_line,
            "pipeline_key": self.pipeline_key
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
            self.stim_delay = state["stim_delay"]
            self.sleep_timeout = state["sleep_timeout"]
            self.select_freq = state["select_freq"]
            self.power_line = state["power_line"]

            self.selected_channel = state["selected_channel"]
            if state["pipeline_key"] in self.pipeline_keys:
                self.pipeline_key = state["pipeline_key"]

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

        if self.stim_delay != 0:
            self.run_dict['stim_delay'] = int(self.stim_delay) / 1000

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
                                  args=(self._pipelines[self.pipeline_key]["processor"],
                                        self._pipelines[self.pipeline_key]["detector"],
                                        self._pipelines[self.pipeline_key]["stimulator"],
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


class SimpleUI:
    def __init__(self, pipelines=PIPELINES):
        self._pipelines = pipelines
    
    def run(self,
            host='192.168.4.1',
            port=8081,
            title='Portiloop Control Center',
            dark=True,
            favicon='🧠',
            reload=False):

        exp_state = ExperimentState(pipelines=self._pipelines)
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
            stimulator = self._pipelines[exp_state.pipeline_key]["stimulator"](RUN_SETTINGS)
            stimulator.test_stimulus()
            del stimulator

        def update_line_plot():
            now = datetime.now()
            x = now.timestamp()
            try:
                # empty the queue
                x = []
                y = []
                while not exp_state.display_q.empty():
                    channel = int(exp_state.selected_channel[-1]) - 1
                    point = exp_state.display_q.get(block=False)
                    time, raw_point, filtered_point = point
                    x.append(time)
                    if exp_state.display_data == 'Raw':
                        point = raw_point[0][channel]
                    elif exp_state.display_data == 'Filter':
                        point = filtered_point[0][channel]
                    else:
                        point = 0.0
                    y.append(point)
            except Exception as e:
                print(f"Caught exception: {e}")

            if len(x) > 0 and len(y) > 0:
                line_plot.push(x, [y])

        def disable_stim_toggle_callback(caller):
            stim_toggle.enable()

        ui.label('Portiloop 🧠').classes('text-4xl font-mono')
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
                    sd_card_timer = ui.timer(0.5, exp_state.check_sd_card)
                    start_button.bind_enabled_to(timer, 'active', forward=lambda x: not x)

            ############### Output Tab ####################
            with ui.tab_panel(output_tab).classes('w-full items-center'):
                ############# Line Plot stuff ################
                line_timer = ui.timer(1/25, update_line_plot, active=False)
                start_button.bind_enabled_to(line_timer, 'active', forward=lambda x: not x)
                line_plot = ui.line_plot(n=1, limit=250 * 5, update_every=25, figsize=(3, 2), layout='tight')

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
                    stim_delay = ui.number(value=exp_state.stim_delay, label='Stimulation Delay (in ms)').bind_value_to(exp_state, 'stim_delay')
                    select_pipeline = ui.select(exp_state.pipeline_keys, value=exp_state.pipeline_key, on_change=disable_stim_toggle_callback, label="Pipeline").bind_value_to(exp_state, 'pipeline_key')
                    select_pipeline.classes('w-1/2')
                    start_button.bind_enabled_to(lsl_checker)
                    start_button.bind_enabled_to(save_checker)
                    start_button.bind_enabled_to(select_pipeline)
                    start_button.bind_enabled_to(stim_delay)
                    start_button.bind_enabled_to(select_freq)
                    start_button.bind_enabled_to(select_notch)
                    start_button.bind_enabled_to(sleep_timeout)
                    start_button.bind_enabled_to(sleep_timeout_timer, 'active', forward=lambda x: not x)

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
    gui = SimpleUI()
    gui.run()
