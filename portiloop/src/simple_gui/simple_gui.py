from multiprocessing import Process, Queue, Value
import time
from nicegui import ui
from nicegui.events import ValueChangeEventArguments
from datetime import datetime
from portiloop.src.capture import start_capture
from portiloop.src.detection import SleepSpindleRealTimeDetector
from portiloop.src.stimulation import SleepSpindleRealTimeStimulator, AlternatingStimulator
from portiloop.src.hardware.leds import Color, LEDs
from portiloop.src.config.constants import RUN_SETTINGS, version, nb_channels
from portiloop.src.utils import DummyAlsaMixer
import os
import socket
import alsaaudio
from alsaaudio import ALSAAudioError
import psutil

WORKSPACE_DIR_SD = "/media/sd_card/workspace/edf_recordings/"
WORKSPACE_DIR_IN = "/home/mendel/workspace/edf_recordings/"

portiloop_ID = socket.gethostname()


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
        self.selected_channel = 'Channel 2'
        self.display_data = 'Raw'
        self.disk_str = f"Disk Usage:"
        self.stim_delay = 0
        self.sleep_timeout = 0
        self.select_freq = 250

    def start(self):
        print(f"Frequency: {self.select_freq}, Sleep_timeout: {self.sleep_timeout}")
        # Set the variables for the experiment
        self.time_started = datetime.now()
        stim_str = "STIMON" if self.stim_on else "STIMOFF"
        time_str = self.time_started.strftime('%Y-%m-%d_%H-%M-%S')
        self.exp_name = f"{portiloop_ID}_{time_str}_{stim_str}.csv"
        print(f"Starting recording {self.exp_name.split('.')[0]}")

        print(f"STIMON = {self.stim_on}, STIMTYPE = {self.stimulator_type}")

        self.run_dict['frequency'] = self.select_freq

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

        if self.stimulator_type == 'Spindle':
            self.stimulator_cls = SleepSpindleRealTimeStimulator
            if self.stim_delay != 0:
                self.run_dict['stim_delay'] = int(self.stim_delay) / 1000
        elif self.stimulator_type == 'Interval':
            self.stimulator_cls = AlternatingStimulator
            self.run_dict['detect'] = False

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
        if self.sd_card:
            self.disk_str = f"Disk Usage: {psutil.disk_usage(os.path.abspath('/media/sd_card/')).percent}%"
        else:
            self.disk_str = f"Disk Usage: {psutil.disk_usage(os.getcwd()).percent}%"
        
    def check_sleep_timeout(self):
        if self.pause_value.value:
            current_time = time.time()
            if current_time > self.time_unpause:
                self.pause_value.value = False

exp_state = ExperimentState()

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
    except Exception:
        print("AAAAAAAAAAAAAAAAAAH")

    if len(x) > 0 and len(y) > 0:
        line_plot.push(x, [y])

def disable_stim_toggle_callback(caller):
    if caller.value == 'Interval':
        stim_toggle.disable()
        stim_toggle.value = 'Stim On'
        exp_state.stim_on = True
    else:
        stim_toggle.enable()

ui.label('Portiloop 🧠').classes('text-4xl font-mono')
ui.label('Control Center').classes('text-2xl font-mono')

ui.html(f"Connected to: <strong>{portiloop_ID}</strong> (v{version} - {nb_channels} channels)")
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
        line_plot = ui.line_plot(n=1, limit=250 * 5, update_every=25, figsize=(3, 2))

        ui.separator()
        ############# Display Control ###############
        with ui.column().classes('w-full items-center'):
            available_channels = [f"Channel {i+1}" for i in range(nb_channels)]
            select_channel_display = ui.select(available_channels, value=available_channels[1], label="Display Channel")
            select_channel_display.bind_value_to(exp_state, 'selected_channel').classes('w-1/2')

            filtered_toggle = ui.toggle(['Raw', 'Filter'], value='Raw')
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
                value=250, 
                label="Sample Frequency (Hz)").bind_value_to(exp_state, 'select_freq').classes('w-3/4')
            ui.separator().classes('w-2/3')
            sleep_timeout = ui.slider(min=0, max=40, value=0).bind_value_to(exp_state, 'sleep_timeout').classes('w-3/4') #.props('label-always')
            ui.label().bind_text_from(sleep_timeout, 'value', backward=lambda x: f"Sleep Timeout: {x} minutes")
            sleep_timeout_timer = ui.timer(10, exp_state.check_sleep_timeout)
            ui.separator().classes('w-2/3')
            lsl_checker = ui.checkbox('Stream LSL').bind_value_to(exp_state, 'lsl')
            save_checker = ui.checkbox('Save Recording Locally', value=True).bind_value_to(exp_state, 'save_local')
            stim_delay = ui.number(value=0, label='Stimulation Delay (in ms)').bind_value_to(exp_state, 'stim_delay')
            select_stimulator = ui.select(['Spindle', 'Interval'], value='Spindle', on_change=disable_stim_toggle_callback, label="Stimulator").bind_value_to(exp_state, 'stimulator_type')
            select_stimulator.classes('w-1/2')
            start_button.bind_enabled_to(lsl_checker)
            start_button.bind_enabled_to(save_checker)
            start_button.bind_enabled_to(select_stimulator)
            start_button.bind_enabled_to(stim_delay)
            start_button.bind_enabled_to(select_freq)
            start_button.bind_enabled_to(sleep_timeout)
            start_button.bind_enabled_to(sleep_timeout_timer, 'active', forward=lambda x: not x)

line_plot.bind_visibility_from(start_button, 'enabled', backward=lambda x: not x)

ui.run(
    host='192.168.4.1', 
    port=8081,
    title='Portiloop Control Center',
    dark=True,
    favicon='🧠',
    reload=False
    )
