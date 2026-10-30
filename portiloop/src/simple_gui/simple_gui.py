import asyncio
import logging
from multiprocessing import Process, Queue, Value
import time
import os
import socket
import subprocess
import threading
from datetime import datetime
import json
from pathlib import Path

import alsaaudio
from alsaaudio import ALSAAudioError
import psutil
from nicegui import app, ui

from portiloop import __version__
from portiloop.src.core.utils import DummyAlsaMixer, get_hardware_version
from portiloop.src.core.hardware.backend import Backend

# The ADS1299 must be reset after power-up, and the Simple GUI starts at boot, before any recording.
# Reset it before importing constants, which reads the number of channels from the ADS.
_backend = Backend(get_hardware_version())
try:
    _backend.reset()
finally:
    _backend.close()
del _backend

from portiloop.src.core.capture import start_capture
from portiloop.src.core.hardware.config_hardware import to_ads_frequency
from portiloop.src.core.hardware.impedance import ImpedanceMeter, results_to_metadata, NOT_CHECKED
from portiloop.src.core.impedance_display import results_html
from portiloop.src.core.constants import SD_CARD, sd_card_detected, csv_path, STATE_PATH, NB_CHANNELS, DEFAULT_CONFIG_DICT
from portiloop.src.custom.config import RUN_SETTINGS
from portiloop.src.custom.custom_pipelines import PIPELINES

logger = logging.getLogger(__name__)

portiloop_ID = socket.gethostname()


ENABLE_DISPLAY = True
LINE_PLOT_WINDOW = 5  # (window in seconds)
LINE_PLOT_UPDATE_EVERY = 2  # plot every N x TIMER_READ_DISPLAY_QUEUE s
LINE_PLOT_FIGSIZE = (3, 2)
LINE_PLOT_STRIDE = 4  # plot only 1 in N datapoints
TIMER_READ_DISPLAY_QUEUE = 1.0

TIMER_SD_CARD = 5.0
TIMER_CHECK_CAPTURE = 2.0
TIMER_IMPEDANCE_DISPLAY = 0.5


PERSISTED_FIELDS = (
    "lsl",
    "save_local",
    "selected_channel",
    "display_data",
    "min_delay",
    "max_delay",
    "inter_stim_delay",
    "sleep_timeout",
    "volume",
    "select_freq",
    "power_line",
    "channel_detection",
    "pipeline_key",
    "custom_exp_name",
    "stim_on",
    "python_clock",
)


class ExperimentState:
    def __init__(self, pipelines=PIPELINES):
        self._pipelines = pipelines

        # Impedance check (the last results are saved in the metadata of the next recordings)
        self.impedance_results = None  # {channel (0-indexed): ImpedanceResult} of the last check, None if no check
        self.impedance_time = None  # datetime of the last check
        self.impedance_error = None
        self.impedance_version = 0  # incremented at each new result or error, to refresh the display
        self._impedance_thread = None
        self._impedance_stop = threading.Event()

        self.pipeline_keys = list(self._pipelines.keys())
        self.pipeline_key = self.pipeline_keys[0]

        self.preset_keys = self._get_presets()
        self.preset_key = None
        self.new_preset_name = ""

        self.point_index = 0
        self.len_plot = int(RUN_SETTINGS['frequency'] * LINE_PLOT_WINDOW / LINE_PLOT_STRIDE)
        self.started = False
        self.time_started = datetime.now()
        self.q_msg = Queue()

        self.pause_value = Value('b', False)
        self._t_capture = None
        self.stim_on = False
        self.custom_exp_name = ""
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
        self.max_delay = 0
        self.inter_stim_delay = 0
        self.sleep_timeout = 0
        self.select_freq = 250
        self.power_line = 60
        self.channel_detection = DEFAULT_CONFIG_DICT.get('channel_detection', 6)
        self.python_clock = RUN_SETTINGS['python_clock']
        self.persistent_file_name = STATE_PATH / "Previous run.json"

        self._init_mixer()
        try:
            self.volume = int(self.mixer.getvolume()[0])
        except Exception:
            self.volume = DEFAULT_CONFIG_DICT.get('volume', 100)

        self.run_dict = self._build_run_dict_from_ui_state()

    def _init_mixer(self):
        try:
            mixers = alsaaudio.mixers()
            if len(mixers) <= 0:
                logger.warning("No ALSA mixer found.")
                self.mixer = DummyAlsaMixer()
            else:
                self.mixer = alsaaudio.Mixer(control='SoftMaster', device='dmixer')
        except Exception as e:
            logger.warning("No ALSA mixer found. Volume control will not be available. (%s)", e)
            self.mixer = DummyAlsaMixer()

    def set_volume(self, val):
        self.volume = int(val)
        if not hasattr(self, 'mixer') or self.mixer is None:
            self._init_mixer()
        try:
            self.mixer.setvolume(self.volume)
        except Exception as e:
            logger.warning("Failed to set ALSA volume: %s", e)

    def _get_presets(self):
        return sorted(p.stem for p in STATE_PATH.glob("*.json"))

    def refresh_presets(self):
        self.preset_keys = self._get_presets()

    def save_preset(self, preset_name: str):
        if not preset_name:
            logger.warning("Preset name cannot be empty")
            return

        filepath = STATE_PATH / f"{preset_name}.json"
        self.save(filepath=filepath)
        self.refresh_presets()
        self.preset_key = preset_name

    def _channel_states(self):
        """
        Enables all the channels, except the electrodes that were OFF at the last impedance check (powered down, they
        record 0). The detection channel is always enabled: if its electrode is off, lead-off detection pauses the
        stimulation. If all the electrodes were OFF, the bias electrode was probably off and all channels are enabled.
        """
        states = ["simple"] * NB_CHANNELS
        results = self.impedance_results or {}
        if results and not all(r.off for r in results.values()):
            for ch, r in results.items():
                if r.off and ch + 1 != int(self.channel_detection):
                    states[ch] = "disabled"
        return states

    def _build_run_dict_from_ui_state(self):
        run_dict = RUN_SETTINGS
        run_dict["channel_states"] = self._channel_states()
        run_dict['frequency'] = self.select_freq
        run_dict["filter_settings"]["power_line"] = self.power_line
        run_dict['channel_detection'] = int(self.channel_detection)
        run_dict['pipeline_name'] = self.pipeline_key
        run_dict['pipeline'] = self.pipeline_key
        run_dict['sleep_timeout'] = self.sleep_timeout
        run_dict['volume'] = int(self.volume)
        run_dict['stimulate'] = self.stim_on
        run_dict['python_clock'] = self.python_clock
        run_dict['lsl'] = self.lsl
        run_dict['record'] = self.save_local
        run_dict['min_delay'] = int(self.min_delay) / 1000 if self.min_delay is not None else 0
        run_dict['max_delay'] = int(self.max_delay) / 1000 if self.max_delay is not None else 0
        run_dict['inter_stim_delay'] = int(self.inter_stim_delay) / 1000 if self.inter_stim_delay is not None else 0
        run_dict['impedances_kohm'] = results_to_metadata(self.impedance_results, NB_CHANNELS)
        run_dict['impedance_check_time'] = self.impedance_time.isoformat(timespec='seconds') if self.impedance_time else NOT_CHECKED
        return run_dict

    def save(self, filepath: Path = None):
        if filepath is None:
            filepath = self.persistent_file_name

        state = {
            "software_version": __version__,
            "nb_channels": NB_CHANNELS,
        }
        for field in PERSISTED_FIELDS:
            state[field] = getattr(self, field)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load(self, filepath: Path = None):

        # if filepath is None, load from the default path:
        if filepath is None:
            filepath = self.persistent_file_name

        # check that the file exists:
        if filepath.is_file():

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    state = json.load(f)

                if state.get("nb_channels") != NB_CHANNELS or state.get("software_version") != __version__:
                    return

                # Populate self:
                for field in PERSISTED_FIELDS:
                    if field in state:
                        setattr(self, field, state[field])
                    else:
                        logger.warning("Missing field in saved GUI state: %s", field)

                # Check that pipeline_key is valid:
                if self.pipeline_key not in self.pipeline_keys:
                    self.pipeline_key = self.pipeline_keys[0]

                if hasattr(self, 'volume'):
                    self.set_volume(self.volume)

            except Exception as e:
                logger.exception("Caught exception while loading app state")

            self.run_dict = self._build_run_dict_from_ui_state()

    def impedance_check_running(self):
        return self._impedance_thread is not None and self._impedance_thread.is_alive()

    def start_impedance_check(self, live):
        """
        Measures the electrode impedances in a background thread, once or until stop_impedance_check (live).
        """
        if self.impedance_check_running() or self._t_capture is not None:
            return
        self._impedance_stop.clear()
        self.impedance_error = None
        self._impedance_thread = threading.Thread(target=self._impedance_worker, args=(live,), daemon=True)
        self._impedance_thread.start()

    def stop_impedance_check(self):
        """
        Stops the impedance check and waits until the ADS1299 is released.
        """
        self._impedance_stop.set()
        if self._impedance_thread is not None:
            self._impedance_thread.join()
            self._impedance_thread = None

    def _impedance_worker(self, live):
        backend = None
        meter = None
        try:
            backend = Backend(get_hardware_version())
            meter = ImpedanceMeter(backend, NB_CHANNELS, vref=RUN_SETTINGS['vref'])
            meter.configure(range(NB_CHANNELS))
            while not self._impedance_stop.is_set():
                results = meter.measure_electrodes(stop_event=self._impedance_stop)
                if results is None:  # interrupted
                    break
                self.impedance_results = results
                self.impedance_time = datetime.now()
                self.impedance_version += 1
                if not live:
                    break
        except Exception as e:
            logger.exception("Impedance check failed.")
            self.impedance_error = str(e)
            self.impedance_version += 1
        finally:
            try:
                if meter is not None:
                    meter.close()
            finally:
                if backend is not None:
                    backend.close()

    def start(self):
        # The impedance check and the recording cannot use the ADS1299 at the same time
        self.stop_impedance_check()

        # Clean up possible messages from previous runs
        while not self.q_msg.empty():
            self.q_msg.get()

        if not self.python_clock:  # ADS clock: force the frequency to an ADS-compatible frequency
            self.select_freq = to_ads_frequency(self.select_freq)

        self.run_dict = self._build_run_dict_from_ui_state()
        self.save()

        # Set the variables for the experiment
        self.time_started = datetime.now()
        stim_str = "CONDB" if self.stim_on else "CONDA"
        time_str = self.time_started.strftime('%Y-%m-%d_%H-%M-%S')
        prefix = self.custom_exp_name or portiloop_ID
        self.exp_name = f"{prefix}_{time_str}_{stim_str}.csv"
        logger.info("Starting recording %s", self.exp_name.split('.')[0])
        logger.info("STIMON = %s", self.stim_on)

        self.point_index = 0
        self.len_plot = int(self.run_dict['frequency'] * LINE_PLOT_WINDOW / LINE_PLOT_STRIDE)

        # Calculating how much time to pause in seconds
        if self.sleep_timeout > 0:
            self.time_unpause = self.time_started.timestamp() + self.sleep_timeout * 60
            self.pause_value.value = True
            logger.info("Currently: %s, Pausing until: %s", self.time_started.timestamp(), self.time_unpause)
        else:
            self.pause_value.value = False

        self.set_volume(self.volume)
        self.run_dict['volume'] = int(self.volume)

        workspace_dir = csv_path()  # the SD card may have been inserted since the start of the GUI
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
        logger.info("PID start process: %s. Kill this process if program crashes before end of execution.", self._t_capture.pid)

    def stop(self):
        logger.info("Stopping recording...")
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
        logger.info("Done.")

    def capture_exitcode(self):
        """
        Returns the exit code of the recording process if it has ended, None if it is running or was not started.
        """
        if self._t_capture is None or self._t_capture.is_alive():
            return None
        return self._t_capture.exitcode

    def toggle_stim(self):
        self.stim_on = not self.stim_on

    def check_sd_card(self):
        if sd_card_detected():
            self.sd_card = True
            self.disk_str = f"Disk Usage: {psutil.disk_usage(str(SD_CARD)).percent}%"
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
            logger.warning("Caught exception while loading persistent state: %s", e)

        def start():
            exp_state.start()
            start_button.enabled = False

        def stop():
            exp_state.stop()
            start_button.enabled = True

        def check_capture_ended():
            # Reset the GUI when the recording ends on its own (end of duration or crash)
            exitcode = exp_state.capture_exitcode()
            if exitcode is None:
                return
            if exitcode != 0:
                logger.error("Recording process crashed (exit code %s).", exitcode)
            else:
                logger.info("Recording ended.")
            stop()

        def test_sound():
            exp_state.set_volume(exp_state.volume)
            stimulator_cls = self._pipelines[exp_state.pipeline_key]["stimulator"]
            if stimulator_cls is not None:
                stimulator = self._pipelines[exp_state.pipeline_key]["stimulator"](RUN_SETTINGS)
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
                    channel = int(exp_state.selected_channel.split()[-1]) - 1
                    point = exp_state.display_q.get(block=False)
                    time_value, raw_point, filtered_point = point
                    if exp_state.display_data == 'Raw':
                        point = raw_point[0][channel]
                    elif exp_state.display_data == 'Filter':
                        point = filtered_point[0][channel]
                    else:
                        point = 0.0
                    exp_state.point_index += 1
                    if exp_state.point_index % LINE_PLOT_STRIDE == 0:
                        x.append(time_value)
                        y.append(point)
            except Exception as e:
                logger.error("Caught exception: %s", e)

            # update the actual plot 
            if len(x) > 0 and len(y) > 0:
                line_plot.push(x, [y])

        def add_preset():
            try:
                exp_state.save_preset(exp_state.new_preset_name)
                select_preset.options = exp_state.preset_keys
                select_preset.update()
                logger.info("Saved preset")
            except Exception as e:
                logger.warning("Caught exception while saving preset: %s", e)

        def disable_stim_toggle_callback(caller):
            stim_toggle.enable()

        def preset_callback(caller):
            if exp_state.preset_key is not None:
                try:
                    filepath = STATE_PATH / f"{exp_state.preset_key}.json"
                    exp_state.load(filepath=filepath)  # load persistent state
                    logger.info("Loaded preset %s", filepath)
                except Exception as e:
                    logger.warning("Caught exception while loading preset: %s", e)
            else:
                logger.warning("Attempted to load a preset but preset_key is None")

        async def sync_clock(client=None, force=False):
            # Avoid changing the system clock during an active recording to prevent timestamp jumps
            if exp_state._t_capture is not None and exp_state._t_capture.is_alive():
                logger.info("Recording in progress, skipping clock synchronization.")
                return

            try:
                if client is not None and hasattr(client, 'connected'):
                    try:
                        await asyncio.wait_for(client.connected(), timeout=5.0)
                    except Exception:
                        pass

                if client is not None and hasattr(client, 'run_javascript'):
                    browser_time = await client.run_javascript('Date.now() / 1000', timeout=5.0)
                else:
                    browser_time = await ui.run_javascript('Date.now() / 1000', timeout=5.0)

                if browser_time is None:
                    return

                browser_time = float(browser_time)
                device_time = time.time()
                diff = browser_time - device_time

                # If the difference is negligible (< 1.5s) and not forced, consider it already in sync
                if not force and abs(diff) < 1.5:
                    logger.debug("Portiloop clock is already in sync with browser (delta: %.2fs).", diff)
                    return

                logger.info("Synchronizing Portiloop clock with browser (delta: %.2fs)...", diff)
                res = subprocess.run(
                    ['sudo', 'date', '-u', '-s', f'@{browser_time:.3f}'],
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info("System clock set via date: %s", res.stdout.strip())

                # Persist to hardware RTC if available
                try:
                    subprocess.run(['sudo', 'hwclock', '-w'], check=False)
                except Exception:
                    pass

                logger.info("System clock successfully synchronized to: %s", datetime.now())
                msg = f"Portiloop clock synchronized with browser ({datetime.now().strftime('%H:%M:%S')})"
                if client is not None:
                    try:
                        with client:
                            ui.notify(msg, type='positive')
                    except Exception:
                        ui.notify(msg, type='positive')
                else:
                    ui.notify(msg, type='positive')
            except Exception as e:
                err = e.stderr.strip() if hasattr(e, 'stderr') and e.stderr else str(e)
                logger.warning("Could not synchronize clock with browser: %s", err)
                if force:
                    ui.notify(f"Failed to sync clock: {err}", type='warning')

        app.on_connect(sync_clock)
        app.on_connect(exp_state.check_sd_card)  # the SD card may have been inserted since the last visit

        async def manual_sync():
            await sync_clock(force=True)

        def toggle_live_impedance(e):
            if e.value:
                exp_state.start_impedance_check(live=True)
            else:
                exp_state.stop_impedance_check()

        displayed_impedance_version = [-1]

        def refresh_impedance():
            recording = exp_state._t_capture is not None
            running = exp_state.impedance_check_running()
            check_button.enabled = not (recording or running)
            live_switch.enabled = not recording
            if live_switch.value and not running:  # stopped by a recording or an error
                live_switch.value = False
            if displayed_impedance_version[0] != exp_state.impedance_version:
                displayed_impedance_version[0] = exp_state.impedance_version
                if exp_state.impedance_results is not None:
                    impedance_html.content = results_html(exp_state.impedance_results)
            if recording:
                status = "Unavailable during a recording."
            elif exp_state.impedance_error is not None:
                status = f"Error: {exp_state.impedance_error}"
            elif running:
                status = "Measuring..."
            elif exp_state.impedance_time is not None:
                status = f"Last check: {exp_state.impedance_time.strftime('%H:%M:%S')} (saved with the next recording)"
            else:
                status = "No check performed yet."
            impedance_status.set_text(status)

        ui.label('Portiloop 🧠').classes('text-4xl font-mono')
        ui.label('Control Center').classes('text-2xl font-mono')

        ui.html(f"Connected to: <strong>{portiloop_ID}</strong> (v{RUN_SETTINGS['hardware_version']} - {RUN_SETTINGS['nb_channels']} channels)")
        ui.separator()

        with ui.tabs(on_change=lambda _: exp_state.check_sd_card()).classes('w-full') as tabs:
            control_tab = ui.tab('Control', icon='home')
            check_tab = ui.tab('Check', icon='Ω')  # not in Material Icons: rendered with a fallback font
            output_tab = ui.tab('Output', icon='timeline')
            advanced_tab = ui.tab('Advanced', icon='settings')

        with ui.tab_panels(tabs, value=control_tab).classes('w-full'):
            ############### First Tab ##################
            with ui.tab_panel(control_tab).classes('w-full items-center'):
                ################ Simple Options ################
                with ui.column().classes('w-full items-center'):
                    sd_card_checker = ui.checkbox('SD Card').bind_value_from(exp_state, 'sd_card').disable()

                    test_sound_button = ui.button('Test Sound 🔊', on_click=test_sound)

                    stim_toggle = ui.toggle(['Cond A', 'Cond B'], value='Cond B' if exp_state.stim_on else 'Cond A').bind_value(
                        exp_state, 'stim_on',
                        forward=lambda x: x == 'Cond B',
                        backward=lambda x: 'Cond B' if x else 'Cond A'
                    )

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
                        backward=lambda x: f"Current experiment: {x.split('.')[0]}")
                    timer = ui.timer(1.0, lambda: time_label.set_text(f'Timer: {str(datetime.now() - exp_state.time_started).split(".")[0]}'))
                    sd_card_timer = ui.timer(TIMER_SD_CARD, exp_state.check_sd_card)
                    capture_timer = ui.timer(TIMER_CHECK_CAPTURE, check_capture_ended)
                    start_button.bind_enabled_to(timer, 'active', forward=lambda x: not x)

            ############### Check Tab ####################
            with ui.tab_panel(check_tab).classes('w-full items-center'):
                with ui.column().classes('w-full items-center'):
                    ui.label('Electrode impedance check').classes('text-2xl')
                    with ui.row().classes('items-center'):
                        check_button = ui.button('Check ⚡', on_click=lambda: exp_state.start_impedance_check(live=False)).classes('text-xl')
                        live_switch = ui.switch('Live', on_change=toggle_live_impedance)
                    impedance_html = ui.html('')
                    impedance_status = ui.label()
                    ui.label("chN is the electrode on pin N of the connector. "
                             "The impedance of the reference cannot be measured, only its connection. "
                             "Electrodes OFF at the last check are disabled in the next recording "
                             "(except the detection channel).").classes('text-sm')
                    ui.timer(TIMER_IMPEDANCE_DISPLAY, refresh_impedance)

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
                    select_channel_display.bind_value(exp_state, 'selected_channel').classes('w-1/2')

                    filtered_toggle = ui.toggle(['Raw', 'Filter'], value=exp_state.display_data)
                    filtered_toggle.bind_value(exp_state, 'display_data')

            ############### Advanced Tab #############
            with ui.tab_panel(advanced_tab).classes('w-full items-center'):
                ################ Advanced Options ###################
                with ui.column().classes('w-full items-center'):
                    ui.label("If you are a subject in an experiment, do not change any of these options unless explicitly prompted to!").classes('text-1.5xl').style('color:#d9a011')
                    ui.separator()
                    space_label = ui.label(f"Disk usage: {psutil.disk_usage(os.getcwd())}%").bind_text_from(
                        exp_state,
                        'disk_str'
                    ).classes('text-2xl')

                    select_preset = ui.select(exp_state.preset_keys, value=exp_state.preset_key, on_change=preset_callback, label="Preset").bind_value(exp_state, 'preset_key')
                    select_preset.classes('w-3/4')

                    select_pipeline = ui.select(exp_state.pipeline_keys, value=exp_state.pipeline_key, on_change=disable_stim_toggle_callback, label="Pipeline").bind_value(exp_state, 'pipeline_key')
                    select_pipeline.classes('w-3/4')

                    possible_freqs = [50, 100, 250, 500, 1000]
                    select_freq = ui.select(
                        possible_freqs,
                        value=exp_state.select_freq,
                        label="Sample frequency (Hz)").bind_value(exp_state, 'select_freq').classes('w-3/4')
                    ui.separator().classes('w-2/3')
                    possible_notches = [60, 50]
                    select_notch = ui.select(
                        possible_notches,
                        value=exp_state.power_line,
                        label="Notch filter frequency (Hz)").bind_value(exp_state, 'power_line').classes('w-3/4')
                    ui.separator().classes('w-2/3')
                    possible_channels = [i + 1 for i in range(RUN_SETTINGS['nb_channels'])]
                    select_channel_detect = ui.select(
                        possible_channels,
                        value=exp_state.channel_detection,
                        label="Detection channel").bind_value(exp_state, 'channel_detection').classes('w-3/4')
                    ui.separator().classes('w-2/3')
                    ui.label('Clock:')
                    clock_toggle = ui.toggle(['ADS', 'Coral'], value='Coral' if exp_state.python_clock else 'ADS').bind_value(
                        exp_state, 'python_clock',
                        forward=lambda x: x == 'Coral',
                        backward=lambda x: 'Coral' if x else 'ADS'
                    )
                    sync_clock_btn = ui.button('Sync Clock with Browser 🕒', on_click=manual_sync)
                    ui.separator().classes('w-2/3')
                    sleep_timeout = ui.slider(min=0, max=180, value=exp_state.sleep_timeout).bind_value(exp_state, 'sleep_timeout').classes('w-3/4') #.props('label-always')
                    ui.label().bind_text_from(sleep_timeout, 'value', backward=lambda x: f"Stimulation starts after: {x} minutes")
                    sleep_timeout_timer = ui.timer(10, exp_state.check_sleep_timeout)
                    ui.separator().classes('w-2/3')
                    volume_slider = ui.slider(min=0, max=100, step=1, value=exp_state.volume, on_change=lambda e: exp_state.set_volume(e.value)).bind_value(exp_state, 'volume').classes('w-3/4')
                    ui.label().bind_text_from(volume_slider, 'value', backward=lambda x: f"Sound output volume: {int(x)}%")
                    ui.separator().classes('w-2/3')
                    lsl_checker = ui.checkbox('Stream LSL', value=exp_state.lsl).bind_value(exp_state, 'lsl')
                    save_checker = ui.checkbox('Save recording locally', value=exp_state.save_local).bind_value(exp_state, 'save_local')
                    filename_box = ui.input(value='', label='Recording name').props('clearable').bind_value(exp_state, 'custom_exp_name')
                    min_delay = ui.number(value=exp_state.min_delay, label='Min stim delay (ms)').bind_value(exp_state, 'min_delay')
                    max_delay = ui.number(value=exp_state.max_delay, label='Max stim delay (ms)').bind_value(exp_state, 'max_delay')
                    inter_stim_delay = ui.number(value=exp_state.inter_stim_delay, label='Inter-stim delay (ms)').bind_value(exp_state, 'inter_stim_delay')
                    ui.separator().classes('w-2/3')
                    preset_name_box = ui.input(value=exp_state.new_preset_name, label="Preset name").props('clearable').bind_value(exp_state, 'new_preset_name')
                    preset_save_button = ui.button('Save preset', on_click=add_preset, color='primary')

                    start_button.bind_enabled_to(lsl_checker)
                    start_button.bind_enabled_to(save_checker)
                    start_button.bind_enabled_to(select_preset)
                    start_button.bind_enabled_to(preset_name_box)
                    start_button.bind_enabled_to(preset_save_button)
                    start_button.bind_enabled_to(select_pipeline)
                    start_button.bind_enabled_to(min_delay)
                    start_button.bind_enabled_to(max_delay)
                    start_button.bind_enabled_to(inter_stim_delay)
                    start_button.bind_enabled_to(filename_box)
                    start_button.bind_enabled_to(select_freq)
                    start_button.bind_enabled_to(select_notch)
                    start_button.bind_enabled_to(select_channel_detect)
                    start_button.bind_enabled_to(clock_toggle)
                    start_button.bind_enabled_to(sync_clock_btn)
                    start_button.bind_enabled_to(sleep_timeout)
                    start_button.bind_enabled_to(sleep_timeout_timer, 'active', forward=lambda x: not x)
                    start_button.bind_enabled_to(volume_slider)

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
    try:
        gui = SimpleUI()
        gui.run()
    except Exception:
        logger.exception("Simple GUI crashed.")
        raise
