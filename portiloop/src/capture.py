from abc import ABC, abstractmethod
import json
from multiprocessing import Process, Queue, Value
import queue  # for exceptions
import os
from time import sleep
import time
import numpy as np
from copy import deepcopy
from datetime import datetime
import warnings
from threading import Thread, Lock
from portiloop.src import ADS
from portiloop.src.hardware.leds import Color, LEDs
from portiloop.src.detection import Detector
from portiloop.src.stimulation import Stimulator
from portiloop.src.delayers import TimingDelayer, UpStateDelayer, SOPhaseDelayer
from portiloop.src.processing import BaseFilter

if ADS:
    import alsaaudio
    from alsaaudio import ALSAAudioError
    from portiloop.src.hardware.frontend import Frontend
print("ADS")

from portiloop.src.config.config_hardware import mod_config, LEADOFF_CONFIG, FRONTEND_CONFIG, to_ads_frequency
from portiloop.src.config.constants import CSV_PATH, RECORDING_PATH, CALIBRATION_PATH
from portiloop.src.utils import ADSFrontend, Dummy, FileFrontend, LSLStreamer, LiveDisplay, DummyAlsaMixer, CSVRecorder, get_portiloop_version
from portiloop.src.config.constants import RUN_SETTINGS

from IPython.display import clear_output, display
import ipywidgets as widgets
import socket
from pathlib import Path


PORTILOOP_ID = f"{socket.gethostname()}-portiloop"


def capture_process(
    p_data_o, p_msg_io, duration, frequency, python_clock, time_msg_in, channel_states
):
    """
    Args:
        p_data_o: multiprocessing.Pipe: captured datapoints are put here
        p_msg_io: mutliprocessing.Pipe: to communicate with the parent process
        duration: float: max duration of the experiment in seconds
        frequency: float: sampling frequency
        ptyhon_clock: bool: if True, the Coral clock is used, otherwise, the ADS interrupts are used
        time_msg_in: float: min time between attempts to recv incomming messages
    """
    if duration <= 0:
        duration = np.inf

    sample_time = 1 / frequency

    version = get_portiloop_version()
    frontend = Frontend(version)

    try:
        config = FRONTEND_CONFIG
        if python_clock:  # set ADS to 2 * frequency
            datarate = 2 * frequency
        else:  # set ADS to frequency
            datarate = frequency
        config = mod_config(config, datarate, channel_states)

        frontend.write_regs(0x00, config)
        # data = frontend.read_regs(0x00, len(config))

        c = True

        it = 0
        t_start = time.time()
        t_max = t_start + duration
        t = t_start

        # first sample:
        reading = frontend.read()
        datapoint = reading.channels()
        p_data_o.send(datapoint)

        t_next = t + sample_time
        t_chk_msg = t + time_msg_in

        # sampling loop:
        while c and t < t_max:
            t = time.time()
            if python_clock:
                if t <= t_next:
                    time.sleep(t_next - t)
                t_next += sample_time
                reading = frontend.read()
            else:
                reading = frontend.wait_new_data()
            datapoint = reading.channels()
            p_data_o.send(datapoint)

            # Check for messages
            if t >= t_chk_msg:
                t_chk_msg = t + time_msg_in
                if p_msg_io.poll():
                    message = p_msg_io.recv()
                    if message == "STOP":
                        c = False
            it += 1
        t = time.time()
        tot = (t - t_start) / it

        p_msg_io.send(("PRT", f"Average frequency: {1 / tot} Hz for {it} samples"))

    finally:
        p_msg_io.send("STOP")
        p_msg_io.close()
        p_data_o.close()


def start_capture(
    detector_type,
    stimulator_type,
    capture_dictionary,
    q_msg,
    q_display,
    pause_value,
):
    # print(f"DEBUG: Channel states: {capture_dictionary['channel_states']}")
    detector_cls = Detector.get_detector(detector_type)
    stimulator_cls = Stimulator.get_stimulator(stimulator_type)

    # Initialize the LED
    leds = LEDs()
    if capture_dictionary["stimulate"]:
        leds.led1(Color.CYAN)
    else:
        leds.led1(Color.PURPLE)

    print(capture_dictionary)
    # Initialize data frontend
    fake_filename = RECORDING_PATH / capture_dictionary["fake_filename"]
    capture_frontend = ADSFrontend(
        duration=capture_dictionary['duration'],
        frequency=capture_dictionary['frequency'],
        python_clock=capture_dictionary['python_clock'],
        channel_states=capture_dictionary['channel_states'],
        vref=capture_dictionary['vref'],
        process=capture_process,
    ) if capture_dictionary['signal_input'] == "ADS" else FileFrontend(
        fake_filename, 
        capture_dictionary['nb_channels'],
        capture_dictionary['channel_detection'],
    )

    # Initialize detector, LSL streamer and stimulatorif requested
    detector = (
        Detector.get_detector(detector_type)(
            threshold=capture_dictionary["threshold"],
            channel=capture_dictionary["channel_detection"],
        )
        if capture_dictionary["detect"] and detector_type
        else None
    )
    streams = {
            'filtered': filter,
            'markers': detector is not None,
        }

    lsl_streamer = LSLStreamer(streams, capture_dictionary['nb_channels'], capture_dictionary['frequency'], id=PORTILOOP_ID) if capture_dictionary['lsl'] else Dummy()
    stimulator = stimulator_cls(soundname=capture_dictionary['detection_sound'], lsl_streamer=lsl_streamer,sham=not capture_dictionary['stimulate']) if stimulator_cls is not None else None
    # Initialize filtering pipeline
    if capture_dictionary['filter']:
        fp = BaseFilter.get_filter(detector_type or 'Spindle')(
            nb_channels=capture_dictionary["nb_channels"],
            sampling_rate=capture_dictionary["frequency"],
            power_line_fq=capture_dictionary["filter_settings"]["power_line"],
            use_custom_fir=capture_dictionary["filter_settings"]["custom_fir"],
            custom_fir_order=capture_dictionary["filter_settings"]["custom_fir_order"],
            custom_fir_cutoff=capture_dictionary["filter_settings"][
                "custom_fir_cutoff"
            ],
            alpha_avg=capture_dictionary["filter_settings"]["polyak_mean"],
            alpha_std=capture_dictionary["filter_settings"]["polyak_std"],
            epsilon=capture_dictionary["filter_settings"]["epsilon"],
            filter_args=capture_dictionary["filter_settings"]["filter_args"],
        )

    # Launch the capture process
    capture_frontend.init_capture()

    # Initialize display if requested
    live_disp_activated = capture_dictionary['display']
    live_disp = LiveDisplay(channel_names=capture_dictionary['signal_labels'], window_len=capture_dictionary['width_display']) if live_disp_activated else Dummy()

    # Initialize recording if requested
    recorder = CSVRecorder(capture_dictionary['filename']) if capture_dictionary['record'] else Dummy()

    # Buffer used for the visualization and the recording
    buffer = []
    detection_buffer = []

    # Initialize stimulation delayer if requested
    delay = not (
        (capture_dictionary["stim_delay"] == 0.0)
        and (capture_dictionary["inter_stim_delay"] == 0.0)
    ) and (stimulator is not None)
    delay_phase = (
        (not delay)
        and (not capture_dictionary["spindle_detection_mode"] == "Fast")
        and (stimulator is not None)
    )
    if delay:
        stimulation_delayer = TimingDelayer(
            stimulation_delay=capture_dictionary["stim_delay"],
            inter_stim_delay=capture_dictionary["inter_stim_delay"],
        )
    elif delay_phase:
        stimulation_delayer = UpStateDelayer(
            capture_dictionary["frequency"],
            capture_dictionary["spindle_detection_mode"] == "Peak",
            0.3,
        )
    elif detector_type == 'SlowOscillation' and capture_dictionary['so_phase_delay']:
        stimulation_delayer = SOPhaseDelayer()
    else:
        stimulation_delayer = Dummy()

    if stimulator is not None:
        stimulator.add_delayer(stimulation_delayer)

    # Get the metadata and save it to a file
    metadata = capture_dictionary
    # Split the original path into its components
    dirname, basename = os.path.split(capture_dictionary["filename"])
    # Split the file name into its name and extension components
    name, _ = os.path.splitext(basename)
    # Define the new file name
    new_name = f"{name}_metadata.json"
    # Join the components back together into the new file path
    metadata_path = os.path.join(dirname, new_name)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # Initialize the variable to keep track of whether we are in a detection state or not for the markers
    prev_pause = pause_value.value

    if detector is not None:
        marker_str = LSLStreamer.string_for_detection_activation(prev_pause)
        lsl_streamer.push_marker(marker_str)

    start_time = time.time()
    last_time = 0

    # Main capture loop
    while True:
        # First, we send all outgoing messages to the capture process
        try:
            msg = q_msg.get_nowait()
            capture_frontend.send_msg(msg)
        except queue.Empty as e:
            pass
        except queue.ShutDown as e:
            raise e
        
        # Then, we check if we have received a message from the capture process
        msg = capture_frontend.get_msg()
        # Either we have received a stop message, or a print message.
        if msg is None:
            pass
        elif msg == "STOP":
            break
        elif msg[0] == "PRT":
            print(msg[1])

        # Then, we retrieve the data from the capture process
        raw_point = capture_frontend.get_data()
        # If we have no data, we continue to the next iteration
        if raw_point is None:
            continue

        # Go through filtering pipeline
        if filter:
            filtered_point = fp.filter(deepcopy(raw_point))
        else:
            filtered_point = deepcopy(raw_point)
        # print(f'main loop: filtered point shape; {filtered_point.shape}')
        # Contains the filtered point (if filtering is off, contains a copy of the raw point)
        filtered_point = filtered_point.tolist()
        raw_point = raw_point.tolist()

        # Send both raw and filtered points over LSL
        lsl_streamer.push_raw(raw_point[-1])
        if filter:
            lsl_streamer.push_filtered(filtered_point[-1])

        # Check if detection is on or off
        pause = pause_value.value

        # If the state has changed since last iteration, we send a marker
        if pause != prev_pause and detector is not None:
            lsl_streamer.push_marker(LSLStreamer.string_for_detection_activation(pause))
            prev_pause = pause

        # If detection is on
        if detector is not None and not pause:
            # Detect using the latest point
            detection_signal = detector.detect(filtered_point)
            # Stimulate
            if stimulator is not None:
                stim = stimulator.stimulate(detection_signal)
                if stim is None:
                    stim = detection_signal
                if capture_dictionary['detect']:
                    detection_buffer += stim

                # Send a stimulation every second (uncomment for testing)
                # current_time = time.time()
                # if current_time - last_time >= 1.0:
                #     stimulator.stimulate([True])
                #     last_time = current_time

            # Adds point to buffer for delayed stimulation
            stimulation_delayer.step(
                filtered_point[0][capture_dictionary["channel_detection"] - 1]
            )

        # Add point to the buffer to send to viz and recorder
        buffer += raw_point

        # Adding the raw point an it's timestamp for display
        timestamp = time.time() - start_time
        if q_display is not None:
            q_display.put([timestamp, raw_point, filtered_point])

        if len(buffer) >= 50:
            live_disp.add_datapoints(buffer)
            recorder.add_recording_data(
                buffer,
                detection_buffer,
                capture_dictionary["detect"],
                capture_dictionary["stimulate"],
            )
            buffer = []
            detection_buffer = []

    # close the frontend
    leds.led1(Color.YELLOW)
    capture_frontend.close()
    leds.close()

    del recorder
    del lsl_streamer
    del stimulation_delayer
    del stimulator
    del detector

class Capture:
    def __init__(self):
        """
        params:
            detector_type (str): Name of detector from `portiloop.src.detection.Detector._registry.keys()`
            stimulator_type (str): Name of stimulator from `portiloop.src.stimulation.Stimulator._registry.keys()`        
        """
        # {now.strftime('%m_%d_%Y_%H_%M_%S')}
        self.filename = CSV_PATH / 'recording.csv'
        
        self.version = get_portiloop_version()

        # Check which version of the ADS we are in.
        if self.version != -1:
            frontend = Frontend(self.version)
            self.nb_channels = frontend.get_version()

        # print(f"DEBUG: Current hardware: ADS1299 {self.nb_channels} channels | Portiloop Version: {self.version}")

        # General default parameters
        self.frequency = 250
        self.duration = 28800
        self.power_line = 60

        # Filtering parameters
        self.polyak_mean = 0.1
        self.polyak_std = 0.001
        self.epsilon = 0.000001
        self.custom_fir = False
        self.custom_fir_order = 20
        self.custom_fir_cutoff = 30

        # Calibration parameters
        self.vref = 2.64  # FIXME: this value is a temporary fix for what seems to be a hardware bug

        # Experiment options
        self.filter = True
        self.filter_args = [True, True, True]
        self.record = False
        self.detect = False
        self.stimulate = False
        self.lsl = False
        self.display = False
        self.threshold = 0.82
        self.signal_input = "ADS"
        self.fake_filename = os.listdir(RECORDING_PATH)[-1]
        self.python_clock = True

        # Communication parameters for messages with capture 
        self._t_capture = None
        self.q_msg = Queue()
        self.pause_value = Value('b', True)

        # Channel parameters
        self.signal_labels = [f"ch{i+1}" for i in range(self.nb_channels)]
        self.channel_states = ['disabled' for _ in range(self.nb_channels)]
        self.channel_detection = 2
        self.detection_sound = self.get_capture_dictionary()['detection_sound']

        # Delayer parameters
        self.spindle_detection_mode = 'Fast'
        self.spindle_freq = 10
        self.stim_delay = 0.0
        self.inter_stim_delay = 0.0
        self.so_phase_delay = True

        # Stimulator and detector classes
        self.detector_type = 'Spindle'
        self.stimulator_type = 'Spindle'

        if ADS:
            try:
                mixers = alsaaudio.mixers()
                if len(mixers) <= 0:
                    warnings.warn(f"No ALSA mixer found.")
                    self.mixer = DummyAlsaMixer()
#                 elif 'PCM' in mixers :
#                     self.mixer = alsaaudio.Mixer(control='PCM')
                else:
                    self.mixer = alsaaudio.Mixer(control='SoftMaster', device='dmixer')
            except ALSAAudioError as e:
                print(e)
                warnings.warn(f"No ALSA mixer found. Volume control will not be available from notebook.\nAvailable mixers were:\n{mixers}")
                self.mixer = DummyAlsaMixer()
            
            self.volume = self.mixer.getvolume()[0]  # we will set the same volume on all channels
        else:
            self.mixer = DummyAlsaMixer()
            self.volume = self.mixer.getvolume()[0]
        
        # widgets ===============================
        
        # CHANNELS ------------------------------
        self.chann_buttons = []
        for i in range(self.nb_channels):
            self.chann_buttons.append(widgets.RadioButtons(
                disabled=False,
                # button_style='info',  # 'success', 'info', 'warning', 'danger' or ''
                tooltip=f'Enable channel {i+1}',
                options=['disabled', 'simple', 'bias', 'test', 'temp'],
                value='disabled',
            ))
        
        self.b_channel_detect = widgets.Dropdown(
            options=[(f'{i+1}', i+1) for i in range(self.nb_channels)],
            value=2,
            description='Detection Channel:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        
        sound_dir = Path.home() / 'portiloop-software' / 'portiloop' / 'sounds'
        options = [(sound[:-4], sound) for sound in os.listdir(sound_dir) if sound[-4:] == ".wav"]
        self.b_sound_detect = widgets.Dropdown(
            options=options,
            value="stimul_100ms.wav",
            description='Sound:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        
        self.b_accordion_channels = widgets.Accordion(
            children=[
                widgets.GridBox([widgets.Label(f"CH{i+1}") for i in range(self.nb_channels)] + self.chann_buttons, 
                                layout=widgets.Layout(grid_template_columns=f"repeat({self.nb_channels}, 90px)")
                )
            ])
        self.b_accordion_channels.set_title(index = 0, title = 'Channels')
        
        # OTHERS ------------------------------
        
        self.b_capture = widgets.ToggleButtons(
            options=['Stop', 'Start'],
            description='Capture:',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Stop capture', 'Start capture'],
        )
        
        self.b_pause = widgets.ToggleButtons(
            options=['Paused', 'Active'],
            description='Detection',
            disabled=True,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Detector and stimulator paused', 'Detector and stimulator active'],
        )
        
        self.b_clock = widgets.ToggleButtons(
            options=['ADS', 'Coral'],
            description='Clock:',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Use ADS clock (not very precise, very timely)', 'Use Coral clock (very precise, not very timely)'],
        )
        
        self.b_power_line = widgets.ToggleButtons(
            options=['60 Hz', '50 Hz'],
            description='Power line:',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['North America 60 Hz',
                      'Europe 50 Hz'],
        )

        self.b_signal_input = widgets.ToggleButtons(
            options=['ADS', 'File'],
            description='Signal Input:',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Read data from ADS.',
                      'Read data from file.'],
        )
        
        self.b_fake_filename = widgets.Dropdown(
            options=os.listdir(RECORDING_PATH),
            value=os.listdir(RECORDING_PATH)[0],
            description='Fake Filename',
            disabled=True,
            style={'description_width': 'initial'}
        )

        self.b_custom_fir = widgets.ToggleButtons(
            options=['Default', 'Custom'],
            description='FIR filter:',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Use the default 30Hz low-pass FIR from the Portiloop paper',
                      'Use a custom FIR'],
        )
        
        self.b_filename = widgets.Text(
            value='recording.csv',
            description='Recording:',
            disabled=False
        )
        
        self.b_frequency = widgets.IntText(
            value=self.frequency,
            description='Freq (Hz):',
            disabled=False
        )
        
        self.b_threshold = widgets.FloatText(
            value=self.threshold,
            description='Threshold:',
            disabled=True
        )
        
        self.b_polyak_mean = widgets.FloatText(
            value=self.polyak_mean,
            description='Polyak mean:',
            disabled=False
        )
        
        self.b_polyak_std = widgets.FloatText(
            value=self.polyak_std,
            description='Polyak std:',
            disabled=False
        )
        
        self.b_epsilon = widgets.FloatText(
            value=self.epsilon,
            description='Epsilon:',
            disabled=False
        )
        
        self.b_custom_fir_order = widgets.IntText(
            value=self.custom_fir_order,
            description='FIR order:',
            disabled=True
        )
        
        self.b_custom_fir_cutoff = widgets.IntText(
            value=self.custom_fir_cutoff,
            description='FIR cutoff:',
            disabled=True
        )
        
        self.b_use_fir = widgets.Checkbox(
            value=self.filter_args[0],
            description='Use FIR',
            disabled=False,
            indent=False
        )
        
        self.b_use_notch = widgets.Checkbox(
            value=self.filter_args[1],
            description='Use notch',
            disabled=False,
            indent=False
        )
        
        self.b_use_std = widgets.Checkbox(
            value=self.filter_args[2],
            description='Use standardization',
            disabled=False,
            indent=False
        )
        
        self.b_accordion_filter = widgets.Accordion(
            children=[
                widgets.VBox([
                    self.b_custom_fir,
                    self.b_custom_fir_order,
                    self.b_custom_fir_cutoff,
                    self.b_polyak_mean,
                    self.b_polyak_std,
                    self.b_epsilon,
                    widgets.HBox([
                        self.b_use_fir,
                        self.b_use_notch,
                        self.b_use_std
                    ])
                ])
            ])
        
        self.b_accordion_filter.set_title(index = 0, title = 'Filtering')

        self.b_vref = widgets.FloatText(
            value=self.vref,
            description='VREF (V):',
            disabled=False
        )

        self.b_accordion_calibration = widgets.Accordion(
            children=[
                widgets.VBox([
                    self.b_vref
                ])
            ])
        
        self.b_accordion_calibration.set_title(index = 0, title = 'Calibration')


        self.b_detector = widgets.Dropdown(
            options=list(Detector._registry.keys()),
            value='Spindle',
            description='Detector',
            style={'description_width': 'initial'},
            disabled=False,
        )

        self.b_accordion_detector = widgets.Accordion(
            children=[
                widgets.VBox([
                    self.b_detector
                ])
            ])

        self.b_accordion_detector.set_title(index = 0, title = 'Detector')
        
        self.b_stimulator = widgets.Dropdown(
            options=list(Stimulator._registry.keys()),
            value='Spindle',
            description='Stimulator',
            style={'description_width': 'initial'},
            disabled=False,
        )

        self.b_accordion_stimulator = widgets.Accordion(
            children=[
                widgets.VBox([
                    self.b_stimulator
                ])
            ])

        self.b_accordion_stimulator.set_title(index = 0, title = 'Stimulator')
        
        self.b_duration = widgets.IntText(
            value=self.duration,
            description='Time (s):',
            disabled=False
        )
        
        self.b_filter = widgets.Checkbox(
            value=self.filter,
            description='Filter',
            disabled=False,
            indent=False
        )
        
        self.b_detect = widgets.Checkbox(
            value=self.detect,
            description='Detect',
            disabled=False,
            indent=False
        )
        
        self.b_stimulate = widgets.Checkbox(
            value=self.stimulate,
            description='Stimulate',
            disabled=True,
            indent=False
        )

        self.b_record = widgets.Checkbox(
            value=self.record,
            description='Record CSV',
            disabled=False,
            indent=False
        )
        
        self.b_lsl = widgets.Checkbox(
            value=self.lsl,
            description='Stream LSL',
            disabled=False,
            indent=False
        )
        
        self.b_display = widgets.Checkbox(
            value=self.display,
            description='Display',
            disabled=False,
            indent=False
        )
        
        self.b_volume = widgets.IntSlider(
            value=self.volume, 
            min=0,
            max=100,
            step=1,
            description="Volume",
            disabled=False
        )
        
        self.b_test_stimulus = widgets.Button(
            description='Test stimulus',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Send a test stimulus'
        )
        
        self.b_test_impedance = widgets.Button(
            description='Impedance Check',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Check if electrodes are properly connected'
        )

        self.b_stim_delay = widgets.FloatSlider(
            value=self.stim_delay,
            min=0.0,
            max=10.0,
            step=0.01,
            description="Stim Delay",
            tooltip="Time delay in seconds between detection and stimulation", 
            disabled=False,
            style={'description_width': 'initial'}
        )

        self.b_inter_stim_delay = widgets.FloatSlider(
            value=self.inter_stim_delay,
            min=0.0,
            max=10.0,
            step=0.01,
            description="Inter Stim Delay",
            tooltip="Minimum time delay in seconds between stimulation and next detection", 
            disabled=False,
            style={'description_width': 'initial'}
        )
        
        self.b_spindle_mode = widgets.Dropdown(
            options=['Fast', 'Peak', 'Through'],
            value='Fast',
            description='Spindle Stimulation Mode',
            disabled=False,
            style={'description_width': 'initial'}
        )
        
        self.b_spindle_freq = widgets.IntText(
            value=self.spindle_freq,
            description='Spindle Freq (Hz):',
            disabled=False,
            style={'description_width': 'initial'}
        )

        self.b_so_phase_delay = widgets.Checkbox(
            value=self.so_phase_delay,
            description='SO Phase Delay',
            disabled=False,
        )

        self.b_accordion_delaying = widgets.Accordion(
            children=[
                widgets.VBox([
                    self.b_stim_delay,
                    self.b_inter_stim_delay,
                    widgets.HBox([
                        self.b_spindle_mode, 
                        self.b_spindle_freq
                    ]),
                    self.b_so_phase_delay
                ]),
            ]
        )
        self.b_accordion_delaying.set_title(index = 0, title = 'Delaying')
        
        self.register_callbacks()

        self.display_buttons()

        # Get all the metadata from this recording: 

    def get_capture_dictionary(self):
        input_dict = RUN_SETTINGS
        # input_dict = vars(self)
        for k, v in vars(self).items():
            input_dict[k] = v
        basic_types = (int, float, bool, str, list, dict, tuple, set)
        output_dict = {}
        for key, value in input_dict.items():
            # Remove all buttons and button lists from the metadata
            if "button" in key or "_b" in key:
                continue
            if isinstance(value, basic_types):
                output_dict[key] = value
                
        output_dict['filename'] = str(self.filename)

        # Make sure we dont have the filter settings in duplicates 
        for key, value in output_dict['filter_settings'].items():
            if key in output_dict:
                output_dict.pop(key)

        return output_dict


    def on_b_channel_state(self, value, i):
        self.channel_states[i] = value['new']

    def register_callbacks(self):

        for i in range(self.nb_channels):
            def callback_wrapper(channel_idx):
                def callback(change):
                    self.on_b_channel_state(change, channel_idx)
                return callback
            self.chann_buttons[i].observe(callback_wrapper(i), 'value')

        self.b_capture.observe(self.on_b_capture, 'value')
        self.b_clock.observe(self.on_b_clock, 'value')
        self.b_signal_input.observe(self.on_b_signal_input, 'value')
        self.b_fake_filename.observe(self.on_b_fake_filename, 'value')
        self.b_frequency.observe(self.on_b_frequency, 'value')
        self.b_threshold.observe(self.on_b_threshold, 'value')
        self.b_duration.observe(self.on_b_duration, 'value')
        self.b_filter.observe(self.on_b_filter, 'value')
        self.b_use_fir.observe(self.on_b_use_fir, 'value')
        self.b_use_notch.observe(self.on_b_use_notch, 'value')
        self.b_use_std.observe(self.on_b_use_std, 'value')
        self.b_detect.observe(self.on_b_detect, 'value')
        self.b_stimulate.observe(self.on_b_stimulate, 'value')
        self.b_record.observe(self.on_b_record, 'value')
        self.b_lsl.observe(self.on_b_lsl, 'value')
        self.b_display.observe(self.on_b_display, 'value')
        self.b_filename.observe(self.on_b_filename, 'value')
        self.b_channel_detect.observe(self.on_b_channel_detect, 'value')
        self.b_sound_detect.observe(self.on_b_sound_detect, 'value')
        self.b_spindle_mode.observe(self.on_b_spindle_mode, 'value')
        self.b_spindle_freq.observe(self.on_b_spindle_freq, 'value')
        self.b_power_line.observe(self.on_b_power_line, 'value')
        self.b_signal_input.observe(self.on_b_power_line, 'value')
        self.b_custom_fir.observe(self.on_b_custom_fir, 'value')
        self.b_custom_fir_order.observe(self.on_b_custom_fir_order, 'value')
        self.b_custom_fir_cutoff.observe(self.on_b_custom_fir_cutoff, 'value')
        self.b_vref.observe(self.on_b_vref, 'value')
        self.b_polyak_mean.observe(self.on_b_polyak_mean, 'value')
        self.b_polyak_std.observe(self.on_b_polyak_std, 'value')
        self.b_epsilon.observe(self.on_b_epsilon, 'value')
        self.b_volume.observe(self.on_b_volume, 'value')
        self.b_test_stimulus.on_click(self.on_b_test_stimulus)
        self.b_test_impedance.on_click(self.on_b_test_impedance)
        self.b_pause.observe(self.on_b_pause, 'value')
        self.b_stim_delay.observe(self.on_b_delay, 'value')
        self.b_inter_stim_delay.observe(self.on_b_inter_delay, 'value')
        self.b_so_phase_delay.observe(self.on_b_so_phase_delay, 'value')
        self.b_detector.observe(self.on_b_detector, 'value')
        self.b_stimulator.observe(self.on_b_stimulator, 'value')
        


    def __del__(self):
        self.b_capture.close()
    
    def display_buttons(self):
        display(widgets.VBox([self.b_accordion_channels,
                              self.b_channel_detect,
                              self.b_sound_detect,
                              self.b_frequency,
                              self.b_duration,
                              self.b_filename,
                              self.b_fake_filename,
                              self.b_signal_input,
                              self.b_power_line,
                              self.b_clock,
                              widgets.HBox([self.b_filter, self.b_detect, self.b_stimulate, self.b_record, self.b_lsl, self.b_display]),
                              widgets.HBox([self.b_threshold, self.b_test_stimulus]),
                              self.b_volume,
                            #   self.b_test_impedance,
                              self.b_accordion_delaying,
                              self.b_accordion_filter,
                              self.b_accordion_detector, 
                              self.b_accordion_stimulator,
                              self.b_accordion_calibration,
                              self.b_capture,
                              self.b_pause]))

    def enable_buttons(self):
        self.b_frequency.disabled = False
        self.b_duration.disabled = False
        self.b_filename.disabled = False
        self.b_filter.disabled = False
        self.b_detect.disabled = False
        self.b_record.disabled = False
        self.b_lsl.disabled = False
        self.b_display.disabled = False
        self.b_clock.disabled = False
        for i in range(self.nb_channels):
            self.chann_buttons[i].disabled = False
        self.b_power_line.disabled = False
        self.b_signal_input.disabled = False
        self.b_fake_filename.disabled = self.signal_input == 'ADS'
        self.b_channel_detect.disabled = False
        self.b_spindle_freq.disabled = False
        self.b_spindle_mode.disabled = False
        self.b_polyak_mean.disabled = False
        self.b_polyak_std.disabled = False
        self.b_epsilon.disabled = False
        self.b_use_fir.disabled = False
        self.b_use_notch.disabled = False
        self.b_use_std.disabled = False
        self.b_custom_fir.disabled = False
        self.b_custom_fir_order.disabled = not self.custom_fir
        self.b_custom_fir_cutoff.disabled = not self.custom_fir
        self.b_vref.disabled = False
        self.b_stimulate.disabled = not self.detect
        self.b_threshold.disabled = not self.detect
        self.b_pause.disabled = not self.detect
        self.b_test_stimulus.disabled = False # only enabled when running
        self.b_test_impedance.disabled = False
        self.b_stim_delay.disabled = False
        self.b_inter_stim_delay.disabled = False
        self.b_so_phase_delay.disabled = False
        self.b_sound_detect.disabled = False
        self.b_detector.disabled = False
        self.b_stimulator.disabled = False 
        
    def disable_buttons(self):
        self.b_frequency.disabled = True
        self.b_duration.disabled = True
        self.b_filename.disabled = True
        self.b_filter.disabled = True
        self.b_stimulate.disabled = True
        self.b_filter.disabled = True
        self.b_detect.disabled = True
        self.b_record.disabled = True
        self.b_lsl.disabled = True
        self.b_display.disabled = True
        self.b_clock.disabled = True
        for i in range(self.nb_channels):
            self.chann_buttons[i].disabled = True
        self.b_channel_detect.disabled = True
        self.b_spindle_freq.disabled = True
        self.b_spindle_mode.disabled = True
        self.b_signal_input.disabled = True
        self.b_fake_filename.disabled = True
        self.b_power_line.disabled = True
        self.b_polyak_mean.disabled = True
        self.b_polyak_std.disabled = True
        self.b_epsilon.disabled = True
        self.b_use_fir.disabled = True
        self.b_use_notch.disabled = True
        self.b_use_std.disabled = True
        self.b_custom_fir.disabled = True
        self.b_custom_fir_order.disabled = True
        self.b_custom_fir_cutoff.disabled = True
        self.b_vref.disabled = True
        self.b_threshold.disabled = True
        # self.b_test_stimulus.disabled = not self.stimulate # only enabled when running
        self.b_test_impedance.disabled = True
        self.b_stim_delay.disabled = True
        self.b_inter_stim_delay.disabled = True
        self.b_so_phase_delay.disabled = True
        self.b_sound_detect.disabled = True
        self.b_detector.disabled = True
        self.b_stimulator.disabled = True 


    def on_b_sound_detect(self, value):
        self.detection_sound = value['new']

    def on_b_channel_detect(self, value):
        self.channel_detection = value['new']

    def on_b_spindle_freq(self, value): 
        val = value['new']
        if val > 0:
            self.spindle_freq = val
        else:
            self.b_spindle_freq.value = self.spindle_freq
        
    def on_b_spindle_mode(self, value):
        self.spindle_detection_mode = value['new']
        
    def on_b_capture(self, value):
        val = value['new']
        if val == 'Start':
            clear_output()
            self.disable_buttons()
            if not self.python_clock:  # ADS clock: force the frequency to an ADS-compatible frequency
                self.frequency = to_ads_frequency(self.frequency)
                self.b_frequency.value = self.frequency
            self.display_buttons()
            if self._t_capture is not None:
                warnings.warn("Capture already running, operation aborted.")
                return
            detector_type = self.detector_type if self.detect else None
            stimulator_type = self.stimulator_type if self.stimulate else None

            self.filter_settings = {
                "power_line": self.power_line,
                "custom_fir": self.custom_fir,
                "custom_fir_order": self.custom_fir_order,
                "custom_fir_cutoff": self.custom_fir_cutoff,
                "polyak_mean": self.polyak_mean,
                "polyak_std": self.polyak_std,
                "epsilon": self.epsilon,
                "filter_args": self.filter_args,
            }

            self.width_display = 5 * self.frequency # Display 5 seconds of signal

            self._t_capture = Process(
                target=start_capture,
                args=(
                    detector_type,
                    stimulator_type,
                    self.get_capture_dictionary(),
                    self.q_msg,
                    None,  # no q_display, use a LiveDisplay instead
                    self.pause_value,
                )
            )

            self._t_capture.start()
            print(f"PID start process: {self._t_capture.pid}. Kill this process if program crashes before end of execution.")
        elif val == 'Stop':
            self.q_msg.put('STOP')
            assert self._t_capture is not None
            self._t_capture.join()
            self._t_capture = None
            self.enable_buttons()
            
    def on_b_custom_fir(self, value):
        val = value['new']
        if val == 'Default':
            self.custom_fir = False
        elif val == 'Custom':
            self.custom_fir = True
        self.enable_buttons()
    
    def on_b_clock(self, value):
        val = value['new']
        if val == 'Coral':
            self.python_clock = True
        elif val == 'ADS':
            self.python_clock = False

    def on_b_signal_input(self, value):
        val = value['new']
        if val == "ADS":
            self.signal_input = "ADS"
        elif val == "File":
            self.signal_input = "File"
        self.enable_buttons()

    def on_b_fake_filename(self, value):
        val = value['new']
        self.fake_filename = val
        

    def on_b_power_line(self, value):
        val = value['new']
        if val == '60 Hz':
            self.power_line = 60
        elif val == '50 Hz':
            self.power_line = 50
    
    def on_b_frequency(self, value):
        val = value['new']
        if val > 0:
            self.frequency = val
        else:
            self.b_frequency.value = self.frequency
            
    def on_b_threshold(self, value):
        val = value['new']
        if val >= 0 and val <= 1:
            self.threshold = val
        else:
            self.b_threshold.value = self.threshold
            
    def on_b_filename(self, value):
        val = value['new']
        if val != '':
            if not val.endswith('.csv'):
                val += '.csv'
            self.filename = CSV_PATH / val
        else:
            self.filename = CSV_PATH / 'recording.csv'
        
    def on_b_duration(self, value):
        val = value['new']
        if val > 0:
            self.duration = val
    
    def on_b_custom_fir_order(self, value):
        val = value['new']
        if val > 0:
            self.custom_fir_order = val
        else:
            self.b_custom_fir_order.value = self.custom_fir_order
    
    def on_b_custom_fir_cutoff(self, value):
        val = value['new']
        if val > 0 and val < self.frequency / 2:
            self.custom_fir_cutoff = val
        else:
            self.b_custom_fir_cutoff.value = self.custom_fir_cutoff
    
    def on_b_polyak_mean(self, value):
        val = value['new']
        if val >= 0 and val <= 1:
            self.polyak_mean = val
        else:
            self.b_polyak_mean.value = self.polyak_mean
    
    def on_b_polyak_std(self, value):
        val = value['new']
        if val >= 0 and val <= 1:
            self.polyak_std = val
        else:
            self.b_polyak_std.value = self.polyak_std
    
    def on_b_epsilon(self, value):
        val = value['new']
        if val > 0 and val < 0.1:
            self.epsilon = val
        else:
            self.b_epsilon.value = self.epsilon
    
    def on_b_filter(self, value):
        val = value['new']
        self.filter = val
    
    def on_b_use_fir(self, value):
        val = value['new']
        self.filter_args[0] = val
    
    def on_b_use_notch(self, value):
        val = value['new']
        self.filter_args[1] = val
    
    def on_b_use_std(self, value):
        val = value['new']
        self.filter_args[2] = val
    
    def on_b_vref(self, value):
        val = value['new']
        self.vref = val
    
    def on_b_stimulate(self, value):
        val = value['new']
        self.stimulate = val
    
    def on_b_detect(self, value):
        val = value['new']
        self.detect = val
        self.enable_buttons()
    
    def on_b_record(self, value):
        val = value['new']
        self.record = val
    
    def on_b_lsl(self, value):
        val = value['new']
        self.lsl = val
    
    def on_b_display(self, value):
        val = value['new']
        self.display = val
        
    def on_b_volume(self, value):
        val = value['new']
        if val >= 0 and val <= 100:
            self.volume = val
            self.mixer.setvolume(self.volume)
    
    def on_b_test_stimulus(self, b):
        self.run_test_stimulus()
            
    def on_b_test_impedance(self, b):
        self.run_impedance_test()
    
    def on_b_pause(self, value):
        val = value['new']
        if val == 'Active':
            self.pause_value.value = False
        elif val == 'Paused':
            self.pause_value.value = True
    
    def on_b_delay(self, value):
        val = value['new']
        self.stim_delay = val

    def on_b_inter_delay(self, value):
        val = value['new']
        self.inter_stim_delay = val

    def on_b_so_phase_delay(self, value):
        val = value['new']
        self.so_phase_delay = val

    def on_b_detector(self, value):
        self.detector_type = value['new']

    def on_b_stimulator(self, value):
        self.stimulator_type = value['new']


    def run_test_stimulus(self):
        stimulator_class = Stimulator.get_stimulator(self.stimulator_type)(soundname=self.detection_sound)
        stimulator_class.test_stimulus()
        del stimulator_class

    def run_impedance_test(self):
        frontend = Frontend(portiloop_version=2)
        
        def is_set(x, n):
            return x & 1 << n != 0
        
        try:
            frontend.write_regs(0x00, LEADOFF_CONFIG)
            frontend.start()
            start_time = time.time()
            current_time = time.time()
            while current_time - start_time < 2:
                current_time = time.time()
            reading = frontend.read()
            
            # Check if any of the negative bits are set and initialize the impedance array
#             impedance_check = [any([is_set(leadoff_n, i) for i in range(2, 9)])]
            impedance_check = [any([reading.loff_n(i) for i in range(7)])]
            
            for i in range(7):
                impedance_check.append(reading.loff_p(i))
                             
            def print_impedance(impedance):
                names = ["Ref", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "Ch7", "Ch8"]
                vals = [' Y ' if val else ' N ' for val in impedance]
                print(' '.join(str(name) for name in names))
                print(' '.join(str(val) for val in vals))
                    
            print_impedance(impedance_check)
            
        finally: 
            frontend.close()


if __name__ == "__main__":
    pass
