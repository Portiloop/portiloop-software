import warnings
import time
from multiprocessing import Process, Queue, Value
import os

from portiloop.src import ADS
from portiloop.src.core.capture import start_capture
from portiloop.src.core.hardware.config_hardware import to_ads_frequency, LEADOFF_CONFIG
from portiloop.src.core.utils import get_portiloop_version, DummyAlsaMixer
from portiloop.src.core.constants import CSV_PATH
from portiloop.src.custom.custom_processors import FilterPipeline

from portiloop.src.custom.config import RUN_SETTINGS
from portiloop.src.custom.custom_detectors import SleepSpindleRealTimeDetector
from portiloop.src.custom.custom_stimulators import (SleepSpindleRealTimeStimulator,
                                                     SpindleTrainRealTimeStimulator,
                                                     IsolatedSpindleRealTimeStimulator)

from IPython.display import clear_output, display
import ipywidgets as widgets
from pathlib import Path

if ADS:
    import alsaaudio
    from alsaaudio import ALSAAudioError
    from portiloop.src.core.hardware.frontend import Frontend


class JupyterUI:
    def __init__(self, processor_cls=FilterPipeline, detector_cls=None, stimulator_cls=None):
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
        self.python_clock = True

        # Communication parameters for messages with capture
        self._t_capture = None
        self.q_msg = Queue()
        self.pause_value = Value('b', True)

        # Channel parameters
        self.signal_labels = [f"ch{i + 1}" for i in range(self.nb_channels)]
        self.channel_states = ['disabled' for _ in range(self.nb_channels)]
        self.channel_detection = 2
        self.detection_sound = "stimul_100ms.wav"

        # Delayer parameters
        self.spindle_detection_mode = 'Fast'
        self.spindle_freq = 10
        self.stim_delay = 0.0
        self.inter_stim_delay = 0.0

        # Pipeline
        self.processor_cls = processor_cls
        self.detector_cls = detector_cls
        self.stimulator_cls = stimulator_cls

        # Other
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
        self.width_display = 5 * self.frequency

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
                warnings.warn(
                    f"No ALSA mixer found. Volume control will not be available from notebook.\nAvailable mixers were:\n{mixers}")
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
                tooltip=f'Enable channel {i + 1}',
                options=['disabled', 'simple', 'bias', 'test', 'temp'],
                value='disabled',
            ))

        self.b_channel_detect = widgets.Dropdown(
            options=[(f'{i + 1}', i + 1) for i in range(self.nb_channels)],
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
                widgets.GridBox([widgets.Label(f"CH{i + 1}") for i in range(self.nb_channels)] + self.chann_buttons,
                                layout=widgets.Layout(grid_template_columns=f"repeat({self.nb_channels}, 90px)")
                                )
            ])
        self.b_accordion_channels.set_title(index=0, title='Channels')

        # OTHERS ------------------------------

        self.b_capture = widgets.ToggleButtons(
            options=['Stop', 'Start'],
            description='Capture:',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Stop capture', 'Start capture'],
        )

        self.b_pause = widgets.ToggleButtons(
            options=['Paused', 'Active'],
            description='Detection',
            disabled=True,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Detector and stimulator paused', 'Detector and stimulator active'],
        )

        self.b_clock = widgets.ToggleButtons(
            options=['ADS', 'Coral'],
            description='Clock:',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Use ADS clock (not very precise, very timely)',
                      'Use Coral clock (very precise, not very timely)'],
        )

        self.b_power_line = widgets.ToggleButtons(
            options=['60 Hz', '50 Hz'],
            description='Power line:',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['North America 60 Hz',
                      'Europe 50 Hz'],
        )

        self.b_signal_input = widgets.ToggleButtons(
            options=['ADS', 'File'],
            description='Signal Input:',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Read data from ADS.',
                      'Read data from file.'],
        )

        self.b_custom_fir = widgets.ToggleButtons(
            options=['Default', 'Custom'],
            description='FIR filter:',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
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

        self.b_accordion_filter.set_title(index=0, title='Filtering')

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

        self.b_accordion_calibration.set_title(index=0, title='Calibration')

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
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Send a test stimulus'
        )

        self.b_test_impedance = widgets.Button(
            description='Impedance Check',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
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

        self.b_accordion_delaying = widgets.Accordion(
            children=[
                widgets.VBox([
                    self.b_stim_delay,
                    self.b_inter_stim_delay,
                    widgets.HBox([
                        self.b_spindle_mode,
                        self.b_spindle_freq
                    ])
                ]),
            ]
        )
        self.b_accordion_delaying.set_title(index=0, title='Delaying')

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

    def __del__(self):
        self.b_capture.close()

    def display_buttons(self):
        display(widgets.VBox([self.b_accordion_channels,
                              self.b_channel_detect,
                              self.b_sound_detect,
                              self.b_frequency,
                              self.b_duration,
                              self.b_filename,
                              self.b_signal_input,
                              self.b_power_line,
                              self.b_clock,
                              widgets.HBox([self.b_filter, self.b_detect, self.b_stimulate, self.b_record, self.b_lsl,
                                            self.b_display]),
                              widgets.HBox([self.b_threshold, self.b_test_stimulus]),
                              self.b_volume,
                              #   self.b_test_impedance,
                              self.b_accordion_delaying,
                              self.b_accordion_filter,
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
        self.b_test_stimulus.disabled = False  # only enabled when running
        self.b_test_impedance.disabled = False
        self.b_stim_delay.disabled = False
        self.b_inter_stim_delay.disabled = False
        self.b_sound_detect.disabled = False

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
        self.b_sound_detect.disabled = True

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
                warnings.warn("JupyterUI already running, operation aborted.")
                return
            processor_cls = self.processor_cls if self.filter else None
            detector_cls = self.detector_cls if self.detect else None
            stimulator_cls = self.stimulator_cls if self.stimulate else None

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
            self.width_display = 5 * self.frequency  # Display 5 seconds of signal

            self._t_capture = Process(target=start_capture,
                                      args=(processor_cls,
                                            detector_cls,
                                            stimulator_cls,
                                            self.get_capture_dictionary(),
                                            self.q_msg,
                                            None,  # no q_display, use a LiveDisplay instead
                                            self.pause_value,))
            """
            detector_cls,
            stimulator_cls,
            capture_dictionary,
            q_msg,
            q_display,
            pause_value
            """
            self._t_capture.start()
            print(
                f"PID start process: {self._t_capture.pid}. Kill this process if program crashes before end of execution.")
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

    def run_test_stimulus(self):
        stimulator_class = self.stimulator_cls(soundname=self.detection_sound)
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
