

from abc import ABC, abstractmethod
from time import sleep
import time
import numpy as np
from copy import deepcopy
from datetime import datetime
import multiprocessing as mp
import warnings
from threading import Thread, Lock
from portiloop.src import ADS

if ADS:
    import alsaaudio
    from portiloop.src.hardware.frontend import Frontend
    from portiloop.src.hardware.leds import LEDs, Color

from portiloop.src.stimulation import UpStateDelayer


from portiloop.src.processing import FilterPipeline, int_to_float
from portiloop.src.config import mod_config, LEADOFF_CONFIG, FRONTEND_CONFIG, to_ads_frequency
from portiloop.src.utils import ADSFrontend, Dummy, FileFrontend, FileReader, LSLStreamer, LiveDisplay, DummyAlsaMixer, EDFRecorder, EDF_PATH, RECORDING_PATH
from IPython.display import clear_output, display
import ipywidgets as widgets


def capture_process(p_data_o, p_msg_io, duration, frequency, python_clock, time_msg_in, channel_states):
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

    frontend = Frontend()
    leds = LEDs()
    leds.led2(Color.PURPLE)
    leds.aquisition(True)
    
    try:
        data = frontend.read_regs(0x00, 1)
        assert data == [0x3E], "The communication with the ADS failed, please try again."
        leds.led2(Color.BLUE)
        
        config = FRONTEND_CONFIG
        if python_clock:  # set ADS to 2 * frequency
            datarate = 2 * frequency
        else:  # set ADS to frequency
            datarate = frequency
        config = mod_config(config, datarate, channel_states)
        
        frontend.write_regs(0x00, config)
        data = frontend.read_regs(0x00, len(config))
        assert data == config, f"Wrong config: {data} vs {config}"
        frontend.start()
        leds.led2(Color.PURPLE)
        while not frontend.is_ready():
            pass

        # Set up of leds
        leds.aquisition(True)
        sleep(0.5)
        leds.aquisition(False)
        sleep(0.5)
        leds.aquisition(True)

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
                    if message == 'STOP':
                        c = False
            it += 1
        t = time.time()
        tot = (t - t_start) / it        

        p_msg_io.send(("PRT", f"Average frequency: {1 / tot} Hz for {it} samples"))

    finally:
        leds.aquisition(False)
        leds.close()
        frontend.close()
        p_msg_io.send('STOP')
        p_msg_io.close()
        p_data_o.close()
    


class Capture:
    def __init__(self, detector_cls=None, stimulator_cls=None):
        # {now.strftime('%m_%d_%Y_%H_%M_%S')}
        self.filename = EDF_PATH / 'recording.edf'
        
        # Hardware options
        self.nb_channels = 8 # TODO: Change that to adapt according to portiloop version

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
        self._lock_msg_out = Lock()
        self._msg_out = None
        self._t_capture = None

        # Channel parameters
        self.signal_labels = ['Common Mode'] + [f"ch{i+1}" for i in range(self.nb_channels)]
        self.channel_states = ['disabled', 'disabled', 'disabled', 'disabled', 'disabled', 'disabled', 'disabled']
        self.channel_detection = 2

        # Delayer parameters
        self.spindle_detection_mode = 'Fast'
        self.spindle_freq = 10
        
        # Stimulator and detector classes
        self.detector_cls = detector_cls
        self.stimulator_cls = stimulator_cls
        
        # Pause detection
        self._pause_detect_lock = Lock()
        self._pause_detect = True
        
        if ADS:
            try:
                mixers = alsaaudio.mixers()
                if len(mixers) <= 0:
                    warnings.warn(f"No ALSA mixer found.")
                    self.mixer = DummyAlsaMixer()
                elif 'PCM' in mixers:
                    self.mixer = alsaaudio.Mixer(control='PCM')
                else:
                    warnings.warn(f"Could not find mixer PCM, using {mixers[0]} instead.")
                    self.mixer = alsaaudio.Mixer(control=mixers[0])
            except ALSAAudioError as e:
                warnings.warn(f"No ALSA mixer found.")
                self.mixer = DummyAlsaMixer()
            
            self.volume = self.mixer.getvolume()[0]  # we will set the same volume on all channels
        else:
            self.mixer = DummyAlsaMixer()
            self.volume = self.mixer.getvolume()[0]
        
        # widgets ===============================
        
        # CHANNELS ------------------------------
        self.chann_buttons = []
        for i in range(self.nb_channels):
            self.chann_buttons.append(widgets.Button(
                description=f'ch{i+1}',
                disabled=False,
                button_style='info',  # 'success', 'info', 'warning', 'danger' or ''
                tooltip=f'Enable channel {i+1}',
                options=['disabled', 'simple'],
                value='disabled',
            ))

        
        self.b_channel_detect = widgets.Dropdown(
            options=[(f'{i+1}', i+1) for i in range(self.nb_channels)],
            value=2,
            description='Detection Channel:',
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
        
        self.b_accordion_channels = widgets.Accordion(
            children=[
                widgets.GridBox([[f"CH{i+1}"] + [self.chann_buttons[i]] for i in range(self.nb_channels)], 
                                layout=widgets.Layout(grid_template_columns="repeat(7, 90px)")
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
            tooltips=['Detector and stimulator active', 'Detector and stimulator paused'],
        )
        
        self.b_clock = widgets.ToggleButtons(
            options=['ADS', 'Coral'],
            description='Clock:',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Use Coral clock (very precise, not very timely)',
                      'Use ADS clock (not very precise, very timely)'],
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
        
        self.b_custom_fir = widgets.ToggleButtons(
            options=['Default', 'Custom'],
            description='FIR filter:',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Use the default 30Hz low-pass FIR from the Portiloop paper',
                      'Use a custom FIR'],
        )
        
        self.b_filename = widgets.Text(
            value='recording.edf',
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
            description='Record EDF',
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
            disabled=True,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Send a test stimulus'
        )
        
        self.b_test_impedance = widgets.Button(
            description='Impedance Check',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Check if electrodes are properly connected'
        )
        
        self.register_callbacks()

        self.display_buttons()


    def register_callbacks(self):


        #         self.b_radio_ch2.observe(self.on_b_radio_ch2, 'value')
        # self.b_radio_ch3.observe(self.on_b_radio_ch3, 'value')
        # self.b_radio_ch4.observe(self.on_b_radio_ch4, 'value')
        # self.b_radio_ch5.observe(self.on_b_radio_ch5, 'value')
        # self.b_radio_ch6.observe(self.on_b_radio_ch6, 'value')
        # self.b_radio_ch7.observe(self.on_b_radio_ch7, 'value')
        # self.b_radio_ch8.observe(self.on_b_radio_ch8, 'value')
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
        self.b_spindle_mode.observe(self.on_b_spindle_mode, 'value')
        self.b_spindle_freq.observe(self.on_b_spindle_freq, 'value')
        self.b_power_line.observe(self.on_b_power_line, 'value')
        self.b_signal_input.observe(self.on_b_power_line, 'value')
        self.b_custom_fir.observe(self.on_b_custom_fir, 'value')
        self.b_custom_fir_order.observe(self.on_b_custom_fir_order, 'value')
        self.b_custom_fir_cutoff.observe(self.on_b_custom_fir_cutoff, 'value')
        self.b_polyak_mean.observe(self.on_b_polyak_mean, 'value')
        self.b_polyak_std.observe(self.on_b_polyak_std, 'value')
        self.b_epsilon.observe(self.on_b_epsilon, 'value')
        self.b_volume.observe(self.on_b_volume, 'value')
        self.b_test_stimulus.on_click(self.on_b_test_stimulus)
        self.b_test_impedance.on_click(self.on_b_test_impedance)
        self.b_pause.observe(self.on_b_pause, 'value')

    def __del__(self):
        self.b_capture.close()
    
    def display_buttons(self):
        display(widgets.VBox([self.b_accordion_channels,
                              self.b_channel_detect,
                              self.b_frequency,
                              self.b_duration,
                              self.b_filename,
                              self.b_signal_input,
                              self.b_power_line,
                              self.b_clock,
                              widgets.HBox([self.b_filter, self.b_detect, self.b_stimulate, self.b_record, self.b_lsl, self.b_display]),
                              widgets.HBox([self.b_threshold, self.b_test_stimulus]),
                              self.b_volume,
                              widgets.HBox([self.b_spindle_mode, self.b_spindle_freq]),
                              self.b_test_impedance,
                              self.b_accordion_filter,
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
        self.b_radio_ch2.disabled = False
        self.b_radio_ch3.disabled = False
        self.b_radio_ch4.disabled = False
        self.b_radio_ch5.disabled = False
        self.b_radio_ch6.disabled = False
        self.b_radio_ch7.disabled = False
        self.b_radio_ch8.disabled = False
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
        self.b_stimulate.disabled = not self.detect
        self.b_threshold.disabled = not self.detect
        self.b_pause.disabled = not self.detect
        self.b_test_stimulus.disabled = True # only enabled when running
        self.b_test_impedance.disabled = False
    
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
        self.b_radio_ch2.disabled = True
        self.b_radio_ch3.disabled = True
        self.b_radio_ch4.disabled = True
        self.b_radio_ch5.disabled = True
        self.b_radio_ch6.disabled = True
        self.b_radio_ch7.disabled = True
        self.b_radio_ch8.disabled = True
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
        self.b_threshold.disabled = True
        self.b_test_stimulus.disabled = not self.stimulate # only enabled when running
        self.b_test_impedance.disabled = True
    
    def on_b_radio_ch2(self, value):
        self.channel_states[0] = value['new']
    
    def on_b_radio_ch3(self, value):
        self.channel_states[1] = value['new']
    
    def on_b_radio_ch4(self, value):
        self.channel_states[2] = value['new']
    
    def on_b_radio_ch5(self, value):
        self.channel_states[3] = value['new']
    
    def on_b_radio_ch6(self, value):
        self.channel_states[4] = value['new']
    
    def on_b_radio_ch7(self, value):
        self.channel_states[5] = value['new']
    
    def on_b_radio_ch8(self, value):
        self.channel_states[6] = value['new']
        
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
            with self._lock_msg_out:
                self._msg_out = None
            if self._t_capture is not None:
                warnings.warn("Capture already running, operation aborted.")
                return
            detector_cls = self.detector_cls if self.detect else None
            stimulator_cls = self.stimulator_cls if self.stimulate else None
            
            self._t_capture = Thread(target=self.start_capture,
                                args=(self.filter,
                                      self.filter_args,
                                      detector_cls,
                                      self.threshold,
                                      self.channel_detection,
                                      stimulator_cls,
                                      self.record,
                                      self.lsl,
                                      self.display,
                                      2500,
                                      self.python_clock))
            self._t_capture.start()
        elif val == 'Stop':
            with self._lock_msg_out:
                self._msg_out = 'STOP'
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
            if not val.endswith('.edf'):
                val += '.edf'
            self.filename = EDF_PATH / val
        else:
            now = datetime.now()
            self.filename = EDF_PATH / 'recording.edf'
        
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
            with self._pause_detect_lock:
                self._pause_detect = False
        elif val == 'Paused':
            with self._pause_detect_lock:
                self._pause_detect = True

    def run_test_stimulus(self):
        stimulator_class = self.stimulator_cls()
        stimulator_class.test_stimulus()

    def run_impedance_test(self):
        frontend = Frontend()
        
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


def start_capture(self,
                    filter,
                    detector_cls,
                    threshold,
                    channel,
                    stimulator_cls,
                    record,
                    lsl,
                    viz,
                    width,
                    python_clock,
                    nb_channels,
                    frequency,
                    edf_filename,
                    filter_settings,
                    _pause_detect_lock,
                    _pause_detect,):
    
    # Initialize data frontend
    filename = RECORDING_PATH / 'test_recording.csv'
    capture_frontend = ADSFrontend(
        duration=self.duration,
        frequency=self.frequency,
        python_clock=python_clock,
        channel_states=self.channel_states
    ) if self.signal_input == "ADS" else FileFrontend(filename)

    # Initialize filtering pipeline
    if filter:
        fp = FilterPipeline(nb_channels=nb_channels,
                            sampling_rate=frequency,
                            power_line_fq=filter_settings['power_line'],
                            use_custom_fir=filter_settings['custom_fir'],
                            custom_fir_order=filter_settings['custom_fir_order'],
                            custom_fir_cutoff=filter_settings['custom_fir_cutoff'],
                            alpha_avg=filter_settings['polyak_mean'],
                            alpha_std=filter_settings['polyak_std'],
                            epsilon=filter_settings['epsilon'],
                            filter_args=filter_settings['filter_args'])
    
    # Initialize detector and stimulator
    detector = detector_cls(threshold, channel=channel) if detector_cls is not None else None
    stimulator = stimulator_cls() if stimulator_cls is not None else None

    # Launch the capture process
    capture_frontend.init_capture()

    # Initialize display if requested
    live_disp = LiveDisplay(channel_names=self.signal_labels, window_len=width) if viz else Dummy()

    # Initialize recording if requested
    recorder = EDFRecorder(self.signal_labels, edf_filename, frequency) if record else Dummy()
    recorder.open_recording_file()
    
    # Initialize LSL to stream if requested
    streams = {
            'filtered': filter,
            'markers': detector is not None,
        }
    lsl_streamer = LSLStreamer(streams, nb_channels, self.frequency) if lsl else Dummy()

    # Buffer used for the visualization and the recording
    buffer = []

    # Initialize stimulation delayer if requested
    delay = not self.spindle_detection_mode == 'Fast' and stimulator is not None
    stimulation_delayer = UpStateDelayer(self.frequency, self.spindle_detection_mode == 'Peak', 0.3) if delay else Dummy()
    stimulator.add_delayer(stimulation_delayer)

    # Main capture loop
    while True:
        # First, we send all outgoing messages to the capture process
        with self._lock_msg_out:
            if self._msg_out is not None:
                capture_frontend.send_msg(self._msg_out)
                self._msg_out = None

        # Then, we check if we have received a message from the capture process
        msg = capture_frontend.get_msg()
        # Either we have received a stop message, or a print message.
        if msg == 'STOP':
            break
        elif msg[0] == 'PRT':
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

        # Contains the filtered point (if filtering is off, contains a copy of the raw point)
        filtered_point = filtered_point.tolist()
        
        # Send both raw and filtered points over LSL
        raw_point = raw_point.tolist()
        lsl_streamer.push_raw(raw_point[-1])
        if filter:
            lsl_streamer.push_filtered(filtered_point[-1])
        
        # Check if detection is on or off
        with _pause_detect_lock:
            pause = _pause_detect

        # If detection is on
        if detector is not None and not pause:
            # Detect using the latest point
            detection_signal = detector.detect(filtered_point)

            # Stimulate
            if stimulator is not None:                    
                stimulator.stimulate(detection_signal)

            # Adds point to buffer for delayed stimulation
            stimulation_delayer.step_timesteps(filtered_point[0][channel-1])
        
        # Add point to the buffer to send to viz and recorder
        buffer += filtered_point
        if len(buffer) >= 50:
            live_disp.add_datapoints(buffer)
            recorder.add_recording_data(buffer)
            buffer = []

    # close the frontend
    capture_frontend.close()
    
    recorder.close_recording_file()


if __name__ == "__main__":
    pass
