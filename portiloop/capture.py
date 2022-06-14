import os
import sys

from time import sleep
import time
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta
import multiprocessing as mp
import warnings
import shutil
from threading import Thread, Lock
import alsaaudio

from EDFlib.edfwriter import EDFwriter
from scipy.signal import firwin

from portilooplot.jupyter_plot import ProgressPlot
from portiloop.hardware.frontend import Frontend
from portiloop.hardware.leds import LEDs, Color

from IPython.display import clear_output, display
import ipywidgets as widgets


DEFAULT_FRONTEND_CONFIG = [
    # nomenclature: name [default setting] [bits 7-0] : description
    # Read only ID:
    0x3E, # ID [xx] [REV_ID[2:0], 1, DEV_ID[1:0], NU_CH[1:0]] : (RO)
    # Global Settings Across Channels:
    0x96, # CONFIG1 [96] [1, DAISY_EN(bar), CLK_EN, 1, 0, DR[2:0]] : Datarate = 250 SPS
    0xC0, # CONFIG2 [C0] [1, 1, 0, INT_CAL, 0, CAL_AMP0, CAL_FREQ[1:0]] : No tests
    0x60, # CONFIG3 [60] [PD_REFBUF(bar), 1, 1, BIAS_MEAS, BIASREF_INT, PD_BIAS(bar), BIAS_LOFF_SENS, BIAS_STAT] : Power-down reference buffer, no bias
    0x00, # LOFF [00] [COMP_TH[2:0], 0, ILEAD_OFF[1:0], FLEAD_OFF[1:0]] : No lead-off
    # Channel-Specific Settings:
    0x61, # CH1SET [61] [PD1, GAIN1[2:0], SRB2, MUX1[2:0]] : Channel 1 active, 24 gain, no SRB2 & input shorted
    0x61, # CH2SET [61] [PD2, GAIN2[2:0], SRB2, MUX2[2:0]] : Channel 2 active, 24 gain, no SRB2 & input shorted
    0x61, # CH3SET [61] [PD3, GAIN3[2:0], SRB2, MUX3[2:0]] : Channel 3 active, 24 gain, no SRB2 & input shorted
    0x61, # CH4SET [61] [PD4, GAIN4[2:0], SRB2, MUX4[2:0]] : Channel 4 active, 24 gain, no SRB2 & input shorted
    0x61, # CH5SET [61] [PD5, GAIN5[2:0], SRB2, MUX5[2:0]] : Channel 5 active, 24 gain, no SRB2 & input shorted
    0x61, # CH6SET [61] [PD6, GAIN6[2:0], SRB2, MUX6[2:0]] : Channel 6 active, 24 gain, no SRB2 & input shorted
    0x61, # CH7SET [61] [PD7, GAIN7[2:0], SRB2, MUX7[2:0]] : Channel 7 active, 24 gain, no SRB2 & input shorted
    0x61, # CH8SET [61] [PD8, GAIN8[2:0], SRB2, MUX8[2:0]] : Channel 8 active, 24 gain, no SRB2 & input shorted
    0x00, # BIAS_SENSP [00] [BIASP8, BIASP7, BIASP6, BIASP5, BIASP4, BIASP3, BIASP2, BIASP1] : No bias
    0x00, # BIAS_SENSN [00] [BIASN8, BIASN7, BIASN6, BIASN5, BIASN4, BIASN3, BIASN2, BIASN1] No bias
    0x00, # LOFF_SENSP [00] [LOFFP8, LOFFP7, LOFFP6, LOFFP5, LOFFP4, LOFFP3, LOFFP2, LOFFP1] : No lead-off
    0x00, # LOFF_SENSN [00] [LOFFM8, LOFFM7, LOFFM6, LOFFM5, LOFFM4, LOFFM3, LOFFM2, LOFFM1] : No lead-off
    0x00, # LOFF_FLIP [00] [LOFF_FLIP8, LOFF_FLIP7, LOFF_FLIP6, LOFF_FLIP5, LOFF_FLIP4, LOFF_FLIP3, LOFF_FLIP2, LOFF_FLIP1] : No lead-off flip
    # Lead-Off Status Registers (Read-Only Registers):
    0x00, # LOFF_STATP [00] [IN8P_OFF, IN7P_OFF, IN6P_OFF, IN5P_OFF, IN4P_OFF, IN3P_OFF, IN2P_OFF, IN1P_OFF] : Lead-off positive status (RO)
    0x00, # LOFF_STATN [00] [IN8M_OFF, IN7M_OFF, IN6M_OFF, IN5M_OFF, IN4M_OFF, IN3M_OFF, IN2M_OFF, IN1M_OFF] : Laed-off negative status (RO)
    # GPIO and OTHER Registers:
    0x0F, # GPIO [0F] [GPIOD[4:1], GPIOC[4:1]] : All GPIOs as inputs
    0x00, # MISC1 [00] [0, 0, SRB1, 0, 0, 0, 0, 0] : Disable SRBM
    0x00, # MISC2 [00] [00] : Unused
    0x00, # CONFIG4 [00] [0, 0, 0, 0, SINGLE_SHOT, 0, PD_LOFF_COMP(bar), 0] : Single-shot, lead-off comparator disabled
]


FRONTEND_CONFIG = [
    0x3E, # ID (RO)
    0x95, # CONFIG1 [95] [1, DAISY_EN(bar), CLK_EN, 1, 0, DR[2:0]] : Datarate = 500 SPS
    0xC0, # CONFIG2 [C0] [1, 1, 0, INT_CAL, 0, CAL_AMP0, CAL_FREQ[1:0]]
    0xE0, # CONFIG3 [E0] [PD_REFBUF(bar), 1, 1, BIAS_MEAS, BIASREF_INT, PD_BIAS(bar), BIAS_LOFF_SENS, BIAS_STAT] : Power-down reference buffer, no bias
    0x00, # No lead-off
    0x60, # CH1SET [60] [PD1, GAIN1[2:0], SRB2, MUX1[2:0]]
    0x60, # CH2SET
    0x60, # CH3SET
    0x60, # CH4SET
    0x60, # CH5SET
    0x60, # CH6SET
    0x60, # CH7SET
    0x60, # CH8SET
    0x00, # BIAS_SENSP 00
    0x00, # BIAS_SENSN 00
    0x00, # LOFF_SENSP Lead-off on all positive pins?
    0x00, # LOFF_SENSN Lead-off on all negative pins?
    0x00, # Normal lead-off
    0x00, # Lead-off positive status (RO)
    0x00, # Lead-off negative status (RO)
    0x00, # All GPIOs as output ?
    0x20, # Enable SRB1
]

EDF_PATH = Path.home() / 'workspace' / 'edf_recording'


def to_ads_frequency(frequency):
    possible_datarates = [250, 500, 1000, 2000, 4000, 8000, 16000]
    dr = 16000
    for i in possible_datarates:
        if i >= frequency:
            dr = i
            break
    return dr
    

def mod_config(config, datarate, channel_modes):
    
    # datarate:

    possible_datarates = [(250, 0x06),
                          (500, 0x05),
                          (1000, 0x04),
                          (2000, 0x03),
                          (4000, 0x02),
                          (8000, 0x01),
                          (16000, 0x00)]
    mod_dr = 0x00
    for i, j in possible_datarates:
        if i >= datarate:
            mod_dr = j
            break
    
    new_cf1 = config[1] & 0xF8
    new_cf1 = new_cf1 | mod_dr
    config[1] = new_cf1
    
    # bias:

    assert len(channel_modes) == 8
    config[13] = 0x00  # clear BIAS_SENSP
    config[14] = 0x00  # clear BIAS_SENSN
    bias_active = False
    for chan_i, chan_mode in enumerate(channel_modes):
        n = 5 + chan_i
        mod = config[n] & 0x78  # clear PDn and MUX[2:0]
        if chan_mode == 'simple':
            pass  # PDn = 0 and normal electrode (000)
        elif chan_mode == 'disabled':
            mod = mod | 0x81  # PDn = 1 and input shorted (001)
        elif chan_mode == 'with bias':
            bias_active = True
            bit_i = 1 << chan_i
            config[13] = config[13] | bit_i
            config[14] = config[14] | bit_i
        elif chan_mode == 'bias out':
            bias_active = True
            mod = mod | 0x06  # MUX[2:0] = BIAS_DRP (110)
        else:
            assert False, f"Wrong key: {chan_mode}."
        config[n] = mod
    if bias_active:
        config[3] = config[3] | 0x1c  # activate the bias mechanism
    for n, c in enumerate(config):  # print ADS1299 configuration registers
        print(f"config[{n}]:\t{c:08b}\t({hex(c)})")
    return config


def filter_24(value):
    return (value * 4.5) / (2**23 - 1)  # 23 because 1 bit is lost for sign


def filter_2scomplement_np(value):
    return np.where((value & (1 << 23)) != 0, value - (1 << 24), value)


def filter_np(value):
    return filter_24(filter_2scomplement_np(value))


def shift_numpy(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


class FIR:
    def __init__(self, nb_channels, coefficients, buffer=None):
        
        self.coefficients = np.expand_dims(np.array(coefficients), axis=1)
        self.taps = len(self.coefficients)
        self.nb_channels = nb_channels
        self.buffer = np.array(z) if buffer is not None else np.zeros((self.taps, self.nb_channels))
    
    def filter(self, x):
        self.buffer = shift_numpy(self.buffer, 1, x)
        filtered = np.sum(self.buffer * self.coefficients, axis=0)
        return filtered

    
class FilterPipeline:
    def __init__(self,
                 nb_channels,
                 sampling_rate,
                 power_line_fq=60,
                 use_custom_fir=False,
                 custom_fir_order=20,
                 custom_fir_cutoff=30,
                 alpha_avg=0.1,
                 alpha_std=0.001,
                 epsilon=0.000001,
                 filter_args=[]):
        if len(filter_args) > 0:
            use_fir, use_notch, use_std = filter_args
        else:
            use_fir=True,
            use_notch=True,
            use_std=True
        self.use_fir = use_fir
        self.use_notch = use_notch
        self.use_std = use_std
        self.nb_channels = nb_channels
        assert power_line_fq in [50, 60], f"The only supported power line frequencies are 50 Hz and 60 Hz"
        if power_line_fq == 60:
            self.notch_coeff1 = -0.12478308884588535
            self.notch_coeff2 = 0.98729186796473023
            self.notch_coeff3 = 0.99364593398236511
            self.notch_coeff4 = -0.12478308884588535
            self.notch_coeff5 = 0.99364593398236511
        else:
            self.notch_coeff1 = -0.61410695998423581
            self.notch_coeff2 =  0.98729186796473023
            self.notch_coeff3 = 0.99364593398236511
            self.notch_coeff4 = -0.61410695998423581
            self.notch_coeff5 = 0.99364593398236511
        self.dfs = [np.zeros(self.nb_channels), np.zeros(self.nb_channels)]
        
        self.moving_average = None
        self.moving_variance = np.zeros(self.nb_channels)
        self.ALPHA_AVG = alpha_avg
        self.ALPHA_STD = alpha_std
        self.EPSILON = epsilon
        
        if use_custom_fir:
            self.fir_coef = firwin(numtaps=custom_fir_order+1, cutoff=custom_fir_cutoff, fs=sampling_rate)
        else:
            self.fir_coef = [
                0.001623780150148094927192721215192250384,
                0.014988684599373741992978104065059596905,
                0.021287595318265635502275046064823982306,
                0.007349500393709578957568417933998716762,
                -0.025127515717112181709014251396183681209,
                -0.052210507359822452833064687638398027048,
                -0.039273839505489904766477593511808663607,
                0.033021568427940004020193498490698402748,
                0.147606943281569008563636202779889572412,
                0.254000252034505602516389899392379447818,
                0.297330876398883392486283128164359368384,
                0.254000252034505602516389899392379447818,
                0.147606943281569008563636202779889572412,
                0.033021568427940004020193498490698402748,
                -0.039273839505489904766477593511808663607,
                -0.052210507359822452833064687638398027048,
                -0.025127515717112181709014251396183681209,
                0.007349500393709578957568417933998716762,
                0.021287595318265635502275046064823982306,
                0.014988684599373741992978104065059596905,
                0.001623780150148094927192721215192250384]
        self.fir = FIR(self.nb_channels, self.fir_coef)
        
    def filter(self, value):
        """
        value: a numpy array of shape (data series, channels)
        """
        for i, x in enumerate(value):  # loop over the data series
            # FIR:
            if self.use_fir:
                x = self.fir.filter(x)
            # notch:
            if self.use_notch:
                denAccum = (x - self.notch_coeff1 * self.dfs[0]) - self.notch_coeff2 * self.dfs[1]
                x = (self.notch_coeff3 * denAccum + self.notch_coeff4 * self.dfs[0]) + self.notch_coeff5 * self.dfs[1]
                self.dfs[1] = self.dfs[0]
                self.dfs[0] = denAccum
            # standardization:
            if self.use_std:
                if self.moving_average is not None:
                    delta = x - self.moving_average
                    self.moving_average = self.moving_average + self.ALPHA_AVG * delta
                    self.moving_variance = (1 - self.ALPHA_STD) * (self.moving_variance + self.ALPHA_STD * delta**2)
                    moving_std = np.sqrt(self.moving_variance)
                    x = (x - self.moving_average) / (moving_std + self.EPSILON)
                else:
                    self.moving_average = x
            value[i] = x
        return value


class LiveDisplay():
    def __init__(self, channel_names, window_len=100):
        self.datapoint_dim = len(channel_names)
        self.history = []
        self.pp = ProgressPlot(plot_names=channel_names, max_window_len=window_len)
        self.matplotlib = False

    def add_datapoints(self, datapoints):
        """
        Adds 8 lists of datapoints to the plot
        
        Args:
            datapoints: list of 8 lists of floats (or list of 8 floats)
        """
        if self.matplotlib:
            import matplotlib.pyplot as plt
        disp_list = []
        for datapoint in datapoints:
            d = [[elt] for elt in datapoint]
            disp_list.append(d)
            
            if self.matplotlib:
                self.history += d[1]
        
        if not self.matplotlib:
            self.pp.update_with_datapoints(disp_list)
        elif len(self.history) == 1000:
            plt.plot(self.history)
            plt.show()
            self.history = []
    
    def add_datapoint(self, datapoint):
        disp_list = [[elt] for elt in datapoint]
        self.pp.update(disp_list)


def _capture_process(p_data_o, p_msg_io, duration, frequency, python_clock, time_msg_in, channel_states):
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


class DummyAlsaMixer:
    def __init__(self):
        self.volume = 50
    
    def getvolume(self):
        return [self.volume]
    
    def setvolume(self, volume):
        self.volume = volume


class Capture:
    def __init__(self, detector_cls=None, stimulator_cls=None):
        # {now.strftime('%m_%d_%Y_%H_%M_%S')}
        self.filename = EDF_PATH / 'recording.edf'
        self._p_capture = None
        self.__capture_on = False
        self.frequency = 250
        self.duration = 10
        self.power_line = 60
        self.polyak_mean = 0.1
        self.polyak_std = 0.001
        self.epsilon = 0.000001
        self.custom_fir = False
        self.custom_fir_order = 20
        self.custom_fir_cutoff = 30
        self.filter = True
        self.filter_args = [True, True, True]
        self.record = False
        self.detect = False
        self.stimulate = False
        self.threshold = 0.5
        self.lsl = False
        self.display = False
        self.python_clock = True
        self.edf_writer = None
        self.edf_buffer = []
        self.nb_signals = 8
        self.samples_per_datarecord_array = self.frequency
        self.physical_max = 5
        self.physical_min = -5
        self.signal_labels = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
        self._lock_msg_out = Lock()
        self._msg_out = None
        self._t_capture = None
        self.channel_states = ['disabled', 'disabled', 'disabled', 'disabled', 'disabled', 'disabled', 'disabled', 'disabled']
        self.channel_detection = 2
        
        self.detector_cls = detector_cls
        self.stimulator_cls = stimulator_cls
        
        self._test_stimulus_lock = Lock()
        self._test_stimulus = False
        
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
        
        
        # widgets ===============================
        
        # CHANNELS ------------------------------
        
        self.b_radio_ch1 = widgets.RadioButtons(
            options=['disabled', 'simple', 'with bias', 'bias out'],
            value='disabled',
            disabled=True
        )
        
        self.b_radio_ch2 = widgets.RadioButtons(
            options=['disabled', 'simple', 'with bias', 'bias out'],
            value='disabled',
            disabled=False
        )
        
        self.b_radio_ch3 = widgets.RadioButtons(
            options=['disabled', 'simple', 'with bias', 'bias out'],
            value='disabled',
            disabled=False
        )
        
        self.b_radio_ch4 = widgets.RadioButtons(
            options=['disabled', 'simple', 'with bias', 'bias out'],
            value='disabled',
            disabled=False
        )
        
        self.b_radio_ch5 = widgets.RadioButtons(
            options=['disabled', 'simple', 'with bias', 'bias out'],
            value='disabled',
            disabled=False
        )
        
        self.b_radio_ch6 = widgets.RadioButtons(
            options=['disabled', 'simple', 'with bias', 'bias out'],
            value='disabled',
            disabled=False
        )
        
        self.b_radio_ch7 = widgets.RadioButtons(
            options=['disabled', 'simple', 'with bias', 'bias out'],
            value='disabled',
            disabled=False
        )
        
        self.b_radio_ch8 = widgets.RadioButtons(
            options=['disabled', 'simple', 'with bias', 'bias out'],
            value='disabled',
            disabled=False
        )
        
        self.b_channel_detect = widgets.Dropdown(
            options=[('2', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6), ('7', 7), ('8', 8)],
            value=2,
            description='Detection Channel:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        
        self.b_accordion_channels = widgets.Accordion(
            children=[
                widgets.GridBox([
                    widgets.Label('CH1'),
                    widgets.Label('CH2'),
                    widgets.Label('CH3'),
                    widgets.Label('CH4'),
                    widgets.Label('CH5'),
                    widgets.Label('CH6'),
                    widgets.Label('CH7'),
                    widgets.Label('CH8'),
                    self.b_radio_ch1,
                    self.b_radio_ch2,
                    self.b_radio_ch3,
                    self.b_radio_ch4,
                    self.b_radio_ch5,
                    self.b_radio_ch6,
                    self.b_radio_ch7,
                    self.b_radio_ch8], layout=widgets.Layout(grid_template_columns="repeat(8, 90px)")
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
        
        self.b_clock = widgets.ToggleButtons(
            options=['Coral', 'ADS'],
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
        
        # CALLBACKS ----------------------
        
        self.b_capture.observe(self.on_b_capture, 'value')
        self.b_clock.observe(self.on_b_clock, 'value')
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
        self.b_radio_ch2.observe(self.on_b_radio_ch2, 'value')
        self.b_radio_ch3.observe(self.on_b_radio_ch3, 'value')
        self.b_radio_ch4.observe(self.on_b_radio_ch4, 'value')
        self.b_radio_ch5.observe(self.on_b_radio_ch5, 'value')
        self.b_radio_ch6.observe(self.on_b_radio_ch6, 'value')
        self.b_radio_ch7.observe(self.on_b_radio_ch7, 'value')
        self.b_channel_detect.observe(self.on_b_channel_detect, 'value')
        self.b_power_line.observe(self.on_b_power_line, 'value')
        self.b_custom_fir.observe(self.on_b_custom_fir, 'value')
        self.b_custom_fir_order.observe(self.on_b_custom_fir_order, 'value')
        self.b_custom_fir_cutoff.observe(self.on_b_custom_fir_cutoff, 'value')
        self.b_polyak_mean.observe(self.on_b_polyak_mean, 'value')
        self.b_polyak_std.observe(self.on_b_polyak_std, 'value')
        self.b_epsilon.observe(self.on_b_epsilon, 'value')
        self.b_volume.observe(self.on_b_volume, 'value')
        self.b_test_stimulus.on_click(self.on_b_test_stimulus)
        
        self.display_buttons()

    def __del__(self):
        self.b_capture.close()
    
    def display_buttons(self):
        display(widgets.VBox([self.b_accordion_channels,
                              self.b_channel_detect,
                              self.b_frequency,
                              self.b_duration,
                              self.b_filename,
                              self.b_power_line,
                              self.b_clock,
                              widgets.HBox([self.b_filter, self.b_detect, self.b_stimulate, self.b_record, self.b_lsl, self.b_display]),
                              widgets.HBox([self.b_threshold, self.b_test_stimulus]),
                              self.b_volume,
                              self.b_accordion_filter,
                              self.b_capture]))

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
        self.b_power_line.disabled = False
        self.b_channel_detect.disabled = False
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
        self.b_test_stimulus.disabled = True # only enabled when running
    
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
        self.b_channel_detect.disabled = True
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
    
    def on_b_radio_ch2(self, value):
        self.channel_states[1] = value['new']
    
    def on_b_radio_ch3(self, value):
        self.channel_states[2] = value['new']
    
    def on_b_radio_ch4(self, value):
        self.channel_states[3] = value['new']
    
    def on_b_radio_ch5(self, value):
        self.channel_states[4] = value['new']
    
    def on_b_radio_ch6(self, value):
        self.channel_states[5] = value['new']
    
    def on_b_radio_ch7(self, value):
        self.channel_states[6] = value['new']
        
    def on_b_channel_detect(self, value):
        self.channel_detection = value['new']

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
    
    def on_b_power_line(self, value):
        val = value['new']
        if val == '60 Hz':
            self.power_line = 60
        elif val == '50 Hz':
            self.python_clock = 50
    
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
        with self._test_stimulus_lock:
            self._test_stimulus = True
    
    def open_recording_file(self):
        nb_signals = self.nb_signals
        samples_per_datarecord_array = self.samples_per_datarecord_array
        physical_max = self.physical_max
        physical_min = self.physical_min
        signal_labels = self.signal_labels

        print(f"Will store edf recording in {self.filename}")

        self.edf_writer = EDFwriter(p_path=str(self.filename),
                                    f_file_type=EDFwriter.EDFLIB_FILETYPE_EDFPLUS,
                                    number_of_signals=nb_signals)
        
        for signal in range(nb_signals):
            assert self.edf_writer.setSampleFrequency(signal, samples_per_datarecord_array) == 0
            assert self.edf_writer.setPhysicalMaximum(signal, physical_max) == 0
            assert self.edf_writer.setPhysicalMinimum(signal, physical_min) == 0
            assert self.edf_writer.setDigitalMaximum(signal, 32767) == 0
            assert self.edf_writer.setDigitalMinimum(signal, -32768) == 0
            assert self.edf_writer.setSignalLabel(signal, signal_labels[signal]) == 0
            assert self.edf_writer.setPhysicalDimension(signal, 'V') == 0

    def close_recording_file(self):
        assert self.edf_writer.close() == 0
    
    def add_recording_data(self, data):
        self.edf_buffer += data
        if len(self.edf_buffer) >= self.samples_per_datarecord_array:
            datarecord_array = self.edf_buffer[:self.samples_per_datarecord_array]
            self.edf_buffer = self.edf_buffer[self.samples_per_datarecord_array:]
            datarecord_array = np.array(datarecord_array).transpose()
            assert len(datarecord_array) == self.nb_signals, f"len(data)={len(data)}!={self.nb_signals}"
            for d in datarecord_array:
                assert len(d) == self.samples_per_datarecord_array, f"{len(d)}!={self.samples_per_datarecord_array}"
                assert self.edf_writer.writeSamples(d) == 0

    def start_capture(self,
                      filter,
                      filter_args,
                      detector_cls,
                      threshold,
                      channel,
                      stimulator_cls,
                      record,
                      lsl,
                      viz,
                      width,
                      python_clock):
        if self.__capture_on:
            warnings.warn("Capture is already ongoing, ignoring command.")
            return
        else:
            self.__capture_on = True
            p_msg_io, p_msg_io_2 = mp.Pipe()
            p_data_i, p_data_o = mp.Pipe(duplex=False)
        SAMPLE_TIME = 1 / self.frequency

        if filter:
            fp = FilterPipeline(nb_channels=8,
                                sampling_rate=self.frequency,
                                power_line_fq=self.power_line,
                                use_custom_fir=self.custom_fir,
                                custom_fir_order=self.custom_fir_order,
                                custom_fir_cutoff=self.custom_fir_cutoff,
                                alpha_avg=self.polyak_mean,
                                alpha_std=self.polyak_std,
                                epsilon=self.epsilon,
                                filter_args=filter_args)
            
        detector = detector_cls(threshold, channel=channel) if detector_cls is not None else None
        stimulator = stimulator_cls() if stimulator_cls is not None else None

        self._p_capture = mp.Process(target=_capture_process,
                                     args=(p_data_o,
                                           p_msg_io_2,
                                           self.duration,
                                           self.frequency,
                                           python_clock,
                                           1.0,
                                           self.channel_states)
                                    )
        self._p_capture.start()
        print(f"PID capture: {self._p_capture.pid}")

        if viz:
            live_disp = LiveDisplay(channel_names = self.signal_labels, window_len=width)

        if record:
            self.open_recording_file()
        
        if lsl:
            from pylsl import StreamInfo, StreamOutlet
            lsl_info = StreamInfo(name='Portiloop',
                                  type='EEG',
                                  channel_count=8,
                                  nominal_srate=self.frequency,
                                  channel_format='float32',
                                  source_id='portiloop1')  # TODO: replace this by unique device identifier
            lsl_outlet = StreamOutlet(lsl_info)

        buffer = []

        while True:
            with self._lock_msg_out:
                if self._msg_out is not None:
                    p_msg_io.send(self._msg_out)
                    self._msg_out = None
            if p_msg_io.poll():
                mess = p_msg_io.recv()
                if mess == 'STOP':
                    break
                elif mess[0] == 'PRT':
                    print(mess[1])

            # retrieve all data points from p_data and put them in a list of np.array:
            point = None
            if p_data_i.poll(timeout=SAMPLE_TIME):
                point = p_data_i.recv()
            else:
                continue
                
            n_array = np.array([point])
            n_array = filter_np(n_array)
            
            if filter:
                n_array = fp.filter(n_array)
            
            filtered_point = n_array.tolist()
            
            if detector is not None:
                detection_signal = detector.detect(filtered_point)
                if stimulator is not None:
                    stimulator.stimulate(detection_signal)
                    with self._test_stimulus_lock:
                        test_stimulus = self._test_stimulus
                        self._test_stimulus = False
                    if test_stimulus:
                        stimulator.test_stimulus()
            
            if lsl:
                lsl_outlet.push_sample(filtered_point[-1])
            
            buffer += filtered_point
            if len(buffer) >= 50:

                if viz:
                    live_disp.add_datapoints(buffer)

                if record:
                    self.add_recording_data(buffer)
                    
                buffer = []

        # empty pipes
        while True:
            if p_data_i.poll():
                _ = p_data_i.recv()
            elif p_msg_io.poll():
                _ = p_msg_io.recv()
            else:
                break

        p_data_i.close()
        p_msg_io.close()
        
        if record:
            self.close_recording_file()

        self._p_capture.join()
        self.__capture_on = False


if __name__ == "__main__":
    pass
