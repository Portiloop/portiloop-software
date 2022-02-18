from time import sleep
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime, timedelta
import multiprocessing as mp
import warnings
import shutil
from threading import Thread, Lock

import matplotlib.pyplot as plt
from EDFlib.edfwriter import EDFwriter

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
    0xD0, # CONFIG2 [C0] [1, 1, 0, INT_CAL, 0, CAL_AMP0, CAL_FREQ[1:0]]
    0xE0, # CONFIG3 [E0] [PD_REFBUF(bar), 1, 1, BIAS_MEAS, BIASREF_INT, PD_BIAS(bar), BIAS_LOFF_SENS, BIAS_STAT] : Power-down reference buffer, no bias
    0x00, # No lead-off
    0x03, # CH1SET [60] [PD1, GAIN1[2:0], SRB2, MUX1[2:0]]
    0x00, # CH2SET
    0x00, # CH3SET
    0x00, # CH4SET
    0x00, # CH5SET voltage
    0x00, # CH6SET voltage
    0x00, # CH7SET test
    0x04, # CH8SET temperature
    0x00, # BIAS_SENSP
    0x00, # BIAS_SENSN
    0xFF, # LOFF_SENSP Lead-off on all positive pins?
    0xFF, # LOFF_SENSN Lead-off on all negative pins?
    0x00, # Normal lead-off
    0x00, # Lead-off positive status (RO)
    0x00, # Lead-off negative status (RO)
    0x00, # All GPIOs as output ?
    0x20, # Enable SRB1
]

EDF_PATH = Path.home() / 'workspace' / 'edf_recording'

def mod_config(config, datarate):
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
    new_cf1 = new_cf1 | j
    config[1] = new_cf1
    return config

def filter_24(value):
    return (value * 4.5) / (2**23 - 1)  # 23 because 1 bit is lost for sign

def filter_2scomplement_np(value):
    return np.where((value & (1 << 23)) != 0, value - (1 << 24), value)

def filter_np(value):
    return filter_24(filter_2scomplement_np(value))

class LiveDisplay():
    def __init__(self, channel_names, window_len=100):
        self.datapoint_dim = len(channel_names)
        self.pp = ProgressPlot(plot_names=channel_names, max_window_len=window_len)

    def add_datapoints(self, datapoints):
        """
        Adds 8 lists of datapoints to the plot
        
        Args:
            datapoints: list of 8 lists of floats (or list of 8 floats)
        """
        disp_list = []
        for datapoint in datapoints:
            d = [[elt] for elt in datapoint]
            disp_list.append(d)
        self.pp.update_with_datapoints(disp_list)
    
    def add_datapoint(self, datapoint):
        disp_list = [[elt] for elt in datapoint]
        self.pp.update(disp_list)


def _capture_process(p_data_o, p_msg_io, duration, frequency, python_clock=True, time_msg_in=1.0):
    """
    Args:
        p_data_o: multiprocessing.Pipe: captured datapoints are put here
        p_msg_io: mutliprocessing.Pipe: to communicate with the parent process
        duration: float: max duration of the experiment in seconds
        frequency: float: sampling frequency
        ptyhon_clock: bool (default True): if True, the Coral clock is used, otherwise, the ADS interrupts are used
        time_msg_in: float (default 1.0): min time between attempts to recv incomming messages
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
        assert data == [0x3E], "The communication with the ADS cannot be established."
        leds.led2(Color.BLUE)
        
        config = FRONTEND_CONFIG
        if python_clock:  # set ADS to 2 * frequency
            config = mod_config(config, 2 * frequency)
        else:  # set ADS to frequency
            config = mod_config(config, frequency)
        
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

        leds.aquisition(False)

    finally:
        frontend.close()
        leds.close()
        p_msg_io.send('STOP')
        p_msg_io.close()
        p_data_o.close()

        
class Capture:
    def __init__(self):
        # {now.strftime('%m_%d_%Y_%H_%M_%S')}
        self.filename = EDF_PATH / 'recording.edf'
        self._p_capture = None
        self.__capture_on = False
        self.frequency = 250
        self.duration = 10
        self.record = False
        self.display = False
        self.python_clock = True
        self.edf_writer = None
        self.edf_buffer = []
        self.nb_signals = 8
        self.samples_per_datarecord_array = self.frequency
        self.physical_max = 5
        self.physical_min = -5
        self.signal_labels = ['voltage', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'temperature']
        self._lock_msg_out = Lock()
        self._msg_out = None
        self._t_capture = None
        
        # widgets
        
        self.b_capture = widgets.ToggleButtons(
            options=['Stop', 'Start'],
            description='Capture:',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Stop capture', 'Start capture'],
            # icons=['check'] * 2
        )
        
        self.b_clock = widgets.ToggleButtons(
            options=['Coral', 'ADS'],
            description='Clock:',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Use Coral clock (very precise, not very timely)',
                      'Use ADS clock (not very precise, very timely)'],
            # icons=['check'] * 2
        )
        
        self.b_filename = widgets.Text(
            value='recording.edf',
            description='Recording:',
            disabled=False
        )
        
        self.b_frequency = widgets.IntText(
            value=250,
            description='Freq (Hz):',
            disabled=False
        )
        
        self.b_duration = widgets.IntText(
            value=10,
            description='Time (s):',
            disabled=False
        )
        
        self.b_record = widgets.Checkbox(
            value=False,
            description='Record',
            disabled=False,
            indent=False
        )
        
        self.b_display = widgets.Checkbox(
            value=False,
            description='Display',
            disabled=False,
            indent=False
        )
        
        self.b_capture.observe(self.on_b_capture, 'value')
        self.b_clock.observe(self.on_b_clock, 'value')
        self.b_frequency.observe(self.on_b_frequency, 'value')
        self.b_duration.observe(self.on_b_duration, 'value')
        self.b_record.observe(self.on_b_record, 'value')
        self.b_display.observe(self.on_b_display, 'value')
        self.b_filename.observe(self.on_b_filename, 'value')
        
        self.display_buttons()

    def __del__(self):
        self.b_capture.close()
    
    def display_buttons(self):
        display(widgets.VBox([self.b_frequency,
                              self.b_duration,
                              self.b_filename,
                              widgets.HBox([self.b_record, self.b_display]),
                              self.b_clock,
                              self.b_capture]))

    def enable_buttons(self):
        self.b_frequency.disabled = False
        self.b_duration.disabled = False
        self.b_filename.disabled = False
        self.b_record.disabled = False
        self.b_display.disabled = False
        self.b_clock.disabled = False
    
    def disable_buttons(self):
        self.b_frequency.disabled = True
        self.b_duration.disabled = True
        self.b_filename.disabled = True
        self.b_record.disabled = True
        self.b_display.disabled = True
        self.b_clock.disabled = True

    def on_b_capture(self, value):
        val = value['new']
        if val == 'Start':
            clear_output()
            self.disable_buttons()
            self.display_buttons()
            with self._lock_msg_out:
                self._msg_out = None
            if self._t_capture is not None:
                warnings.warn("Capture already running, operation aborted.")
                return
            self._t_capture = Thread(target=self.start_capture,
                                args=(self.record, self.display, 500, self.python_clock))
            self._t_capture.start()
#             self.start_capture(
#                 record=self.record,
#                 viz=self.display,
#                 width=500,
#                 python_clock=self.python_clock)
        elif val == 'Stop':
            with self._lock_msg_out:
                self._msg_out = 'STOP'
            assert self._t_capture is not None
            self._t_capture.join()
            self._t_capture = None
            self.enable_buttons()
    
    def on_b_clock(self, value):
        val = value['new']
        if val == 'Coral':
            self.python_clock = True
        elif val == 'ADS':
            self.python_clock = False
    
    def on_b_frequency(self, value):
        val = value['new']
        if val > 0:
            self.frequency = val
            
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
    
    def on_b_record(self, value):
        val = value['new']
        self.record = val
    
    def on_b_display(self, value):
        val = value['new']
        self.display = val
    
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
                      record,
                      viz,
                      width,
                      python_clock):
        
        p_msg_io, p_msg_io_2 = mp.Pipe()
        p_data_i, p_data_o = mp.Pipe(duplex=False)

        if self.__capture_on:
            warnings.warn("Capture is already ongoing, ignoring command.")
            return
        else:
            self.__capture_on = True
        SAMPLE_TIME = 1 / self.frequency
        self._p_capture = mp.Process(target=_capture_process,
                                     args=(p_data_o,
                                           p_msg_io_2,
                                           self.duration,
                                           self.frequency,
                                           python_clock)
                                    )
        self._p_capture.start()

        if viz:
            live_disp = LiveDisplay(channel_names = self.signal_labels, window_len=width)

        if record:
            self.open_recording_file()

        cc = True
        while cc:
            with self._lock_msg_out:
                if self._msg_out is not None:
                    p_msg_io.send(self._msg_out)
                    self._msg_out = None
            if p_msg_io.poll():
                mess = p_msg_io.recv()
                if mess == 'STOP':
                    cc = False
                elif mess[0] == 'PRT':
                    print(mess[1])

            # retrieve all data points from p_data and put them in a list of np.array:
            res = []
            c = True
            while c and len(res) < 25:
                if p_data_i.poll(timeout=SAMPLE_TIME):
                    point = p_data_i.recv()
                    res.append(point)
                else:
                    c = False
            if len(res) == 0:
                continue

            n_array = np.array(res)
            n_array = filter_np(n_array)

            to_add = n_array.tolist()

            if viz:
                live_disp.add_datapoints(to_add)
            if record:
                self.add_recording_data(to_add)

        # empty pipes
        cc = True 
        while cc:
            if p_data_i.poll():
                _ = p_data_i.recv()
            elif p_msg_io.poll():
                _ = p_msg_io.recv()
            else:
                cc = False

        p_data_i.close()
        p_msg_io.close()
        
        if record:
            self.close_recording_file()

        self._p_capture.join()
        self.__capture_on = False


if __name__ == "__main__":
    # TODO: Argparse this
    pass