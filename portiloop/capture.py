from time import sleep
import time
from playsound import playsound
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime, timedelta
import multiprocessing as mp
from queue import Empty
import warnings
import shutil
from pyedflib import highlevel
from datetime import datetime

import matplotlib.pyplot as plt

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
    print(f"DEBUG: new cf1: {hex(config[1])}")
    return config

def filter_24(value):
    return (value * 4.5) / (2**23 - 1)  # 23 because 1 bit is lost for sign

def filter_2scomplement_np(value):
    v = np.where((value & (1 << 23)) != 0, value - (1 << 24), value)
    return filter_24(v)

class LiveDisplay():
    def __init__(self, datapoint_dim=8, window_len=100):
        self.datapoint_dim = datapoint_dim
        self.queue = mp.Queue()
        channel_names = [f"channel#{i+1}" for i in range(datapoint_dim)]
        channel_names[0] = "voltage"
        channel_names[7] = "temperature"
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


def _capture_process(q_data, q_out, q_in, duration, frequency, python_clock=True):
    """
    Args:
        q_data: multiprocessing.Queue: captured datapoints are put in the queue
        q_out: mutliprocessing.Queue: to pass messages to the parent process
            'STOP': end of the the process
        q_in: mutliprocessing.Queue: to pass messages from the parent process
            'STOP': stops the process
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
        q_data.put(datapoint)
        
        t_next = t + sample_time
        
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
            q_data.put(datapoint)
            

            # Check for messages  # this takes too long :/
#             try:
#                 message = q_in.get_nowait()
#                 if message == 'STOP':
#                     c = False
#             except Empty:
#                 pass
            it += 1
        t = time.time()
        tot = (t - t_start) / it        

        print(f"Average frequency: {1 / tot} Hz for {it} samples")

        leds.aquisition(False)

    finally:
        frontend.close()
        leds.close()
        q_in.close()
        q_out.put('STOP')

        
class Capture:
    def __init__(self):
        
        self.filename = Path.home() / 'edf_recording' / f"recording_{now.strftime('%m_%d_%Y_%H_%M_%S')}.edf"
        self._p_capture = None
        self.__capture_on = False
        self.frequency = 250
        self.duration = 10
        self.record = False
        self.display = False
        self.recording_file = None
        self.python_clock = True
        
        self.binfile = None
        self.temp_path = Path.home() / '.temp'
        
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
            value=self.filename,
            description='Filename:',
            placeholder='All files will be in the edf_recording folder'
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
                              widgets.HBox([self.b_record, self.b_display]),
                              self.b_clock,
                              self.b_capture]))
    
    def on_b_capture(self, value):
        val = value['new']
        if val == 'Start':
            self.start_capture(
                record=self.record,
                viz=self.display,
                width=500,
                python_clock=self.python_clock)
        elif val == 'Stop':
            clear_output()
            self.display_buttons()
        else:
            print(f"This option is not supported: {val}.")
    
    def on_b_clock(self, value):
        val = value['new']
        if val == 'Coral':
            self.python_clock = True
        elif val == 'ADS':
            self.python_clock = False
        else:
            print(f"This option is not supported: {val}.")
    
    def on_b_frequency(self, value):
        val = value['new']
        if val > 0:
            self.frequency = val
        else:
            print(f"Unsupported frequency: {val} Hz")
            
    def on_b_filename(self, value):
        val = value['new']
        if val != '':
            self.filename = Path.home() / 'edf_recording' / val
        else:
            now = datetime.now()
            self.filename = Path.home() / 'edf_recording' / f"recording_{now.strftime('%m_%d_%Y_%H_%M_%S')}.edf"
        
    def on_b_duration(self, value):
        val = value['new']
        if val > 0:
            self.duration = val
        else:
            print(f"Unsupported duration: {val} s")
    
    def on_b_record(self, value):
        val = value['new']
        self.record = val
    
    def on_b_display(self, value):
        val = value['new']
        self.display = val
    
    def open_recording_file(self):
        print(f"Will store edf recording in {self.filename}")
        os.mkdir(self.temp_path)
        self.binfile = open(self.temp_path / 'data.bin', 'wb')
    
    def close_recording_file(self):
        
        print('Saving recording data...')
        # Channel names
        channels = ['Voltage', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Temperature']
        
        # Read binary data
        data = np.fromfile(self.temp_path / 'data.bin', dtype=float)
        data = data.reshape((8, int(data.shape[0]/8)))
        
        # Declare and write EDF format file
        signal_headers = highlevel.make_signal_headers(channels, sample_frequency=self.frequency)
        header = highlevel.make_header(patientname='patient_x', gender='Female')
        highlevel.write_edf(self.filename, data, signal_headers, header)
        
        # Close and delete temp binary file
        self.binfile.close()        
        shutil.rmtree(self.temp_path)
        
        print('...done')
    
    def add_recording_data(self, data):
        np.array(data).tofile(self.binfile)

    def start_capture(self,
                      record=True,
                      viz=False,
                      width=500,
                      python_clock=True):
        self.q_messages_send = mp.Queue()
        self.q_messages_recv = mp.Queue()
        self.q_data = mp.Queue()

        if self.__capture_on:
            print("Capture is already ongoing, ignoring command.")
            return
        else:
            self.__capture_on = True
        SAMPLE_TIME = 1 / frequency
        self._p_capture = mp.Process(target=_capture_process, args=(self.q_data,
                                                                    self.q_messages_recv,
                                                                    self.q_messages_send,
                                                                    self.duration,
                                                                    self.frequency,
                                                                    python_clock))
        self._p_capture.start()
        
        if viz:
            live_disp = LiveDisplay(window_len=width)
        
        if record:
            self.open_recording_file()

        cc = True
        while cc:
            try:
                mess = self.q_messages_recv.get_nowait()
                if mess == 'STOP':
                    cc = False
            except Empty:
                pass

            # retrieve all data points from q_data and put them in a list of np.array:
            res = []
            c = True
            while c and len(res) < 25:
                try:
                    point = self.q_data.get(timeout=SAMPLE_TIME)
                    res.append(point)
                except Empty:
                    c = False
            if len(res) == 0:
                continue
            n_array = np.array(res)
            n_array = filter_2scomplement_np(n_array)
            
            to_add = n_array.tolist()
            
            if viz:
                live_disp.add_datapoints(to_add)
            if record:
                self.add_recording_data(to_add)
        
        # empty q_data
        cc = True 
        while cc:
            try:
                _ = self.q_data.get_nowait()
            except Empty:
                cc = False

        self.q_messages_recv.close()
        self.q_data.close()
        
        if record:
            self.close_recording_file()

        # print("DEBUG: joining capture process...")
        self._p_capture.join()
        # print("DEBUG: capture process joined.")
        self.__capture_on = False


if __name__ == "__main__":
    # TODO: Argparse this
    pass