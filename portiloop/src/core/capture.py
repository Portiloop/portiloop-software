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
from portiloop.src.core.hardware import Color, LEDs

if ADS:
    import alsaaudio
    from alsaaudio import ALSAAudioError
    from portiloop.src.core.hardware import Frontend

from portiloop.src.core.stimulation import TimingDelayer, UpStateDelayer

from portiloop.src.core.processing import FilterPipeline
from portiloop.src.core.hardware.config_hardware import mod_config, LEADOFF_CONFIG, FRONTEND_CONFIG, to_ads_frequency
from portiloop.src.custom.constants import CSV_PATH, RECORDING_PATH, CALIBRATION_PATH
from portiloop.src.core.utils import ADSFrontend, Dummy, FileFrontend, LSLStreamer, LiveDisplay, DummyAlsaMixer, CSVRecorder, get_portiloop_version
from portiloop.src.custom.constants import RUN_SETTINGS

from IPython.display import clear_output, display
import ipywidgets as widgets
import socket
from pathlib import Path


PORTILOOP_ID = f"{socket.gethostname()}-portiloop"


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
                    if message == 'STOP':
                        c = False
            it += 1
        t = time.time()
        tot = (t - t_start) / it        

        p_msg_io.send(("PRT", f"Average frequency: {1 / tot} Hz for {it} samples"))

    finally:
        p_msg_io.send('STOP')
        p_msg_io.close()
        p_data_o.close()
    
def start_capture(
        detector_cls,
        stimulator_cls,
        capture_dictionary,
        q_msg, 
        q_display,
        pause_value
): 
    # print(f"DEBUG: Channel states: {capture_dictionary['channel_states']}")

    # Initialize the LED
    leds = LEDs()
    if capture_dictionary['stimulate']:
        leds.led1(Color.CYAN)
    else:
        leds.led1(Color.PURPLE)

    # Initialize data frontend
    fake_filename = RECORDING_PATH / 'test_recording.csv'
    capture_frontend = ADSFrontend(
        duration=capture_dictionary['duration'],
        frequency=capture_dictionary['frequency'],
        python_clock=capture_dictionary['python_clock'],
        channel_states=capture_dictionary['channel_states'],
        vref=capture_dictionary['vref'],
        process=capture_process,
    ) if capture_dictionary['signal_input'] == "ADS" else FileFrontend(fake_filename, capture_dictionary['nb_channels'], capture_dictionary['channel_detection'])

    # Initialize detector, LSL streamer and stimulatorif requested
    detector = detector_cls(capture_dictionary['threshold'], channel=capture_dictionary['channel_detection']) if capture_dictionary['detect'] else None
    streams = {
            'filtered': filter,
            'markers': detector is not None,
        }

    lsl_streamer = LSLStreamer(streams, capture_dictionary['nb_channels'], capture_dictionary['frequency'], id=PORTILOOP_ID) if capture_dictionary['lsl'] else Dummy()
    stimulator = stimulator_cls(soundname=capture_dictionary['detection_sound'], lsl_streamer=lsl_streamer,sham=not capture_dictionary['stimulate']) if stimulator_cls is not None else None
    # Initialize filtering pipeline
    if filter:
        fp = FilterPipeline(nb_channels=capture_dictionary['nb_channels'],
                            sampling_rate=capture_dictionary['frequency'],
                            power_line_fq=capture_dictionary['filter_settings']['power_line'],
                            use_custom_fir=capture_dictionary['filter_settings']['custom_fir'],
                            custom_fir_order=capture_dictionary['filter_settings']['custom_fir_order'],
                            custom_fir_cutoff=capture_dictionary['filter_settings']['custom_fir_cutoff'],
                            alpha_avg=capture_dictionary['filter_settings']['polyak_mean'],
                            alpha_std=capture_dictionary['filter_settings']['polyak_std'],
                            epsilon=capture_dictionary['filter_settings']['epsilon'],
                            filter_args=capture_dictionary['filter_settings']['filter_args'])
    
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
    delay = not ((capture_dictionary['stim_delay'] == 0.0) and (capture_dictionary['inter_stim_delay'] == 0.0)) and (stimulator is not None)
    delay_phase = (not delay) and (not capture_dictionary['spindle_detection_mode'] == 'Fast') and (stimulator is not None)
    if delay:
        stimulation_delayer = TimingDelayer(
            stimulation_delay=capture_dictionary['stim_delay'],
            inter_stim_delay=capture_dictionary['inter_stim_delay']
        )
    elif delay_phase:
        stimulation_delayer = UpStateDelayer(
            capture_dictionary['frequency'], 
            capture_dictionary['spindle_detection_mode'] == 'Peak', 0.3)
    else:
        stimulation_delayer = Dummy()
        
    if stimulator is not None:
        stimulator.add_delayer(stimulation_delayer)

    # Get the metadata and save it to a file
    metadata = capture_dictionary
    # Split the original path into its components
    dirname, basename = os.path.split(capture_dictionary['filename'])
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
        elif msg == 'STOP':
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
                stimulation_delayer.step(filtered_point[0][capture_dictionary['channel_detection'] - 1])

        # Add point to the buffer to send to viz and recorder
        buffer += raw_point

        # Adding the raw point an it's timestamp for display
        timestamp = time.time() - start_time
        if q_display is not None:
            q_display.put([timestamp, raw_point, filtered_point])

        if len(buffer) >= 50:
            live_disp.add_datapoints(buffer)
            recorder.add_recording_data(buffer, detection_buffer, capture_dictionary['detect'], capture_dictionary['stimulate'])
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


if __name__ == "__main__":
    pass
