"""
This file defines the core logic of Portiloop pipelines in start_capture().

Multichannel signal is captured from the electrodes in a dedicated python process.
Then, it is passed asynchronously to a Processor, a Detector and a Stimulator.
The Processor is in charge of signal processing / filtering.
The Detector detects patterns of interest in the processed signal.
The Stimulator produces stimuli depending on what the Detector found.
"""

import json
import queue  # for exceptions
import os
import time
import numpy as np
from copy import deepcopy
import socket

from portiloop.src.core.hardware.leds import Color, LEDs
from portiloop.src.core.hardware.config_hardware import mod_config, BACKEND_CONFIG
from portiloop.src.core.utils import Dummy, get_portiloop_version
from portiloop.src.core.output import CSVRecorder, LiveDisplay, LSLStreamer
from portiloop.src.core.capture_backend import ADSBackend, FileBackend
from portiloop.src.core.constants import SIGNAL_SAMPLES_FOLDER

from portiloop.src import ADS
if ADS:
    from portiloop.src.core.hardware.backend import Backend


PORTILOOP_ID = f"{socket.gethostname()}-portiloop"
PROFILE = True


def capture_process(p_data_o, p_msg_io, duration, frequency, python_clock, time_msg_in, channel_states):
    """
    Args:
        p_data_o: multiprocessing.Pipe: captured datapoints are put here
        p_msg_io: multiprocessing.Pipe: to communicate with the parent process
        duration: float: max duration of the experiment in seconds
        frequency: float: sampling frequency
        python_clock: bool: if True, the Coral clock is used, otherwise, the ADS interrupts are used
        time_msg_in: float: min time between attempts to recv incomming messages
        channel_states: list: list of strings representing channel states ('disabled', 'simple', etc.)
    """
    if duration <= 0:
        duration = np.inf
    
    sample_time = 1 / frequency

    version = get_portiloop_version()
    backend = Backend(version)
    
    try:
        config = BACKEND_CONFIG
        if python_clock:  # set ADS to 2 * frequency
            datarate = 2 * frequency
        else:  # set ADS to frequency
            datarate = frequency
        config = mod_config(config, datarate, channel_states)
        
        backend.write_regs(0x00, config)
        # data = backend.read_regs(0x00, len(config))

        c = True

        it = 0
        t_start = time.time()
        t_max = t_start + duration
        t = t_start
        
        # first sample:
        reading = backend.read()
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
                reading = backend.read()
            else:
                reading = backend.wait_new_data()
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
        processor_cls,
        detector_cls,
        stimulator_cls,
        config_dict,
        q_msg,
        q_display,
        pause_value):
    """
    Method called by Portiloop GUIs to launch a custom Portiloop pipeline.
    start_capture() defines the core logic of all Portiloop pipelines.
    The output of custom Detectors is the input of custom Stimulators, and is thus arbitrary.
    All configuration arguments are passed to the Processors, Detectors and stimulators via the capture_dictionary.

    Args:
        processor_cls: Processor subclass or None: custom Processor for signal processing
        detector_cls: Detector subclass or None: custom Detector for detection in the processed signal
        stimulator_cls: Stimulator subclass or None: custom Stimulator for stimulation from the detection signal
        config_dict: Dict: Arguments of the pipeline.
        q_msg: Queue
        q_display: Queue
        pause_value: multiprocessing.Value: whether Detection is currently paused.

    Returns:
        None

    """

    # print(f"DEBUG: Channel states: {config_dict['channel_states']}")

    # Initialize the LED
    leds = LEDs()
    if config_dict['stimulate']:
        leds.led1(Color.CYAN)
    else:
        leds.led1(Color.PURPLE)

    # Initialize data backend
    signal_sample = SIGNAL_SAMPLES_FOLDER / config_dict["signal_sample"]  # 'test_spindles.csv'  # test_slow_oscillations.csv
    capture_backend = ADSBackend(
        duration=config_dict['duration'],
        frequency=config_dict['frequency'],
        python_clock=config_dict['python_clock'],
        channel_states=config_dict['channel_states'],
        vref=config_dict['vref'],
        process=capture_process,
    ) if config_dict['signal_input'] == "ADS" else FileBackend(signal_sample, config_dict['nb_channels'], config_dict['channel_detection'], config_dict['frequency'])

    # Initialize detector, LSL streamer and stimulatorif requested
    streams = {
        'filtered': config_dict['filter'],
        'markers': config_dict['detect'],
    }
    lsl_streamer = LSLStreamer(streams, config_dict['nb_channels'], config_dict['frequency'], id=PORTILOOP_ID) if config_dict['lsl'] else Dummy()
    
    # Launch the capture process
    capture_backend.init_capture()

    # Initialize display if requested
    live_disp_activated = config_dict['display']
    live_disp = LiveDisplay(channel_names=config_dict['signal_labels'], window_len=config_dict['width_display']) if live_disp_activated else Dummy()

    create_processor = config_dict['filter'] and processor_cls is not None
    create_detector = config_dict['detect'] and detector_cls is not None
    create_stimulator = config_dict['detect'] and stimulator_cls is not None  # TODO: allow manual stimulation without detection

    # Initialize recording if requested
    if config_dict['record']:
        csv_recorder = CSVRecorder(config_dict['filename'],
                                   raw_signal=True,
                                   filtered_signal=create_processor,  # set to False if you don't want to log the filtered signal
                                   detection_signal=create_detector,
                                   stimulation_signal=create_stimulator,
                                   detection_activated=False,  # stimulation activated is enough
                                   stimulation_activated=True,
                                   default_detection_value=0,
                                   default_stimulation_value=0)
    else:
        csv_recorder = Dummy()

    # Pipeline components:

    detector = detector_cls(config_dict, lsl_streamer, csv_recorder) if create_detector else None
    stimulator = stimulator_cls(config_dict, lsl_streamer, csv_recorder) if create_stimulator else None
    if create_processor:
        processor = processor_cls(config_dict, lsl_streamer, csv_recorder)
    else:
        processor = None

    # Buffer used for the visualization and the recording
    raw_signal_buffer = []
    filtered_signal_buffer = []
    # detection_signal_buffer = []
    stimulation_activated_buffer = []

    # Get the metadata and save it to a file
    metadata = config_dict
    # Split the original path into its components
    dirname, basename = os.path.split(config_dict['filename'])
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

    if PROFILE:
        perf = {"wait msg": [0, 0],
                "no data": [0, 0],
                "got data": [0, 0],
                "filter": [0, 0],
                "lsl": [0, 0],
                "detect": [0, 0],
                "stimulate": [0, 0],
                "buffers": [0, 0],
                "display": [0, 0],
                "csv": [0, 0]}
        t0 = time.perf_counter()

    # Main capture loop
    while True:

        if PROFILE:
            t00 = time.perf_counter()
        
        # First, we send all outgoing messages to the capture process
        try:
            msg = q_msg.get_nowait()
            capture_backend.send_msg(msg)
        except queue.Empty as e:
            pass
        except queue.ShutDown as e:
            raise e
        
        # Then, we check if we have received a message from the capture process
        msg = capture_backend.get_msg()
        # Either we have received a stop message, or a print message.
        if msg is None:
            pass
        elif msg == 'STOP':
            break
        elif msg[0] == 'PRT':
            print(msg[1])

        if PROFILE:
            t1 = time.perf_counter()
            perf["wait msg"][0] += t1 - t00
            perf["wait msg"][1] += 1

        # Then, we retrieve the data from the capture process
        raw_points = capture_backend.get_data()  # np.array (data series x ads_channels), or None
        # If we have no data, we continue to the next iteration
        if raw_points is None:
            if PROFILE:
                t1_1 = time.perf_counter()
                perf["no data"][0] += t1_1 - t1
                perf["no data"][1] += 1
            continue

        if PROFILE:
            t2 = time.perf_counter()
            perf["got data"][0] += t2 - t1
            perf["got data"][1] += 1
        
        # Go through filtering pipeline
        if processor is not None:
            filtered_points = processor.filter(deepcopy(raw_points))
        else:
            filtered_points = deepcopy(raw_points)

        # Contains the filtered points (if filtering is off, contains a copy of the raw points)
        filtered_points = filtered_points.tolist()
        raw_points = raw_points.tolist()

        if PROFILE:
            t3 = time.perf_counter()
            perf["filter"][0] += t3 - t2
            perf["filter"][1] += 1

        # Send both the latest raw and filtered points over LSL
        lsl_streamer.push_raw(raw_points[-1])
        if processor is not None:
            lsl_streamer.push_filtered(filtered_points[-1])
        
        # Check if detection is on or off
        pause = pause_value.value

        # If the state has changed since last iteration, we send a marker
        if pause != prev_pause and detector is not None:
            lsl_streamer.push_marker(LSLStreamer.string_for_detection_activation(pause))
            prev_pause = pause

        if PROFILE:
            t4 = time.perf_counter()
            perf["lsl"][0] += t4 - t3
            perf["lsl"][1] += 1

        stimulator_activated = False
        # If detection is on
        if detector is not None and not pause:
            # Detect using the latest points
            # (Note for core developers: detection_signal is arbitrary. Please do not assume anything about it here.)
            detection_signal = detector.detect(filtered_points)

            if PROFILE:
                t5 = time.perf_counter()
                perf["detect"][0] += t5 - t4
                perf["detect"][1] += 1

            # Stimulate
            if stimulator is not None:
                stimulator_activated = True
                stimulator.stimulate(detection_signal)

                if PROFILE:
                    t6 = time.perf_counter()
                    perf["stimulate"][0] += t6 - t5
                    perf["stimulate"][1] += 1

        if PROFILE:
            t7 = time.perf_counter()

        # Add point to the buffer to send to viz and recorder
        raw_signal_buffer += raw_points
        if processor is not None:
            filtered_signal_buffer += filtered_points
        if stimulator is not None:
            stimulation_activated_buffer += [stimulator_activated] * len(raw_points)

        # Adding the raw point and its timestamp for display
        timestamp = time.time() - start_time
        if q_display is not None:
            q_display.put([timestamp, raw_points, filtered_points])

        if PROFILE:
            t8 = time.perf_counter()
            perf["buffers"][0] += t8 - t7
            perf["buffers"][1] += 1

        if len(raw_signal_buffer) >= 50:  # TODO: make this an argument
            # TODO: give the option to display either raw or filtered signal in live_disp
            live_disp.add_datapoints(raw_signal_buffer)
            # if processor is not None:
            #     live_disp.add_datapoints(filtered_signal_buffer)
            # else:
            #     live_disp.add_datapoints(raw_signal_buffer)

            if PROFILE:
                t9 = time.perf_counter()
                perf["display"][0] += t9 - t8
                perf["display"][1] += 1

            csv_recorder.append_raw_signal_buffer(raw_signal_buffer)
            csv_recorder.append_filtered_signal_buffer(filtered_signal_buffer)
            csv_recorder.append_stimulation_activated_buffer(stimulation_activated_buffer)
            csv_recorder.write()

            raw_signal_buffer = []
            filtered_signal_buffer = []
            stimulation_activated_buffer = []

            if PROFILE:
                t10 = time.perf_counter()
                perf["csv"][0] += t10 - t9
                perf["csv"][1] += 1

    if PROFILE:
        t_end = time.perf_counter()
        print(f"Performance summary:")
        tt = 0
        for k, v in perf.items():
            if v[1] == 0:
                continue
            tot = v[0]
            avg = tot / v[1]
            print(f"{k}: {tot} (avg: {avg*1000} ms/call)")
            tt += tot
        print(f"total measured time: {tt} vs real: {t_end - t0}")

    # close the backend
    leds.led1(Color.YELLOW)
    capture_backend.close()
    leds.close()

    del csv_recorder
    del lsl_streamer
    del stimulator
    del detector


if __name__ == "__main__":
    pass
