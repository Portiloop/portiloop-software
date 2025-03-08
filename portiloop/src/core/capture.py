import json
import queue  # for exceptions
import os
import time
import numpy as np
from copy import deepcopy
import socket

from portiloop.src.core.hardware.leds import Color, LEDs
from portiloop.src.custom.custom_stimulators import TimingDelayer, UpStateDelayer
from portiloop.src.core.hardware.config_hardware import mod_config, FRONTEND_CONFIG
from portiloop.src.core.utils import ADSFrontend, Dummy, FileFrontend, LSLStreamer, LiveDisplay, CSVRecorder, get_portiloop_version
from portiloop.src.core.constants import RECORDING_FOLDER

from portiloop.src import ADS
if ADS:
    from portiloop.src.core.hardware.frontend import Frontend


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
        processor_cls,
        detector_cls,
        stimulator_cls,
        capture_dictionary,
        q_msg, 
        q_display,
        pause_value):

    # print(f"DEBUG: Channel states: {capture_dictionary['channel_states']}")

    # Initialize the LED
    leds = LEDs()
    if capture_dictionary['stimulate']:
        leds.led1(Color.CYAN)
    else:
        leds.led1(Color.PURPLE)

    # Initialize data frontend
    fake_filename = RECORDING_FOLDER / 'test_recording.csv'
    capture_frontend = ADSFrontend(
        duration=capture_dictionary['duration'],
        frequency=capture_dictionary['frequency'],
        python_clock=capture_dictionary['python_clock'],
        channel_states=capture_dictionary['channel_states'],
        vref=capture_dictionary['vref'],
        process=capture_process,
    ) if capture_dictionary['signal_input'] == "ADS" else FileFrontend(fake_filename, capture_dictionary['nb_channels'], capture_dictionary['channel_detection'])

    # Initialize detector, LSL streamer and stimulatorif requested
    streams = {
        'filtered': capture_dictionary['filter'],
        'markers': capture_dictionary['detect'],
    }
    lsl_streamer = LSLStreamer(streams, capture_dictionary['nb_channels'], capture_dictionary['frequency'], id=PORTILOOP_ID) if capture_dictionary['lsl'] else Dummy()

    detector = detector_cls(capture_dictionary, lsl_streamer) if capture_dictionary['detect'] else None
    stimulator = stimulator_cls(capture_dictionary, lsl_streamer) if stimulator_cls is not None else None

    # Initialize filtering pipeline
    if capture_dictionary['filter']:
        processor = processor_cls(capture_dictionary, lsl_streamer)
    else:
        processor = None
    
    # Launch the capture process
    capture_frontend.init_capture()

    # Initialize display if requested
    live_disp_activated = capture_dictionary['display']
    live_disp = LiveDisplay(channel_names=capture_dictionary['signal_labels'], window_len=capture_dictionary['width_display']) if live_disp_activated else Dummy()

    # Initialize recording if requested
    if capture_dictionary['record']:
        recorder = CSVRecorder(capture_dictionary['filename'],
                               raw_signal=True,
                               filtered_signal=True,
                               detection_signal=detector is not None,
                               stimulation_signal=stimulator is not None,
                               detection_activated=False,  # stimulation activated is enough
                               stimulation_activated=True,
                               default_detection_value=0,
                               default_stimulation_value=0)
    else:
        recorder = Dummy()

    # Buffer used for the visualization and the recording
    raw_signal_buffer = []
    filtered_signal_buffer = []
    detection_signal_buffer = []
    stimulation_activated_buffer = []

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

    if PROFILE:
        perf = {"wait msg": [0, 0],
                "no data": [0, 0],
                "got data": [0, 0],
                "filter": [0, 0],
                "lsl": [0, 0],
                "detect": [0, 0],
                "stimulate": [0, 0],
                "csv": [0, 0]}
        t0 = time.perf_counter()

    # Main capture loop
    while True:

        if PROFILE:
            t00 = time.perf_counter()
        
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

        if PROFILE:
            t1 = time.perf_counter()
            perf["wait msg"][0] += t1 - t00
            perf["wait msg"][1] += 1

        # Then, we retrieve the data from the capture process
        raw_points = capture_frontend.get_data()  # np.array (data series x ads_channels), or None
        # If we have no data, we continue to the next iteration
        if raw_points is None:
            if PROFILE:
                t11 = time.perf_counter()
                perf["no data"][0] += t11 - t1
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
            detection_signal = detector.detect(filtered_points)

            if PROFILE:
                t5 = time.perf_counter()
                perf["detect"][0] += t5 - t4
                perf["detect"][1] += 1

            # Stimulate
            if stimulator is not None:
                stimulator_activated = True
                stim = stimulator.stimulate(detection_signal)
                if stim is None:
                    stim = detection_signal
                if capture_dictionary['detect']:
                    detection_signal_buffer += stim

                if PROFILE:
                    t6 = time.perf_counter()
                    perf["stimulate"][0] += t6 - t5
                    perf["stimulate"][1] += 1

                # Send a stimulation every second (uncomment for testing)
                # current_time = time.time()
                # if current_time - last_time >= 1.0:
                #     stimulator.stimulate([True])
                #     last_time = current_time

                # Adds point to buffer for delayed stimulation
                stimulation_delayer.step(filtered_points[0][capture_dictionary['channel_detection'] - 1])

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

        if len(raw_signal_buffer) >= 50:  # TODO: make this an argument
            # live_disp.add_datapoints(raw_signal_buffer)
            if processor is not None:
                live_disp.add_datapoints(filtered_signal_buffer)
            else:
                live_disp.add_datapoints(raw_signal_buffer)
            # recorder.add_recording_data(raw_signal_buffer, detection_signal_buffer, capture_dictionary['detect'], capture_dictionary['stimulate'])

            recorder.append_raw_signal_buffer(raw_signal_buffer)
            recorder.append_filtered_signal_buffer(filtered_signal_buffer)
            recorder.append_detection_signal_buffer(detection_signal_buffer)
            recorder.append_stimulation_activated_buffer(stimulation_activated_buffer)
            recorder.write()

            raw_signal_buffer = []
            filtered_signal_buffer = []
            detection_signal_buffer = []
            stimulation_activated_buffer = []

        if PROFILE:
            t8 = time.perf_counter()
            perf["csv"][0] += t8 - t7
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
            print(f"{k}: {tot} (avg: {avg*1000} ms)")
            tt += tot
        print(f"total measured time: {tt} vs real: {t_end - t0}")

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
