import numpy as np
from portiloop.src.custom.custom_detectors import SleepSpindleRealTimeDetector
from portiloop.src.custom.custom_stimulators import UpStateDelayer
from portiloop.src.core.processing import FilterPipeline
from portiloop.src.custom.demo.utils import OfflineIsolatedSpindleRealTimeStimulator, OfflineSpindleTrainRealTimeStimulator, compute_output_table, sleep_stage, xdf2array, offline_detect, offline_filter, OfflineSleepSpindleRealTimeStimulator
import gradio as gr


def run_offline(xdf_file, detect_filter_opts, threshold, channel_num, freq, detect_trains, stimulation_phase="Fast", buffer_time=0.25):
    # Get the options from the checkbox group
    offline_filtering = 0 in detect_filter_opts
    lacourse = 1 in detect_filter_opts
    wamsley = 2 in detect_filter_opts
    online_filtering = 3 in detect_filter_opts
    online_detection = 4 in detect_filter_opts

    # Make sure the inputs make sense:
    if not offline_filtering and (lacourse or wamsley):
        raise gr.Error("You can't use the offline detection methods without offline filtering.")

    if not online_filtering and online_detection:
        raise gr.Error("You can't use the online detection without online filtering.")

    if xdf_file is None:
        raise gr.Error("Please upload a .xdf file.")

    freq = int(freq)

    # Read the xdf file to a numpy array
    print("Loading xdf file...")
    data_whole, columns = xdf2array(xdf_file.name, int(channel_num))

    # Do the offline filtering of the data
    if offline_filtering:
        print("Filtering offline...")
        offline_filtered_data = offline_filter(data_whole[:, columns.index("raw_signal")], freq)
        # Expand the dimension of the filtered data to match the shape of the other columns
        offline_filtered_data = np.expand_dims(offline_filtered_data, axis=1)
        data_whole = np.concatenate((data_whole, offline_filtered_data), axis=1)
        columns.append("offline_filtered_signal")

    # Do the sleep staging approximation
    if wamsley or lacourse:
        print("Sleep staging...")
        mask = sleep_stage(data_whole[:, columns.index("offline_filtered_signal")], threshold=150, group_size=100)

    #  Do Wamsley's method
    if wamsley:
        print("Running Wamsley detection...")
        wamsley_data = offline_detect("Wamsley", \
            data_whole[:, columns.index("offline_filtered_signal")],\
                data_whole[:, columns.index("time_stamps")],\
                    freq, mask)
        wamsley_data = np.expand_dims(wamsley_data, axis=1)
        data_whole = np.concatenate((data_whole, wamsley_data), axis=1)
        columns.append("wamsley_spindles")

    # Do Lacourse's method
    if lacourse:
        print("Running Lacourse detection...")
        lacourse_data = offline_detect("Lacourse", \
            data_whole[:, columns.index("offline_filtered_signal")],\
                data_whole[:, columns.index("time_stamps")],\
                    freq, mask)
        lacourse_data = np.expand_dims(lacourse_data, axis=1)
        data_whole = np.concatenate((data_whole, lacourse_data), axis=1)
        columns.append("lacourse_spindles")

    # Get the data from the raw signal column
    data = data_whole[:, columns.index("raw_signal")]
    
    # Create the online filtering pipeline
    if online_filtering:
        filter = FilterPipeline(nb_channels=1, sampling_rate=freq)

    # Create the detector
    if online_detection:
        detector = SleepSpindleRealTimeDetector(threshold=threshold, channel=1) # always 1 because we have only one channel

        if detect_trains == "All Spindles":
            stimulator = OfflineSleepSpindleRealTimeStimulator()
        elif detect_trains == "Trains":
            stimulator = OfflineSpindleTrainRealTimeStimulator()
        elif detect_trains == "Isolated & First":
            stimulator = OfflineIsolatedSpindleRealTimeStimulator()

        if stimulation_phase != "Fast":
            stimulation_delayer = UpStateDelayer(freq, stimulation_phase == 'Peak', time_to_buffer=buffer_time, stimulate=lambda: None)
            stimulator.add_delayer(stimulation_delayer)
            

    if online_filtering or online_detection:
        print("Running online filtering and detection...")

        points = []
        online_activations = []
        delayed_stims = []

        # Go through the data
        for index, point in enumerate(data):
            # Filter the data
            if online_filtering:
                filtered_point = filter.filter(np.array([point]))
            else:
                filtered_point = point
            filtered_point = filtered_point.tolist()
            points.append(filtered_point[0])

            if online_detection:
                # Detect the spindles
                result = detector.detect([filtered_point])

                if stimulation_phase != "Fast":
                    delayed_stim = stimulation_delayer.step_timesteps(filtered_point[0])
                    if delayed_stim:
                        delayed_stims.append(1)
                    else:
                        delayed_stims.append(0)

                # Stimulate if necessary
                stim = stimulator.stimulate(result)
                if stim:
                    online_activations.append(1)
                else:
                    online_activations.append(0)

    if online_filtering:
        online_filtered = np.array(points)
        online_filtered = np.expand_dims(online_filtered, axis=1)
        data_whole = np.concatenate((data_whole, online_filtered), axis=1)
        columns.append("online_filtered_signal")

    if online_detection:
        online_activations = np.array(online_activations)
        online_activations = np.expand_dims(online_activations, axis=1)
        data_whole = np.concatenate((data_whole, online_activations), axis=1)
        columns.append("online_stimulations")

        if stimulation_phase != "Fast":
            delayed_stims = np.array(delayed_stims)
            delayed_stims = np.expand_dims(delayed_stims, axis=1)
            data_whole = np.concatenate((data_whole, delayed_stims), axis=1)
            columns.append("delayed_stimulations")

    print("Saving output...")
    # Output the data to a csv file
    np.savetxt("output.csv", data_whole, delimiter=",", header=",".join(columns), comments="")

    # Compute the overlap of online stimulations with the 

    output_table = compute_output_table(
        data_whole[:, columns.index("online_stimulations")],
        data_whole[:, columns.index("online_stimulations_portiloop")],
        data_whole[:, columns.index("lacourse_spindles")] if lacourse else None, 
        data_whole[:, columns.index("wamsley_spindles")] if wamsley else None,)

    print("Done!")
    return "output.csv", output_table