import matplotlib.pyplot as plt
import numpy as np
from portiloop.src.detection import SleepSpindleRealTimeDetector
plt.switch_backend('agg')
from portiloop.src.processing import FilterPipeline
from portiloop.src.demo.utils import xdf2array, offline_detect, offline_filter, OfflineSleepSpindleRealTimeStimulator
import gradio as gr


def run_offline(xdf_file, detect_filter_opts, threshold, channel_num, freq):
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

    #  Do Wamsley's method
    if wamsley:
        print("Running Wamsley detection...")
        wamsley_data = offline_detect("Wamsley", \
            data_whole[:, columns.index("offline_filtered_signal")],\
                data_whole[:, columns.index("time_stamps")],\
                    freq)
        wamsley_data = np.expand_dims(wamsley_data, axis=1)
        data_whole = np.concatenate((data_whole, wamsley_data), axis=1)
        columns.append("wamsley_spindles")

    # Do Lacourse's method
    if lacourse:
        print("Running Lacourse detection...")
        lacourse_data = offline_detect("Lacourse", \
            data_whole[:, columns.index("offline_filtered_signal")],\
                data_whole[:, columns.index("time_stamps")],\
                    freq)
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
        stimulator = OfflineSleepSpindleRealTimeStimulator()

    if online_filtering or online_detection:
        print("Running online filtering and detection...")

        points = []
        online_activations = []

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

    print("Saving output...")
    # Output the data to a csv file
    np.savetxt("output.csv", data_whole, delimiter=",", header=",".join(columns), comments="")

    print("Done!")
    return "output.csv"