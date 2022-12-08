import matplotlib.pyplot as plt
import numpy as np
from portiloop.src.detection import SleepSpindleRealTimeDetector
plt.switch_backend('agg')
from portiloop.src.processing import FilterPipeline
from portiloop.src.demo.utils import xdf2array, offline_detect, offline_filter, OfflineSleepSpindleRealTimeStimulator
import gradio as gr


def run_offline(xdf_file, offline_filtering, online_filtering, online_detection, lacourse, wamsley, threshold, channel_num, freq):

    print("Starting offline processing...")
    # Make sure the inputs make sense:
    if not offline_filtering and (lacourse or wamsley):
        raise gr.Error("You can't use the offline detection methods without offline filtering.")

    if not online_filtering and online_detection:
        raise gr.Error("You can't use the online detection without online filtering.")

    freq = int(freq)

    # Read the xdf file to a numpy array
    print("Loading xdf file...")
    yield None, None, "Loading xdf file..."
    data_whole, columns = xdf2array(xdf_file.name, int(channel_num))
    print(data_whole.shape)
    # Do the offline filtering of the data
    print("Filtering offline...")
    yield None, None, "Filtering offline..."
    if offline_filtering:
        offline_filtered_data = offline_filter(data_whole[:, columns.index("raw_signal")], freq)
        # Expand the dimension of the filtered data to match the shape of the other columns
        offline_filtered_data = np.expand_dims(offline_filtered_data, axis=1)
        data_whole = np.concatenate((data_whole, offline_filtered_data), axis=1)
        columns.append("offline_filtered_signal")

    #  Do Wamsley's method
    print("Running Wamsley detection...")
    yield None, None, "Running Wamsley detection..."
    if wamsley:
        wamsley_data = offline_detect("Wamsley", \
            data_whole[:, columns.index("offline_filtered_signal")],\
                data_whole[:, columns.index("time_stamps")],\
                    freq)
        wamsley_data = np.expand_dims(wamsley_data, axis=1)
        data_whole = np.concatenate((data_whole, wamsley_data), axis=1)
        columns.append("wamsley_spindles")

    # Do Lacourse's method
    print("Running Lacourse detection...")
    yield None, None, "Running Lacourse detection..."
    if lacourse:
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

    print("Running online filtering and detection...")
    yield None, None, "Running online filtering and detection..."
    if online_filtering or online_detection:
        # Plotting variables
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
            
            # Function to return a list of all indexes where activations have happened
            def get_activations(activations):
                return [i for i, x in enumerate(activations) if x == 1]

            # Plot the data
            if index % (10 * freq) == 0 and index >= (10 * freq):
                plt.close()
                fig = plt.figure(figsize=(20, 10))
                plt.clf()
                plt.plot(np.linspace(0, 10, num=freq*10), points[-10 * freq:], label="Data")
                # Draw vertical lines for activations
                for index in get_activations(online_activations[-10 * freq:]):
                    plt.axvline(x=index / freq, color='r', label="Portiloop Stimulation")
                # Add axis titles and legend
                plt.legend()
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                yield fig, None, "Running online filtering and detection..."

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
    yield None, "output.csv", "Done!"