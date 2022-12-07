import gradio as gr
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from portiloop.src.demo.demo_stimulator import DemoSleepSpindleRealTimeStimulator
from portiloop.src.detection import SleepSpindleRealTimeDetector

from portiloop.src.stimulation import UpStateDelayer
plt.switch_backend('agg')
from portiloop.src.processing import FilterPipeline


def do_treatment(csv_file, filtering, threshold, detect_channel, freq, spindle_freq, spindle_detection_mode, time_to_buffer):

    # Read the csv file to a numpy array
    data_whole = np.loadtxt(csv_file.name, delimiter=',')

    # Get the data from the selected channel
    detect_channel = int(detect_channel)
    freq = int(freq)
    data = data_whole[:, detect_channel - 1]

    # Create the detector and the stimulator
    detector = SleepSpindleRealTimeDetector(threshold=threshold, channel=1) # always 1 because we have only one channel
    stimulator = DemoSleepSpindleRealTimeStimulator()
    if spindle_detection_mode != 'Fast':
        delayer = UpStateDelayer(freq, spindle_freq, spindle_detection_mode == 'Peak', time_to_buffer=time_to_buffer)
        stimulator.add_delayer(delayer)
    
    # Create the filtering pipeline
    if filtering:
        filter = FilterPipeline(nb_channels=1, sampling_rate=freq)

    # Plotting variables
    points = []
    activations = []
    delayed_activations = []

    # Go through the data
    for index, point in enumerate(data):
        # Step the delayer if exists
        if spindle_detection_mode != 'Fast':
            delayed = delayer.step(point)
            if delayed:
                delayed_activations.append(1)
            else:
                delayed_activations.append(0)

        # Filter the data
        if filtering:
            filtered_point = filter.filter(np.array([point]))
        else:
            filtered_point = point
        
        filtered_point = filtered_point.tolist()

        # Detect the spindles
        result = detector.detect([filtered_point])

        # Stimulate if necessary
        stim = stimulator.stimulate(result)
        if stim:
            activations.append(1)
        else:
            activations.append(0)
        
        # Add data to plotting buffer
        points.append(filtered_point[0])

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
            for index in get_activations(activations[-10 * freq:]):
                plt.axvline(x=index / freq, color='r', label="Fast Stimulation")
            if spindle_detection_mode != 'Fast':
                for index in get_activations(delayed_activations[-10 * freq:]):
                    plt.axvline(x=index / freq, color='g', label="Delayed Stimulation")
            # Add axis titles and legend
            plt.legend()
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            yield fig, None

    # Put all points and activations back in numpy arrays
    points = np.array(points)
    activations = np.array(activations)
    delayed_activations = np.array(delayed_activations)
    # Concatenate with the original data
    data_whole = np.concatenate((data_whole, points.reshape(-1, 1), activations.reshape(-1, 1), delayed_activations.reshape(-1, 1)), axis=1)
    # Output the data to a csv file
    np.savetxt('output.csv', data_whole, delimiter=',')

    yield None, "output.csv"
        

    

with gr.Blocks() as demo:
    gr.Markdown("# Portiloop Demo")
    gr.Markdown("This Demo takes as input a csv file containing EEG data and outputs a csv file with the following added: \n * The data filtered by the Portiloop online filter \n * The stimulations made by Portiloop.")
    gr.Markdown("Upload your CSV file and click **Run Inference** to start the processing...")

    # Row containing all inputs:
    with gr.Row():
        # CSV file
        csv_file = gr.UploadButton(label="CSV File", file_count="single")
        # Filtering (Boolean)
        filtering = gr.Checkbox(label="Filtering (On/Off)", value=True)
        # Threshold value
        threshold = gr.Slider(0, 1, value=0.82, step=0.01, label="Threshold", interactive=True)
        # Detection Channel
        detect_column = gr.Dropdown(choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], value="1", label="Detection Column in CSV", interactive=True) 
        # Frequency
        freq = gr.Dropdown(choices=["100", "200", "250", "256", "500", "512", "1000", "1024"], value="250", label="Sampling Frequency (Hz)", interactive=True)
        # Spindle Frequency
        spindle_freq = gr.Slider(10, 16, value=12, step=1, label="Spindle Frequency (Hz)", interactive=True)
        # Spindle Detection Mode
        spindle_detection_mode = gr.Dropdown(choices=["Fast", "Peak", "Valley"], value="Peak", label="Spindle Detection Mode", interactive=True)
        # Time to buffer
        time_to_buffer = gr.Slider(0, 1, value=0.3, step=0.01, label="Time to Buffer (s)", interactive=True)

    # Output plot
    output_plot = gr.Plot()
    # Output file
    output_array = gr.File(label="Output CSV File")

    # Row containing all buttons:
    with gr.Row():
        # Run inference button
        run_inference = gr.Button(value="Run Inference")
        # Reset button
        reset = gr.Button(value="Reset", variant="secondary")
    run_inference.click(fn=do_treatment, inputs=[csv_file, filtering, threshold, detect_column, freq, spindle_freq, spindle_detection_mode, time_to_buffer], outputs=[output_plot, output_array])

demo.queue()
demo.launch(share=True)
