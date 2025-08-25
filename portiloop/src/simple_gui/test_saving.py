import plotly.graph_objs as go
import plotly.subplots
import numpy as np
import time


def plot_real_time_data_with_output():
    n_seconds = 5  # Display n seconds of data
    sample_rate = 250  # 250Hz sample rate
    data_length = n_seconds * sample_rate

    # Initialize data and model output arrays
    data = np.zeros(data_length)
    model_output = np.zeros(data_length)

    # Create initial figure
    fig = plotly.subplots.make_subplots(rows=2, cols=1, subplot_titles=("EEG", "Model Output"))
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Data'), row=1, col=1)
    fig.add_trace(go.Bar(x=["Output"], y=[0], name='Model Output'), row=2, col=1)

    fig.update_layout(
        # title='Real-time Data and Model Output',
        showlegend=False
    )

    # Show figure
    fig.show()

    # Start data streaming
    stream = fig.stream(go.Scatter, name='Data', overwrite=False)
    stream_output = fig.stream(go.Bar, name='Model Output', overwrite=True)

    # Simulate incoming data and model output updates
    for i in range(data_length):
        data[i] = np.random.rand()  # Replace with your incoming data
        model_output[i] = np.random.rand()  # Replace with your model output

        # Update plot with new data
        fig.update_traces(x=np.arange(i + 1) / sample_rate, y=data[:i + 1], selector=dict(name='Data'))
        fig.update_traces(x=["Output"], y=[np.mean(model_output[:i + 1])], selector=dict(name='Model Output'))

        # Plotly streaming update
        stream.write({'x': np.arange(i + 1) / sample_rate, 'y': data[:i + 1]})
        stream_output.write({'x': ["Output"], 'y': [np.mean(model_output[:i + 1])]})

        time.sleep(1 / sample_rate)  # Simulate real-time data


plot_real_time_data_with_output()
