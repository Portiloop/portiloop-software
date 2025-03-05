import numpy as np
import pyxdf


STREAM_NAMES = {
    'filtered_data': 'Portiloop Filtered',
    'raw_data': 'Portiloop Raw Data',
    'stimuli': 'Portiloop_stimuli'
}


def read_xdf_file(xdf_file, channel):
    """
    Read a single xdf file and return the data of the given channel as a dataframe.
    """
    xdf_data, _ = pyxdf.load_xdf(xdf_file)

    # Load all streams given their names
    filtered_stream, raw_stream, markers = None, None, None
    for stream in xdf_data:
        # print(stream['info']['name'])
        if stream['info']['name'][0] == STREAM_NAMES['filtered_data']:
            filtered_stream = stream
        elif stream['info']['name'][0] == STREAM_NAMES['raw_data']:
            raw_stream = stream
        elif stream['info']['name'][0] == STREAM_NAMES['stimuli']:
            markers = stream

    if filtered_stream is None or raw_stream is None:
        raise ValueError("One of the necessary streams could not be found. Make sure that at least one signal stream is present in XDF recording")

    # Add all samples from raw and filtered signals
    points = []
    diffs = []
    shortest_stream = min(int(filtered_stream['footer']['info']['sample_count'][0]),
                          int(raw_stream['footer']['info']['sample_count'][0]))
    for i in range(shortest_stream):
        if markers is not None:
            datapoint = [filtered_stream['time_stamps'][i], 
                        float(filtered_stream['time_series'][i, channel-1]), 
                        raw_stream['time_series'][i, channel-1], 
                        0]
        else:
            datapoint = [filtered_stream['time_stamps'][i], 
                        float(filtered_stream['time_series'][i, channel-1]), 
                        raw_stream['time_series'][i, channel-1]]
        diffs.append(abs(filtered_stream['time_stamps'][i] - raw_stream['time_stamps'][i]))
        points.append(datapoint)

    # Add markers
    columns = ["Time Stamps", "Filtered Signal", "Raw Signal"]
    if markers is not None:
        columns.append("Stimuli")
        for time_stamp in markers['time_stamps']:
            new_index = np.abs(filtered_stream['time_stamps'] - time_stamp).argmin()
            points[new_index][3] = 1
    
    # Create dataframe
    array = np.array(points)

    return array, columns


if __name__ == "__main__": 
    pass