import numpy as np
import pyxdf
from wonambi.detect.spindle import DetectSpindle, detect_Lacourse2018, detect_Wamsley2012
from scipy.signal import butter, filtfilt, iirnotch, detrend
import time
from portiloop.src.stimulation import Stimulator


STREAM_NAMES = {
    'filtered_data': 'Portiloop Filtered',
    'raw_data': 'Portiloop Raw Data',
    'stimuli': 'Portiloop_stimuli'
}


class OfflineSleepSpindleRealTimeStimulator(Stimulator):
    def __init__(self):
        self.last_detected_ts = time.time()
        self.wait_t = 0.4  # 400 ms
        self.delayer = None
    
    def stimulate(self, detection_signal):
        stim = False
        for sig in detection_signal:
            # We detect a stimulation
            if sig:
                # Record time of stimulation
                ts = time.time()
                
                # Check if time since last stimulation is long enough
                if ts - self.last_detected_ts > self.wait_t:
                    if self.delayer is not None:
                        # If we have a delayer, notify it
                        self.delayer.detected()
                    stim = True

                self.last_detected_ts = ts
        return stim

    def add_delayer(self, delayer):
        self.delayer = delayer
        self.delayer.stimulate = lambda: True

def xdf2array(xdf_path, channel):
    xdf_data, _ = pyxdf.load_xdf(xdf_path)

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
    csv_list = []
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
        csv_list.append(datapoint)

    # Add markers
    columns = ["time_stamps", "online_filtered_signal_portiloop", "raw_signal"]
    if markers is not None:
        columns.append("online_stimulations_portiloop")
        for time_stamp in markers['time_stamps']:
            new_index = np.abs(filtered_stream['time_stamps'] - time_stamp).argmin()
            csv_list[new_index][3] = 1
    
    return np.array(csv_list), columns
    

def offline_detect(method, data, timesteps, freq):
    # Get the spindle data from the offline methods
    time = np.arange(0, len(data)) / freq
    if method == "Lacourse":
        detector = DetectSpindle(method='Lacourse2018')
        spindles, _, _ = detect_Lacourse2018(data, freq, time, detector)
    elif method == "Wamsley":
        detector = DetectSpindle(method='Wamsley2012')
        spindles, _, _ = detect_Wamsley2012(data, freq, time, detector)
    else:
        raise ValueError("Invalid method")

    # Convert the spindle data to a numpy array
    spindle_result = np.zeros(data.shape)
    for spindle in spindles:
        start = spindle["start"]
        end = spindle["end"]
        # Find index of timestep closest to start and end
        start_index = np.argmin(np.abs(timesteps - start))
        end_index = np.argmin(np.abs(timesteps - end))
        spindle_result[start_index:end_index] = 1
    return spindle_result


def offline_filter(signal, freq):

    # Notch filter
    f0 = 60.0  # Frequency to be removed from signal (Hz)
    Q = 100.0  # Quality factor
    b, a = iirnotch(f0, Q, freq)
    signal = filtfilt(b, a, signal)

    # Bandpass filter
    lowcut = 0.5
    highcut = 40.0
    order = 4
    b, a = butter(order, [lowcut / (freq / 2.0), highcut / (freq / 2.0)], btype='bandpass')
    signal = filtfilt(b, a, signal)

    # Detrend the signal
    signal = detrend(signal)

    return signal
