import numpy as np
import pyxdf
from wonambi.detect.spindle import DetectSpindle, detect_Lacourse2018, detect_Wamsley2012
from scipy.signal import butter, filtfilt, iirnotch, detrend
import time
from portiloop.src.core.stimulation import Stimulator


STREAM_NAMES = {
    'filtered_data': 'Portiloop Filtered',
    'raw_data': 'Portiloop Raw Data',
    'stimuli': 'Portiloop_stimuli'
}


def sleep_stage(data, threshold=150, group_size=2):
    """Sleep stage approximation using a threshold and a group size.
        Returns a numpy array containing all indices in the input data which CAN be used for offline detection. 
        These indices can then be used to reconstruct the signal from the original data.
    """
    # Find all indexes where the signal is above or below the threshold
    above = np.where(data > threshold)
    below = np.where(data < -threshold)
    indices = np.concatenate((above, below), axis=1)[0]

    indices = np.sort(indices)
    # Get all the indices where the difference between two consecutive indices is larger than 100
    groups = np.where(np.diff(indices) <= group_size)[0] + 1
    # Get the important indices
    important_indices = indices[groups]
    # Get all the indices between the important indices
    group_filler = [np.arange(indices[groups[n] - 1] + 1, index) for n, index in enumerate(important_indices)]
    # Create flat array from fillers
    group_filler = np.concatenate(group_filler)
    # Append all group fillers to the indices
    masked_indices = np.sort(np.concatenate((indices, group_filler)))
    unmasked_indices = np.setdiff1d(np.arange(len(data)), masked_indices)

    return unmasked_indices



class OfflineSleepSpindleRealTimeStimulator(Stimulator):
    def __init__(self):
        self.last_detected_ts = time.time()
        self.wait_t = 0.4  # 400 ms
        self.wait_timesteps = int(self.wait_t * 250)
        self.delayer = None
        self.index = 0
    
    def stimulate(self, detection_signal):
        self.index += 1
        stim = False
        for sig in detection_signal:
            # We detect a stimulation
            if sig:
                # Record time of stimulation
                ts = self.index
                
                # Check if time since last stimulation is long enough
                if ts - self.last_detected_ts > self.wait_timesteps:
                    if self.delayer is not None:
                        # If we have a delayer, notify it
                        self.delayer.detected()
                    stim = True

                self.last_detected_ts = ts
        return stim

    def add_delayer(self, delayer):
        self.delayer = delayer
        self.delayer.stimulate = lambda: True


class OfflineSpindleTrainRealTimeStimulator(OfflineSleepSpindleRealTimeStimulator):
    def __init__(self):
        super().__init__()
        self.max_spindle_train_t = 6.0
    
    def stimulate(self, detection_signal):
        self.index += 1
        stim = False
        for sig in detection_signal:
            # We detect a stimulation
            if sig:
                # Record time of stimulation
                ts = self.index
                
                elapsed = ts - self.last_detected_ts
                # Check if time since last stimulation is long enough
                if self.wait_timesteps < elapsed < int(self.max_spindle_train_t * 250):
                    if self.delayer is not None:
                        # If we have a delayer, notify it
                        self.delayer.detected()
                    stim = True

                self.last_detected_ts = ts
        return stim
    
class OfflineIsolatedSpindleRealTimeStimulator(OfflineSpindleTrainRealTimeStimulator):
    def stimulate(self, detection_signal):
        self.index += 1
        stim = False
        for sig in detection_signal:
            # We detect a stimulation
            if sig:
                # Record time of stimulation
                ts = self.index
                
                elapsed = ts - self.last_detected_ts
                # Check if time since last stimulation is long enough
                if int(self.max_spindle_train_t * 250) < elapsed:
                    if self.delayer is not None:
                        # If we have a delayer, notify it
                        self.delayer.detected()
                    stim = True

                self.last_detected_ts = ts
        return stim


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
        csv_list.append(datapoint)

    # Add markers
    columns = ["time_stamps", "online_filtered_signal_portiloop", "raw_signal"]
    if markers is not None:
        columns.append("online_stimulations_portiloop")
        for time_stamp in markers['time_stamps']:
            new_index = np.abs(filtered_stream['time_stamps'] - time_stamp).argmin()
            csv_list[new_index][3] = 1
    
    return np.array(csv_list), columns
    

def offline_detect(method, data, timesteps, freq, mask):
    # Extract only the interesting elements from the mask
    data_masked = data[mask]

    # Get the spindle data from the offline methods
    time = np.arange(0, len(data)) / freq
    time_masked = time[mask] 
    if method == "Lacourse":
        detector = DetectSpindle(method='Lacourse2018')
        spindles, _, _ = detect_Lacourse2018(data_masked, freq, time_masked, detector)
    elif method == "Wamsley":
        detector = DetectSpindle(method='Wamsley2012')
        spindles, _, _ = detect_Wamsley2012(data_masked, freq, time_masked, detector)
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

def compute_output_table(irl_online_stimulations, online_stimulation, lacourse_spindles, wamsley_spindles, time_overlap_s=2.0):


    # Count the number of spindles in this run which overlap with spindles found IRL
    irl_spindles_count = sum(irl_online_stimulations)
    both_online_irl = sum([1 for index, spindle in enumerate(online_stimulation)\
         if spindle == 1 and 1 in irl_online_stimulations[index - int((time_overlap_s / 2) * 250):index + int((time_overlap_s / 2) * 250)]])

    # Count the number of spindles detected by each method
    online_stimulation_count = np.sum(online_stimulation)
    if lacourse_spindles is not None:
        lacourse_spindles_count = sum([1 for index, spindle in enumerate(lacourse_spindles) if spindle == 1 and lacourse_spindles[index - 1] == 0])
        # Count how many spindles were detected by both online and lacourse
        both_online_lacourse = sum([1 for index, spindle in enumerate(online_stimulation) if spindle == 1 and lacourse_spindles[index] == 1])
    
    if wamsley_spindles is not None:
        wamsley_spindles_count = sum([1 for index, spindle in enumerate(wamsley_spindles) if spindle == 1 and wamsley_spindles[index - 1] == 0])
        # Count how many spindles were detected by both online and wamsley
        both_online_wamsley = sum([1 for index, spindle in enumerate(online_stimulation) if spindle == 1 and wamsley_spindles[index] == 1])
    
    # Create markdown table with the results
    table = "| Method | # of Detected spindles | Overlap with Online (in tool) |\n"
    table += "| --- | --- | --- |\n"
    table += f"| Online in Tool | {online_stimulation_count} | {online_stimulation_count} |\n"
    table += f"| Online detection IRL | {irl_spindles_count} | {both_online_irl} |\n"
    if lacourse_spindles is not None:
        table += f"| Lacourse | {lacourse_spindles_count} | {both_online_lacourse} |\n"
    if wamsley_spindles is not None:
        table += f"| Wamsley | {wamsley_spindles_count} | {both_online_wamsley} |\n"
    return table
    