import csv
import os
import numpy as np
from portilooplot.jupyter_plot import ProgressPlot

class CSVRecorder:
    def __init__(self, filename):
        self.writing_buffer = []
        self.max_write = 1
        self.filename = filename
        
        file_exists = os.path.exists(self.filename)
        file_empty = file_exists and os.path.getsize(self.filename) == 0
        
        self.file = open(self.filename, 'a')
        self.writer = csv.writer(self.file)

        if not file_exists or file_empty:
            self.writer.writerow(['Fpz', 'Cz', 'Fz', 'Pz', 'e5', 'e6', 'stim'])
        
        print(f"Saving file to {self.filename}")
        self.out_format = 'csv' # 'npy'

    def __del__(self):
        print(f"Closing")
        # self.file.close()
        if hasattr(self, 'file') and not self.file.closed:
            self.file.close()

    def add_recording_data(self, points, detection_info, detection_on, stim_on):
        stim_label = 2 if stim_on else 1
        
        #detection_info = (np.array(detection_info).astype(int) * stim_label).tolist()
        # No need to bother np arrays
        detection_info = [stim_label*x for x in detection_info]

        # If detection is on but we do not have any points, we add 0s
        if detection_on and len(detection_info) == 0:
            for point in points:
                point.append(0)
        # If detection is not on we simply pass
        elif not detection_on:
            pass
        # If detection_info has points
        elif len(detection_info) > 0:
            # This takes care of the case when detection is turned on by unpausing between two saves
            diff_points = len(points) - len(detection_info)

            if diff_points != 0:
                detection_info = [0.0] * diff_points + detection_info

            assert len(points) == len(detection_info)
            for idx, point in enumerate(points):
                point.append(detection_info[idx])
            
        data = points
        self.writing_buffer += data
        # write to file

        if len(self.writing_buffer) >= self.max_write:
            if self.out_format == 'csv':
                self.writer.writerows(self.writing_buffer)
                #np.savetxt(self.file, np.array(self.writing_buffer), delimiter=',')
            elif self.out_format == 'npy':
                np.save(self.file, np.array(self.writing_buffer))
            self.writing_buffer = []



class LiveDisplay():
    def __init__(self, channel_names, window_len=100):
        self.history = []
        self.pp = ProgressPlot(plot_names=channel_names,
                               max_window_len=window_len)
        self.matplotlib = False

    def add_datapoints(self, datapoints):
        """
        Adds 8 lists of datapoints to the plot

        Args:
            datapoints: list of 8 lists of floats (or list of 8 floats)
        """
        if self.matplotlib:
            import matplotlib.pyplot as plt
        disp_list = []
        for datapoint in datapoints:
            d = [[elt] for elt in datapoint]
            disp_list.append(d)

            if self.matplotlib:
                self.history += d[1]

        if not self.matplotlib:
            # print(disp_list)
            # print(datapoints)
            self.pp.update_with_datapoints(disp_list)
        elif len(self.history) == 1000:
            plt.plot(self.history)
            plt.show()
            self.history = []

    def add_datapoint(self, datapoint):
        disp_list = [[elt] for elt in datapoint]
        self.pp.update(disp_list)


class LSLStreamer:
    def __init__(self, streams, channel_count, frequency, id):
        from pylsl import StreamInfo, StreamOutlet

        self.streams = streams

        lsl_info_raw = StreamInfo(name='Portiloop Raw Data',
                                  type='eeg',
                                  channel_count=channel_count,
                                  nominal_srate=frequency,
                                  channel_format='float32',
                                  source_id=id)
        self.lsl_outlet_raw = StreamOutlet(lsl_info_raw)

        if streams['filtered']:
            lsl_info = StreamInfo(name='Portiloop Filtered',
                                  type='eeg',
                                  channel_count=channel_count,
                                  nominal_srate=frequency,
                                  channel_format='float32',
                                  source_id=id)
            self.lsl_outlet = StreamOutlet(lsl_info)

        if streams['markers']:
            lsl_markers_info = StreamInfo(name='Portiloop_stimuli',
                                          type='Markers',
                                          channel_count=1,
                                          channel_format='string',
                                          source_id=id)
            self.lsl_outlet_markers = StreamOutlet(lsl_markers_info)

    def push_filtered(self, data):
        self.lsl_outlet.push_sample(data)

    def push_raw(self, data):
        self.lsl_outlet_raw.push_sample(data)

    def push_marker(self, text):
        self.lsl_outlet_markers.push_sample([text])

    def __del__(self):
        print("Closing LSL streams")
        self.lsl_outlet_raw.__del__()
        if self.streams['filtered']:
            self.lsl_outlet.__del__()
        if self.streams['markers']:
            self.lsl_outlet_markers.__del__()

    @staticmethod
    def string_for_detection_activation(pause):
        return "DETECT_OFF" if pause else "DETECT_ON"

def get_temperature_celsius(value_microvolt):
    return (value_microvolt - 145300) / 490 + 25
