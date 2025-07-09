import csv
from pathlib import Path

from portilooplot.jupyter_plot import ProgressPlot


class CSVRecorder:
    def __init__(self,
                 filename,
                 raw_signal=True,
                 filtered_signal=False,
                 detection_signal=False,
                 stimulation_signal=False,
                 detection_activated=False,
                 stimulation_activated=False,
                 default_detection_value=0,
                 default_stimulation_value=0):

        if not (raw_signal or filtered_signal):
            err_str = "At least raw_signal or filtered_signal need to be activated."
            print(err_str)
            raise RuntimeError(err_str)
        self.raw_signal_buffer = [] if raw_signal else None
        self.filtered_signal_buffer = [] if filtered_signal else None
        self.detection_signal_buffer = [] if detection_signal else None
        self.stimulation_signal_buffer = [] if stimulation_signal else None
        self.detection_activated_buffer = [] if detection_activated else None
        self.stimulation_activated_buffer = [] if stimulation_activated else None
        self.default_detection_value = default_detection_value
        self.default_stimulation_value = default_stimulation_value

        # create/open CSV:

        self.filename = filename

        print(f"INFO: Writing data to {self.filename}")

        self.header_written = False
        file_exists = Path(filename).exists()
        if file_exists:
            print(f"INFO: {self.filename} already exists. The writer will append new data.")
            with open(self.filename, 'r') as f:
                if f.readline():
                    self.header_written = True
        self.file = open(self.filename, 'a')
        self.writer = csv.writer(self.file)

        self.writing_buffer = []
        self.max_write = 1

    def write_header(self, nb_channels):
        line = []
        if self.raw_signal_buffer is not None:
            for i in range(nb_channels):
                line.append(f'raw_ch{i+1}')
        if self.filtered_signal_buffer is not None:
            for i in range(nb_channels):
                line.append(f'filtered_ch{i + 1}')
        if self.detection_signal_buffer is not None:
            line.append('detection')
        if self.stimulation_signal_buffer is not None:
            line.append('stimulation')
        if self.detection_activated_buffer is not None:
            line.append('detection_on')
        if self.stimulation_activated_buffer is not None:
            line.append('stimulation_on')
        self.writer.writerows([line])  # write header
        self.header_written = True

    def append_raw_signal_buffer(self, buffer):
        """
        Args:
            buffer: list of lists of floats
        """
        if self.raw_signal_buffer is not None:
            self.raw_signal_buffer += buffer

    def append_filtered_signal_buffer(self, buffer):
        """
        Args:
            buffer: list of lists of floats
        """
        if self.filtered_signal_buffer is not None:
            self.filtered_signal_buffer += buffer

    def append_detection_signal_buffer(self, buffer):
        """
        Args:
            buffer: list of floats
        """
        if self.detection_signal_buffer is not None:
            self.detection_signal_buffer += buffer

    def append_stimulation_signal_buffer(self, buffer):
        """
        Args:
            buffer: list of floats
        """
        if self.stimulation_signal_buffer is not None:
            self.stimulation_signal_buffer += buffer

    def append_detection_activated_buffer(self, buffer):
        """
        Args:
            buffer: list of 0/1
        """
        if self.detection_activated_buffer is not None:
            self.detection_activated_buffer += buffer

    def append_stimulation_activated_buffer(self, buffer):
        """
        Args:
            buffer: list of 0/1
        """
        if self.stimulation_activated_buffer is not None:
            self.stimulation_activated_buffer += buffer

    def __del__(self):
        print(f"Closing")
        # self.file.close()

    def reset_buffers(self):
        if self.raw_signal_buffer is not None:
            self.raw_signal_buffer = []
        if self.filtered_signal_buffer is not None:
            self.filtered_signal_buffer = []
        if self.detection_signal_buffer is not None:
            self.detection_signal_buffer = []
        if self.stimulation_signal_buffer is not None:
            self.stimulation_signal_buffer = []
        if self.detection_activated_buffer is not None:
            self.detection_activated_buffer = []
        if self.stimulation_activated_buffer is not None:
            self.stimulation_activated_buffer = []
        self.writing_buffer = []

    def write(self):

        # compute the number of lines to write:

        if self.raw_signal_buffer is not None:
            len_data = len(self.raw_signal_buffer)
            nb_channels = len(self.raw_signal_buffer[0])
            if self.filtered_signal_buffer is not None and len(self.filtered_signal_buffer) != len_data:
                err_str = f"raw and filtered buffer sizes mismatch: {len_data} != {len(self.filtered_signal_buffer)}"
                print(err_str)
                raise RuntimeError(err_str)
        else:
            len_data = len(self.filtered_signal_buffer)
            nb_channels = len(self.filtered_signal_buffer[0])

        if len_data == 0:
            return

        if not self.header_written:
            self.write_header(nb_channels)

        # pad missing data and check dimensions:

        if self.detection_signal_buffer is not None:
            len_buf = len(self.detection_signal_buffer)
            diff = len_data - len_buf
            if diff != 0:
                self.detection_signal_buffer = [self.default_detection_value] * diff + self.detection_signal_buffer

        if self.stimulation_signal_buffer is not None:
            len_buf = len(self.stimulation_signal_buffer)
            diff = len_data - len_buf
            if diff != 0:
                self.stimulation_signal_buffer = [self.default_stimulation_value] * diff + self.stimulation_signal_buffer

        if self.detection_activated_buffer is not None:
            len_buf = len(self.detection_activated_buffer)
            if len_buf == 0:
                self.detection_activated_buffer = [0] * len_data
            elif len_buf != len_data:
                err_str = f"stimulation activated size mismatch: {len_buf} != {len_data}"
                print(err_str)
                raise RuntimeError(err_str)

        if self.stimulation_activated_buffer is not None:
            len_buf = len(self.stimulation_activated_buffer)
            if len_buf == 0:
                self.stimulation_activated_buffer = [0] * len_data
            elif len_buf != len_data:
                err_str = f"stimulation activated size mismatch: {len_buf} != {len_data}"
                print(err_str)
                raise RuntimeError(err_str)

        # generate lines:

        lines = []
        for idx in range(len_data):
            line = []
            if self.raw_signal_buffer is not None:
                line += self.raw_signal_buffer[idx]  # list of n channels
            if self.filtered_signal_buffer is not None:
                line += self.filtered_signal_buffer[idx]  # list of n channels
            if self.detection_signal_buffer is not None:
                line.append(self.detection_signal_buffer[idx])  # single float
            if self.stimulation_signal_buffer is not None:
                line.append(self.stimulation_signal_buffer[idx])  # single float
            if self.detection_activated_buffer is not None:
                line.append(int(self.detection_activated_buffer[idx]))  # single float (bool)
            if self.stimulation_activated_buffer is not None:
                line.append(int(self.stimulation_activated_buffer[idx]))  # single float (bool)
            lines.append(line)

        self.writing_buffer += lines
        if len(self.writing_buffer) >= self.max_write:
            self.writer.writerows(self.writing_buffer)
            self.reset_buffers()

    # def add_recording_data(self, points, detection_info, detection_on, stim_on):
    #     """
    #     Deprecated
    #     """
    #     stim_label = 2 if stim_on else 1
    #
    #     #detection_info = (np.array(detection_info).astype(int) * stim_label).tolist()
    #     # No need to bother np arrays
    #     detection_info = [stim_label*x for x in detection_info]
    #
    #     # If detection is on but we do not have any points, we add 0s
    #     if detection_on and len(detection_info) == 0:
    #         for point in points:
    #             point.append(0)
    #     # If detection is not on we simply pass
    #     elif not detection_on:
    #         pass
    #     # If detection_info has points
    #     elif len(detection_info) > 0:
    #         # This takes care of the case when detection is turned on by unpausing between two saves
    #         diff_points = len(points) - len(detection_info)
    #
    #         if diff_points != 0:
    #             detection_info = [0.0] * diff_points + detection_info
    #
    #         assert len(points) == len(detection_info)
    #         for idx, point in enumerate(points):
    #             point.append(detection_info[idx])
    #
    #     data = points
    #     self.writing_buffer += data
    #     # write to file
    #
    #     if len(self.writing_buffer) >= self.max_write:
    #         if self.out_format == 'csv':
    #             self.writer.writerows(self.writing_buffer)
    #             # np.savetxt(self.file, np.array(self.writing_buffer), delimiter=',')
    #         elif self.out_format == 'npy':
    #             np.save(self.file, np.array(self.writing_buffer))
    #         self.writing_buffer = []


class LiveDisplay:
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
