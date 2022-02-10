from abc import abstractmethod
import threading
import time
import logging
import random
import Queue

from frontend import Frontend
from leds import LEDs, Color
from portilooplot.jupyter_plot import ProgressPlot

import ctypes
import numpy as np
import json


class Datapoint:
    '''
    Class to represent a single reading
    '''
    def __init__(self, raw_datapoint, temperature=[], num_channels=8):
        # Initialize necessary data structures
        self.num_channels = num_channels
        self.reading = np.array(num_channels, dtype=float)

        assert len(temperature) <= len(raw_datapoint), "Temperature array length must be lesser or equal to number of channels"
        self.temperature = temperature

        self._filter_datapoint(raw_datapoint, Datapoint.filter_2scomplement)

    def _filter_datapoint(self, raw_datapoint, filter):
        # Filter one datapoint with the given filter
        assert len(raw_datapoint) == self.num_channels, "Datapoint dimensions do not match channel number"
        for idx, point in enumerate(raw_datapoint):
            # If the given index is a temperature, add that filter to get correct reading
            if idx in self.temperature:
                filter = lambda x : Datapoint.filter_temp(filter(x))
            self.reading[idx] = filter(point)

    def get_datapoint(self):
        '''
        Get readings of all channels in numpy array format
        '''
        return self.reading

    def get_channel(self, channel_idx):
        '''
        Reading at the channel specified by channel_idx (0-7)
        Returns a tuple (value(float), temperature(boolean)) --> temperature is True if channel is a temperature
        '''
        assert 0 <= channel_idx < self.num_channels - 1, "Channel index must be in range [0 - channel_num-1]"
        return self.reading[channel_idx], (channel_idx in self.temperature)

    def get_portilooplot(self):
        '''
        Returns the portilooplot ready version of the Datapoint
        '''
        return [[point] for point in self.reading]

    @staticmethod
    def filter_2scomplement(value):
        '''
        Convert from binary two's complement to binary int
        '''
        if (value & (1 << 23)) != 0:
            value = value - (1 << 24)
        return Datapoint.filter_23(value)

    @staticmethod
    def filter_23(value):
        '''
        Convert from binary int to normal int
        '''
        return (value * 4.5) / (2**23 - 1)  # 23 because 1 bit is lost for sign

    @staticmethod
    def filter_temp(value):
        '''
        Convert from voltage reading to temperature reading in Celcius
        '''
        return int((value * 1000000.0 - 145300.0) / 490.0 + 25.0)


class CaptureThread(threading.Thread):
    '''
    Producer thread which reads from the EEG device. Thread does not process the data
    '''

    def __init__(self, q, freq=250, timeout=None, target=None, name=None):
        super(CaptureThread, self).__init__()
        self.timeout = timeout
        self.target = target
        self.name = name
        self.q = q
        self.freq = freq
        self.frontend = Frontend()
        self.leds = LEDs()

    def run(self):
        '''
        Run the data capture continuously or until timeout
        '''
        self.init_checks()
        start_time = time.time()
        prev_ts = time.time()
        ts_len = 1 / self.freq
        
        while True:
            if not self.q.full():
                # Wait for frontend and for minimum time limit
                while not self.frontend.is_ready() and not time.time() - prev_ts >= ts_len:
                    pass
                prev_ts = time.time()
                
                # Read values and add to q
                values = self.frontend.read()
                self.q.put(values)
                
                # Wait until reading is fully ompleted
                while self.frontend.is_ready():
                    pass

            # Check for timeout
            if time.time() - start_time > self.timeout:
                break
        return

    def init_checks(self):
        '''
        Run Initial threads to the registers to make sure we can start reading
        '''
        data = self.frontend.read_regs(0x00, len(FRONTEND_CONFIG))
        assert data == FRONTEND_CONFIG, f"Wrong config: {data} vs {FRONTEND_CONFIG}"
        self.frontend.start()
        print("EEG Frontend configured")
        self.leds.led2(Color.PURPLE)
        while not self.frontend.is_ready():
            pass
        print("Ready for data")


class FilterThread(threading.Thread):
    def __init__(self, q, target=None, name=None, temperature=[], num_channels=8):
        '''
        Consume raw datapoint from the Capture points, filter them, put resulting datapoint objects into all queues in list
        '''
        super(FilterThread, self).__init__()
        self.target = target
        self.name = name

        # Initialize thread safe datastructures for both consuming and producing
        self.raw_q = q
        self.qs = []

        # Initilialize settings for the filtering
        self.temperature = temperature
        self.num_channels = num_channels

        return

    def run(self):
        while True:
            raw_data = None
            # Get an item from CaptureThread
            if not self.raw_q.empty():
                raw_data = self.raw_q.get()
            assert raw_data is not None, "Got a None item from CaptureThread in FilterThread"

            datapoint = Datapoint(raw_data, )

            # Put Item to all ConsumerThreads
            for q in self.qs:
                if not q.full():
                    q.put(item)
        return

    def add_q(self, q):
        '''
        Add a Queue to the list of queues where filtered values get added
        '''
        self.qs.append(q)

    def remove_q(self, q):
        '''
        Remove a queue from the list
        '''
        self.qs.remove(q)

    def update_settings(self, temperature=None, num_channels=None):
        '''
        Update Settings on the go
        '''
        if self.temperatures is not None:
            self.temperature = temperature
        if num_channels is not None:
            self.num_channels = num_channels


class ConsumerThread(threading.Thread):
    def __init__(self, q, target=None, name=None):
        '''
        Implemetns basic consumer logic, needs _consume_item() to be implemented 
        '''
        super(ConsumerThread,self).__init__()
        self.target = target
        self.name = name
        self.q = q

    def run(self):
        try:
            while True:
                item = None
                if not self.q.empty():
                    item = self.q.get()

                assert item is not None, "Got a None value from FilterThread in ConsumerThread"
                self._consume_item(item)
        except Exception:
            self._on_exit()

    def get_id(self):
 
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id
  
    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
              ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')

    @abstractmethod
    def _consume_item(self, item):
        raise NotImplementedError("_consume_item needs to be implemented in Subclass of ConsumerThread")

    @abstractmethod
    def _on_exit(self):
        raise NotImplementedError("_on_exit needs to be implemented in Subclass of ConsumerThread")


class DisplayThread(ConsumerThread):
    def __init__(self, q, max_window_len=100, num_channel=8, target=None, name=None):
        super().__init__(q, target, name)
        self.pp = ProgressPlot(plot_names=[f"channel#{i+1}" for i in range(num_channel)], max_window_len=max_window_len)

    def _consume_item(self, item):
        self.pp.update(item.get_portilooplot())

    def _on_exit(self):
        self.pp.finalize()


class SaveThread(ConsumerThread):
    def __init__(self, q, default_loc='', target=None, name=None):
        super().__init__(q, target, name)
        self.save = []
        self.default_loc = default_loc

    def to_disk(self, destination):
        print('Saving Method is not yet implemented')
        pass

    def _consume_item(self, item):
        self.save.append(item.get_datapoint().to_list())

    def _on_exit(self):
        self.to_disk(self.default_loc)


class Capture:
    def __init__(self, viz=True, record=True):
        self.viz = viz
        self.record = record

        # Initialize data structures for capture and filtering
        raw_q = Queue.Queue()
        self.capture_thread = CaptureThread(raw_q)
        self.filter_thread = FilterThread(raw_q)

        # Declare data structures for viz and record functionality
        self.viz_q = None
        self.record_q = None
        self.viz_thread = None
        self.record_thread = None

        self.capture_thread.start()
        self.filter_thread.start()

        if viz:
            self.start_viz()
        
        if record:
            self.start_record()

    def start_viz(self):
        self.viz_q = Queue.Queue()
        self.viz_thread = DisplayThread(self.viz_q)
        self.filter_thread.add_q(self.viz_q)
        self.viz_thread.start()

    def stop_viz(self):
        self.filter_thread.remove_q(self.viz_q)
        self.viz_q = None
        self.viz_thread.raise_exception()

    def start_record(self):
        self.record_q = Queue.Queue()
        self.record_thread = SaveThread(self.record_q)
        self.filter_thread.add_q(self.record_q)
        self.record_thread.start()

    def stop_record(self):
        self.filter_thread.remove_q(self.viz_q)
        self.viz_q = None
        self.record_thread.raise_exception()

    def save(self, destination=None):
        if destination is not None:
            self.record_thread.save(destination)
        else:
            self.record_thread.save()

  



if __name__ == "__main__":
    # TODO: Argparse this
    pass