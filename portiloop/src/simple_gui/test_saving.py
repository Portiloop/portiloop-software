# import csv
# import numpy as np
# from portiloop.src.utils import EDFRecorder
# from tqdm import tqdm

from portiloop.src.stimulation import SleepSpindleRealTimeStimulator, AlternatingStimulator


stimulator = AlternatingStimulator()

try: 
    while True:
        # print("stimulating...")
        stimulator.stimulate(None)
except KeyboardInterrupt:
    pass



