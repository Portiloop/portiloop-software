import os
import wave
import time
import random
from threading import Lock

from portiloop.src import ADS
if ADS:
    import alsaaudio

# Import the default pipelines:

from portiloop.src.custom.custom_pipelines import PIPELINES

# Import the pipeline components that we want to reuse:

from portiloop.src.custom.custom_processors import SpindleFilter
from portiloop.src.custom.custom_detectors import SleepSpindleRealTimeDetector
from portiloop.src.custom.custom_stimulators import SleepSpindleRealTimeStimulator

# Import the SimpleUI so that we can launch it from this file

from portiloop.src.simple_gui.simple_gui import SimpleUI

# Import the default Portiloop sound folder for reading all sound files that are present there:

from portiloop.src.core.constants import SOUNDS_FOLDER


# In this example, we define a custom stimulator based on SleepSpindleRealTimeStimulator.
# Our custom stimulator will be the same as SleepSpindleRealTimeStimulator,
# except it will play a random sound on stimulation.
# Additionally, the index of the played sound will be logged in the CSV output.
# (Note: CSV logging will revert to boolean when using a delayer)

class MyCustomStimulator(SleepSpindleRealTimeStimulator):
    def __init__(self, config_dict, lsl_streamer=None, csv_recorder=None, sound_files=[]):
        super().__init__(config_dict, lsl_streamer, csv_recorder)

        self.sound_files = sound_files
        if not len(self.sound_files):  # read all files in SOUNDS_FOLDER
            self.sound_files = [SOUNDS_FOLDER / sound for sound in os.listdir(SOUNDS_FOLDER) if sound[-4:] == ".wav"]
        self.nb_sounds = len(self.sound_files)
        
        # initialize PCMs
        self._lock = Lock()
        self._sounds = []
        self._sound_idx = 0  # random sound index
        for sound_file in self.sound_files:
            with wave.open(str(sound_file), 'rb') as f:
                device = 'softvol'
                self.duration = f.getnframes() / float(f.getframerate())
                # 8bit is unsigned in wav files
                if f.getsampwidth() == 1:
                    frmt = alsaaudio.PCM_FORMAT_U8
                # Otherwise we assume signed data, little endian
                elif f.getsampwidth() == 2:
                    frmt = alsaaudio.PCM_FORMAT_S16_LE
                elif f.getsampwidth() == 3:
                    frmt = alsaaudio.PCM_FORMAT_S24_3LE
                elif f.getsampwidth() == 4:
                    frmt = alsaaudio.PCM_FORMAT_S32_LE
                else:
                    raise ValueError('Unsupported format')
                self.periodsize = f.getframerate() // 8

                pcm = alsaaudio.PCM(channels=f.getnchannels(), rate=f.getframerate(), format=frmt, periodsize=self.periodsize, device=device)

                # Store data in list to avoid reopening the file
                wav_list = []
                while True:
                    data = f.readframes(self.periodsize)
                    if data:
                        wav_list.append(data)
                    else:
                        break
                
                with self._lock:
                    self._sounds.append((pcm, wav_list))
    
    def __del__(self):
        with self._lock:
            self._sounds.clear()
        return super().__del__()
    
    def stimulate(self, detection_signal):
        detection_points, filtered_points = detection_signal
        size = len(detection_points)
        assert len(filtered_points) == size

        for i in range(size):
            filtered_point = filtered_points[i]
            detection_point = detection_points[i]
            result = 0
            if detection_point:
                ts = time.time()
                if ts - self.last_detected_ts > self.wait_t:
                    # sample a random sound index
                    result = random.randint(a=1, b=self.nb_sounds)
                    with self._lock:
                        self._sound_idx = result
                    if self.delayer is not None:
                        self.delayer.detected()
                        self.send_stimulation("FAST_STIM", False)
                    else:
                        # No delayer: send actual stimulation immediately
                        # This launches a thread which calls play_sound()
                        self.send_stimulation("STIM", True)
                self.last_detected_ts = ts

            if self.delayer is not None:
                res_del = self.delayer.step(filtered_point[self.config_dict['channel_detection'] - 1])
                self.csv_recorder.append_stimulation_signal_buffer([int(res_del)])
            else:
                self.csv_recorder.append_stimulation_signal_buffer([int(result)])
    
    def play_sound(self):
        with self._lock:
            i = self._sound_idx - 1
            pcm, wav_list = self._sounds[i]
        for data in wav_list:
            pcm.write(data)


# Finally, we add our custom pipeline to the PIPELINES dictionary:
# (Note: you can create a new dictionary instead)

PIPELINES["Sleep spindles random sound"] = {
    "processor": SpindleFilter,
    "detector": SleepSpindleRealTimeDetector,
    "stimulator": MyCustomStimulator,
    "config_modifiers": {}
}


if __name__ == "__main__":
    # Launch the SimpleUI server on port 8082 when executing this script:
    gui = SimpleUI(pipelines=PIPELINES)
    gui.run(port=8082)
