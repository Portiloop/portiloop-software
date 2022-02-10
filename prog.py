from time import sleep
from playsound import playsound
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

from frontend import Frontend
from leds import LEDs, Color

DEFAULT_FRONTEND_CONFIG = [
    0x3E, # ID (RO)
    0x96, # Datarate = 250 SPS
    0xC0, # No tests
    0x60, # Power-down reference buffer, no bias
    0x00, # No lead-off
    0x61, # Channel 1 active, 24 gain, no SRB2 & input shorted
    0x61, # Channel 2 active, 24 gain, no SRB2 & input shorted
    0x61, # Channel 3 active, 24 gain, no SRB2 & input shorted
    0x61, # Channel 4 active, 24 gain, no SRB2 & input shorted
    0x61, # Channel 5 active, 24 gain, no SRB2 & input shorted
    0x61, # Channel 6 active, 24 gain, no SRB2 & input shorted
    0x61, # Channel 7 active, 24 gain, no SRB2 & input shorted
    0x61, # Channel 8 active, 24 gain, no SRB2 & input shorted
    0x00, # No bias
    0x00, # No bias
    0x00, # No lead-off
    0x00, # No lead-off
    0x00, # No lead-off flip
    0x00, # Lead-off positive status (RO)
    0x00, # Laed-off negative status (RO)
    0x0F, # All GPIOs as inputs
    0x00, # Disable SRB1
    0x00, # Unused
    0x00, # Single-shot, lead-off comparator disabled
]

FRONTEND_CONFIG = [
    0x3E, # ID (RO)
    0x95, # Datarate = 500 SPS
    0xC0, # No tests
    0xE0, # Power-down reference buffer, no bias
    0x00, # No lead-off
    0x68, # Channel 1 active, 24 gain, no SRB2 & normal input
    0x68, # Channel 2 active, 24 gain, no SRB2 & normal input
    0x68, # Channel 3 active, 24 gain, no SRB2 & normal input
    0x68, # Channel 4 active, 24 gain, no SRB2 & normal input
    0x68, # Channel 5 active, 24 gain, no SRB2 & normal input
    0x68, # Channel 6 active, 24 gain, no SRB2 & normal input
    0x68, # Channel 7 active, 24 gain, no SRB2 & normal input
    0xE0, # Channel 8 disabled, 24 gain, no SRB2 & normal input
    0x00, # No bias
    0x00, # No bias
    0xFF, # Lead-off on all positive pins?
    0xFF, # Lead-off on all negative pins?
    0x00, # Normal lead-off
    0x00, # Lead-off positive status (RO)
    0x00, # Lead-off negative status (RO)
    0x00, # All GPIOs as output ?
    0x20, # Disable SRB1
]

frontend = Frontend()
leds = LEDs()

try:
    data = frontend.read_regs(0x00, 1)
    assert data == [0x3E], "Wrong output"
    print("EEG Frontend responsive")
    leds.led2(Color.BLUE)

    print("Configuring EEG Frontend")
    frontend.write_regs(0x00, FRONTEND_CONFIG)
    #config = DEFAULT_FRONTEND_CONFIG[:]
    #config[0x02] = 0xD0 # Activate test signals
    #config[0x03] = 0xE0 # Power-up reference buffer
    #for i in range(0x05, 0x0D):
    #    config[i] = 0x05 # Channel active, 1 gain, no SRB2 & Test signal
    #frontend.write_regs(0x00, config)
    data = frontend.read_regs(0x00, len(FRONTEND_CONFIG))
    assert data == FRONTEND_CONFIG, f"Wrong config: {data} vs {FRONTEND_CONFIG}"
    frontend.start()
    print("EEG Frontend configured")
    leds.led2(Color.PURPLE)
    while not frontend.is_ready():
        pass
    print("Ready for data")

    leds.aquisition(True)
    sleep(0.5)
    leds.aquisition(False)
    sleep(0.5)
    leds.aquisition(True)

    points = []
    START = datetime.now()
    NUM_STEPS = 2000
    #times = [timedelta(milliseconds=i) for i in range(NUM_STEPS)]
    times = [i / 250 for i in range(NUM_STEPS)]
    for x in range(NUM_STEPS):
        while not frontend.is_ready():
            pass
        values = frontend.read()
        print(values.channels())
        points.append(values.channels())
        while frontend.is_ready():
            pass
    leds.aquisition(False)

    points = np.transpose(np.array(points))
    fig, ax = plt.subplots()
    for i in [0]:#range(8):
        ax.plot(times, points[i] * 4.5 / 2**24, label='channel #' + str(i))
    ax.set_xlabel('Time since start (s)')
    ax.set_ylabel('Value')
    ax.set_title('Test readings')
    ax.legend()
    plt.savefig('readings.png')

finally:
    frontend.close()
    leds.close()
