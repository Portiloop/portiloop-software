from time import sleep
from playsound import playsound
import numpy as np
import matplotlib as plt
import os

from frontend import Frontend
from leds import LEDs, Color

FRONTEND_CONFIG = [
    0x3E, # Overwrite ID?
    0x95, # Datarate = 500 SPS
    0xC0, # No tests
    0xE1, # Power-down reference buffer, no bias
    0x00, # No lead-off
    0x60, # Channel 1 active, 24 gain, no SRB2 & normal input
    0x60, # Channel 2 active, 24 gain, no SRB2 & normal input
    0x60, # Channel 3 active, 24 gain, no SRB2 & normal input
    0x60, # Channel 4 active, 24 gain, no SRB2 & normal input
    0x60, # Channel 5 active, 24 gain, no SRB2 & normal input
    0x60, # Channel 6 active, 24 gain, no SRB2 & normal input
    0x60, # Channel 7 active, 24 gain, no SRB2 & normal input
    0xE0, # Channel 8 disabled, 24 gain, no SRB2 & normal input
    0x00, # No bias
    0x00, # No bias
    0xFF, # Lead-off on all positive pins?
    0xFF, # Lead-off on all negative pins?
    0x00, # Normal lead-off
    0x00, # Lead-off positive status (RO) ?
    0x01, # Lead-off negative status (RO) ?
    0x00, # All GPIOs as output ???
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
    data = frontend.write_regs(0x00, FRONTEND_CONFIG)
    print("EEG Frontend configured")
    leds.led2(Color.PURPLE)

    leds.aquisition(True)
    sleep(0.5)
    leds.aquisition(False)
    sleep(0.5)
    leds.aquisition(True)

    points = []
    for x in range(2000):
        sleep(0.001)
        values = frontend.read()
        print(values.channels())

finally:
    frontend.close()
    leds.close()
