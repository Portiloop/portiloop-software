from time import sleep
from playsound import playsound

from frontend import Frontend
from leds import LEDs

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

playsound('sample.mp3')

try:
    data = frontend.read_reg(0x00, 1)
    assert data == [0x3E], "Wrong output"
    print("EEG Frontend responsive")

    print("Configuring EEG Frontend")
    data = frontend.write_reg(0x00, FRONTEND_CONFIG)
    print("EEG Frontend configured")

    leds.aquisition(True)
    sleep(0.5)
    leds.aquisition(False)
    sleep(0.5)
    leds.aquisition(True)

    for i in range(200):
        red = (i % 10) * 10
        blue = ((i % 100) // 10) * 10
        leds.led1(red, 0, blue)
        sleep(0.02)

    for state in [RED, BLUE, PURPLE, CLOSED] * 3:
        leds.led2(state)
        sleep(0.2)

    for state in [RED, CLOSED] * 3:
        leds.led3(state)
        sleep(0.2)

finally:
    frontend.close()
    leds.close()
