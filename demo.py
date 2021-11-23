from time import sleep
from playsound import playsound

from frontend import Frontend
from leds import LEDs

frontend = Frontend()
leds = LEDs()

playsound('sample.mp3')

try:
    data = frontend.read_reg(0x00, 1)
    assert data == [0x3E], "Wrong output"
    print("EEG Frontend responsive")

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
