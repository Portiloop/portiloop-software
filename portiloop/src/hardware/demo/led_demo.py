from time import sleep
from playsound import playsound

from frontend import Frontend
from leds import LEDs, Color

frontend = Frontend()
leds = LEDs()

print("Testing audio output")
playsound('sample.mp3')
print("Audio playback ended")

try:
    print("Testing EEG Frontend")
    data = frontend.read_regs(0x00, 1)
    assert data == [0x3E], "Wrong output"
    print("EEG Frontend responsive")

    print("Testing LEDs")
    print("Aquisition LED")
    leds.aquisition(True)
    sleep(0.5)
    leds.aquisition(False)
    sleep(0.5)
    leds.aquisition(True)

    print("USER1 (PWM) LED")
    for i in range(200):
        red = (i % 10) * 10
        blue = ((i % 100) // 10) * 10
        leds.led1(red, 0, blue)
        sleep(0.02)

    print("USER2 (2-color) LED")
    for state in [Color.RED, Color.BLUE, Color.PURPLE, Color.CLOSED] * 3:
        leds.led2(state)
        sleep(0.2)

    print("USER3 LED")
    for state in [Color.RED, Color.CLOSED] * 3:
        leds.led3(state)
        sleep(0.2)

    print("LEDs testing ended")
finally:
    frontend.close()
    leds.close()
