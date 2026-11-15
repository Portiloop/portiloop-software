from enum import Enum
from portiloop.src import ADS, IS_RDKX5
if ADS:
    from periphery import GPIO


class Color(Enum):
    BLACK = 0
    RED = 1
    GREEN = 2
    BLUE = 3
    YELLOW = 4
    PURPLE = 5
    CYAN = 6
    WHITE = 7


class LEDs:
    def __init__(self):
        if IS_RDKX5:
            # 40-pin header pins: 7 (R), 13 (B), 11 (G), 40 (leds_ctl)
            self.led1_R = GPIO("/dev/gpiochip3", 9, "out")
            self.led1_B = GPIO("/dev/gpiochip4", 0, "out")
            self.led1_G = GPIO("/dev/gpiochip4", 1, "out")
            self.leds_ctl = GPIO("/dev/gpiochip3", 13, "out")
        else:
            self.led1_R = GPIO("/dev/gpiochip0", 22, "out")
            self.led1_B = GPIO("/dev/gpiochip0", 10, "out")
            self.led1_G = GPIO("/dev/gpiochip0", 9, "out")
            self.leds_ctl = GPIO("/dev/gpiochip0", 39, "out")

        # LED1
        self.led1_R.write(False)
        self.led1_B.write(False)
        self.led1_G.write(False)

        self.leds_ctl.write(False)
    
    def activate(self, active=True):
        self.leds_ctl.write(active)

    def led1(self, value: Color):
        if value == Color.RED:
            self.led1_R.write(True)
            self.led1_G.write(False)
            self.led1_B.write(False)
        elif value == Color.GREEN:
            self.led1_R.write(False)
            self.led1_G.write(True)
            self.led1_B.write(False)
        elif value == Color.BLUE:
            self.led1_R.write(False)
            self.led1_G.write(False)
            self.led1_B.write(True)
        elif value == Color.YELLOW:
            self.led1_R.write(True)
            self.led1_G.write(True)
            self.led1_B.write(False)
        elif value == Color.PURPLE:
            self.led1_R.write(True)
            self.led1_G.write(False)
            self.led1_B.write(True)
        elif value == Color.CYAN:
            self.led1_R.write(False)
            self.led1_G.write(True)
            self.led1_B.write(True)
        elif value == Color.WHITE:
            self.led1_R.write(True)
            self.led1_G.write(True)
            self.led1_B.write(True)
        elif value == Color.BLACK:
            self.led1_R.write(False)
            self.led1_G.write(False)
            self.led1_B.write(False)
        else:
            assert False, "Unknown color"

    def close(self):
        # LED1
        self.led1_R.write(False)
        self.led1_B.write(False)
        self.led1_G.write(False)
        self.led1_R.close()
        self.led1_B.close()
        self.led1_G.close()


class DummyLEDs:
    def __init__(self):
        pass

    def led1(self, value: Color):
        pass

    def close(self):
        pass
