from periphery import GPIO, PWM
from enum import Enum

class Color(Enum):
    RED = 0
    BLUE = 1
    PURPLE = 2
    CLOSED = 3

class LEDs:
    def __init__(self):
        self.led1_R = PWM(0, 0)
        self.led1_B = PWM(2, 0)
        self.led1_gnd = PWM(1, 0)

        self.led2_R = GPIO("/dev/gpiochip0", 8, "out")
        self.led2_B = GPIO("/dev/gpiochip0", 6, "out")
        self.led2_gnd = GPIO("/dev/gpiochip0", 7, "out")

        self.led3_R = GPIO("/dev/gpiochip2", 0, "out")
        #self.led3_B = GPIO("/dev/gpiochip2", 5, "out")
        self.led3_gnd = GPIO("/dev/gpiochip2", 8, "out")

        self.acq = GPIO("/dev/gpiochip2", 20, "out")

        # Init LEDs
        # RGBs
        self.led1_R.frequency = 1e3
        self.led1_R.duty_cycle = 0.0
        self.led1_R.enable()

        self.led1_B.frequency = 1e3
        self.led1_B.duty_cycle = 0.0
        self.led1_B.enable()

        self.led1_gnd.frequency = 1e3
        self.led1_gnd.duty_cycle = 0.0
        self.led1_gnd.enable()

        # LED2
        self.led2_R.write(False)
        self.led2_B.write(False)
        self.led2_gnd.write(False)

        # LED3
        self.led3_R.write(False)
        #self.led3_B.write(False)
        self.led3_gnd.write(False)

    def aquisition(self, val: bool):
        self.acq.write(val)

    # red, green & blue are between 0 and 100 inclusively
    def led1(self, red: int, green: int, blue: int):
        assert 0 <= red <= 100, "Red should be between 0 and 100"
        assert 0 <= green <= 100, "Green should be between 0 and 100"
        assert 0 <= blue <= 100, "Blue should be between 0 and 100"
        self.led1_R.duty_cycle = red / 100
        self.led1_B.duty_cycle = blue / 100

    def led2(self, value: Color):
        if value == Color.RED:
            self.led2_R.write(True)
            self.led2_B.write(False)
        elif value == Color.BLUE:
            self.led2_R.write(False)
            self.led2_B.write(True)
        elif value == Color.PURPLE:
            self.led2_R.write(True)
            self.led2_B.write(True)
        elif value == Color.CLOSED:
            self.led2_R.write(False)
            self.led2_B.write(False)
        else:
            assert False, "Unknown color"

    def led3(self, value: Color):
        if value == Color.RED:
            self.led3_R.write(True)
        elif value == Color.CLOSED:
            self.led3_R.write(False)
        else:
            assert False, "Unknown color"

    def close(self):
        # LED1
        self.led1_R.disable()
        self.led1_B.disable()
        self.led1_gnd.disable()
        self.led1_R.close()
        self.led1_B.close()
        self.led1_gnd.close()

        # LED2
        self.led2_R.write(False)
        self.led2_B.write(False)
        self.led2_gnd.write(False)
        self.led2_R.close()
        self.led2_B.close()
        self.led2_gnd.close()

        # LED3
        self.led3_R.write(False)
        #self.led3_B.write(False)
        self.led3_gnd.write(False)
        self.led3_R.close()
        #self.led3_B.close()
        self.led3_gnd.close()

        # AQUISITION
        self.acq.write(False)
        self.acq.close()

