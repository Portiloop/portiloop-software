from periphery import GPIO
from spidev import SpiDev
from time import sleep

WAKEUP = 0x02
STANDBY = 0x04
RESET = 0x06
START = 0x08
STOP = 0x0A

RDATAC = 0x10
SDATAC = 0x11
RDATA = 0x12

RREG = 0x20
WREG = 0x40

class Reading:
    def __init__(self, values: [int]):
#         print(values)
        assert values[0] & 0xF0 == 0xC0, "Invalid readback"
        self.loff_statp = (values[0] & 0x0F) << 4 | (values[1] & 0xF0) >> 4
        self.loff_statn = (values[1] & 0x0F) << 4 | (values[2] & 0xF0) >> 4
        self.gpios = values[2] & 0x0F

        self._channels = [
            (values[3 + i * 3] << 16) | (values[4 + i * 3] << 8) | values[5 + i * 3]
            for i in range(8)
        ]

#         print(self.loff_statp, self.loff_statn, self.gpios, self._channels)

    def channels(self):
        return self._channels

    def gpio(self, idx: int):
        assert 0 <= idx <= 3, "Invalid gpio index"
        return (self.gpios >> idx) & 0x01 == 0x01

    def loff_p(self, idx: int):
        assert 0 <= idx <= 7, "Invalid loff index"
        return (self.loff_statp >> idx) & 0x01 == 0x01

    def loff_n(self, idx: int):
        assert 0 <= idx <= 7, "Invalid loff index"
        return (self.loff_statn >> idx) & 0x01 == 0x01

class Frontend:
    def __init__(self):
        self.nrst = GPIO("/dev/gpiochip2", 9, "out")
        self.pwdn = GPIO("/dev/gpiochip2", 12, "out")
        self._start = GPIO("/dev/gpiochip3", 29, "out")
        self.drdy = GPIO("/dev/gpiochip3", 28, "in")
        self.drdy.edge = "falling"
        self.dev = SpiDev()
        self.dev.open(0, 0)
        self.dev.max_speed_hz = 1000000
        self.dev.mode = 0b01

        self._start.write(False)
        self.powerup()
        self.reset()
        self.stop_continuous()

    def powerup(self):
        self.pwdn.write(True)
        sleep(0.1)

    def powerdown(self):
        self.pwdn.write(False)

    def reset(self):
        self.nrst.write(False)
        sleep(0.01)
        self.nrst.write(True)
        sleep(0.1)

    def read_regs(self, start, len):
        values = self.dev.xfer([RREG | (start & 0x1F), (len - 1) & 0x1F] + [0x00] * len)
        return values[2:]

    def write_regs(self, start, values):
        self.dev.xfer([WREG | (start & 0x1F), (len(values) - 1) & 0x1F] + values)

    def read(self):
        values = self.dev.xfer([RDATA] + [0x00] * 27)
        return Reading(values[1:])

    def start_continuous(self):
        self.dev.xfer([RDATAC])

    def stop_continuous(self):
        self.dev.xfer([SDATAC])

    def read_continuous(self):
        values = self.dev.xfer([0x00] * 27)
        return Reading(values)

    def start(self):
        self._start.write(True)
        self.dev.xfer([START])

    def stop(self):
        self._start.write(False)
        self.dev.xfer([STOP])

    def is_ready(self):
        return not self.drdy.read()
    
    def wait_new_data(self):
        self.drdy.poll(timeout=None)  # poll the falling edge event
        self.drdy.read_event()  # consume the event
        return self.read()  # read SPI with RDATA

    def close(self):
        self.dev.close()