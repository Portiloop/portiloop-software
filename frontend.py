from periphery import GPIO
from spidev import SpiDev

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

class Frontend:
    def __init__(self):
        self.nrst = GPIO("/dev/gpiochip2", 9, "out")
        self.pwdn = GPIO("/dev/gpiochip2", 12, "out")
        self.start = GPIO("/dev/gpiochip3", 29, "out")
        self.dev = SpiDev()
        self.dev.open(0, 0)
        self.dev.max_speed_hz = 8000000
        self.dev.mode = 0b01

        self.start.write(False)
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

    def read(self, len):
        values = self.dev.xfer([RDATA] + [0x00] * len)
        return values[1:]

    def start_continuous(self):
        self.dev.xfer([RDATAC])

    def stop_continuous(self):
        self.dev.xfer([SDATAC])

    def read_continuous(self, len):
        values = self.dev.xfer([0x00] * len)

    def close(self):
        self.dev.close()
