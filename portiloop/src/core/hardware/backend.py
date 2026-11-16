import logging
import os
import struct
from time import sleep
from portiloop.src import ADS
if ADS:
    from periphery import GPIO
    from spidev import SpiDev

logger = logging.getLogger(__name__)

# DRDY edge events of the GPIO character device. python-periphery uses the v2 API when the kernel supports it
# (python-periphery >= 2.5, e.g. on the RDK X5), otherwise the v1 API (python-periphery 2.4 on Mendel):
# struct gpioevent_data (v1): __u64 timestamp, __u32 id, padded to 16 bytes
_DRDY_EVENT_V1 = struct.Struct("=QI4x")
# struct gpio_v2_line_event (v2): __u64 timestamp_ns, __u32 id, offset, seqno, line_seqno, __u32 padding[6]
_DRDY_EVENT_V2 = struct.Struct("=QIIII24x")
_DRDY_QUEUE = 16  # events kept by the kernel for a line, older ones are dropped (drivers/gpio/gpiolib.c)


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
    def __init__(self, values, nb_channels):
        assert values[0] & 0xF0 == 0xC0, "Invalid readback"
        self.loff_statp = (values[0] & 0x0F) << 4 | (values[1] & 0xF0) >> 4
        self.loff_statn = (values[1] & 0x0F) << 4 | (values[2] & 0xF0) >> 4
        self.gpios = values[2] & 0x0F

        self._channels = [
            (values[3 + i * 3] << 16) | (values[4 + i * 3] << 8) | values[5 + i * 3]
            for i in range(nb_channels)
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


class Backend:
    def __init__(self, portiloop_version):

        if portiloop_version < 2:
            logger.warning(f"Your version ({portiloop_version}) of the Portiloop PCB has known issues and is not supported. Upgrade to Version >= 2.3.")

        # Register commands need 4 tCLK (about 2 us) between bytes to be decoded: they stay at 1 MHz (they fail at
        # 10 MHz). The data reads (RDATA + 27 bytes) have no such constraint and can run faster where tested.
        max_speed = 1000000
        self.data_speed = max_speed

        self.start_gpio = None
        self.drdy = None
        spi_bus, spi_cs = 0, 0
        if portiloop_version == 1:
            # self.nrst = GPIO("/dev/gpiochip2", 9, "out")
            # self.pwdn = GPIO("/dev/gpiochip2", 12, "out")
            self.drdy = GPIO("/dev/gpiochip3", 28, "in")

        elif portiloop_version == 2:
            self.drdy = GPIO("/dev/gpiochip0", 45, "in")
            # GPIO37 (header pin 35) is SW1's other branch to the ADS1299 START pin.
            # In the supported (non-stackable) switch position it's disconnected from START,
            # but we still hold it low defensively in case that branch is closed instead.
            self.start_gpio = GPIO("/dev/gpiochip0", 37, "out")
            self.start_gpio.write(False)

        elif portiloop_version == 3:
            # D-Robotics RDK X5: header pin 24 (CE0) is wired to SoC SPI1_CSN1 (/dev/spidev1.1),
            # DRDY on pin 37, START on pin 35
            spi_bus, spi_cs = 1, 1
            # 10 MHz data reads: about 0.1 ms per sample (Backend.read()) instead of about 0.33 ms at 1 MHz
            self.data_speed = 10000000
            self.drdy = GPIO("/dev/gpiochip4", 22, "in")
            self.start_gpio = GPIO("/dev/gpiochip3", 11, "out")
            self.start_gpio.write(False)

        # size of the DRDY events read in wait_new_data(): the v2 events are 48 bytes, and counting them as 16-byte v1
        # events would report 2 missed conversions for each one (each sample recorded 3 times)
        self._drdy_event = _DRDY_EVENT_V1
        if self.drdy is not None:
            self.drdy.edge = "falling"
            if type(self.drdy).__name__ == "Cdev2GPIO":
                self._drdy_event = _DRDY_EVENT_V2
        self.dev = SpiDev()
        self.dev.open(spi_bus, spi_cs)
        self.dev.max_speed_hz = max_speed
        self.dev.mode = 0b01
        self._nb_channels = None  # read once, at the first reading

        self.stop_continuous()

    def get_nb_channels(self):
        config_bits = self.read_regs(0x00, 1)
        channel_bits = config_bits[0] & 0x03
        if channel_bits == 0x00:
            return 4
        elif channel_bits == 0x01:
            return 6
        elif channel_bits == 0x02:
            return 8
        else:
            return -1

    # def powerup(self):
    #     self.pwdn.write(True)
    #     sleep(0.1)

    # def powerdown(self):
    #     self.pwdn.write(False)

    def reset(self):
        """
        Resets the ADS1299 with the RESET command: all the registers return to their default values.

        The datasheet requires a reset after power-up (SBAS499C, 11.1 Power-Up Sequencing), and the RESET pin
        is not driven by the Coral on v2 (pulled up on the PCB). Only call it before configuring the ADS: it
        would break the recording of another process using it.
        """
        self.dev.xfer([RESET])
        sleep(0.001)  # 18 tCLK (9 us) to execute the command, without sending any other command
        self.stop_continuous()  # the device wakes up in RDATAC mode

    def read_regs(self, start, len):
        values = self.dev.xfer([RREG | (start & 0x1F), (len - 1) & 0x1F] + [0x00] * len)
        return values[2:]

    def write_regs(self, start, values):
        self.dev.xfer([WREG | (start & 0x1F), (len(values) - 1) & 0x1F] + values)

    def read(self):
        if self._nb_channels is None:
            self._nb_channels = self.get_nb_channels()
        values = self.dev.xfer([RDATA] + [0x00] * 27, self.data_speed)
        return Reading(values[1:], self._nb_channels)

    def start_continuous(self):
        self.dev.xfer([RDATAC])

    def stop_continuous(self):
        self.dev.xfer([SDATAC])

    def read_continuous(self):
        values = self.dev.xfer([0x00] * 27, self.data_speed)
        return Reading(values, self.get_nb_channels())

    def is_ready(self):
        if self.drdy is None:
            return True
        return not self.drdy.read()

    def wait_new_data(self):
        """
        Waits for the next conversion and reads it with RDATA.

        Returns:
            (Reading, int): the latest conversion, and the number of conversions missed since the previous call
                (DRDY events still pending when the caller is late: RDATA only returns the latest conversion)
        """
        if self.drdy is None:
            raise RuntimeError("DRDY GPIO is not configured on this hardware version; use python_clock=True")
        # blocks until a DRDY falling edge is queued, then returns all the queued events at once
        size = self._drdy_event.size
        events = os.read(self.drdy._line_fd, size * _DRDY_QUEUE)
        missed = len(events) // size - 1
        return self.read(), missed  # read SPI with RDATA

    def start_conversion(self):
        if self.start_gpio is not None:
            self.start_gpio.write(False)  # ensure START pin is low, in case SW1's GPIO37 branch is closed
        self.dev.xfer([START])
    
    def stop_conversion(self):
        self.dev.xfer([STOP])

    def close(self):
        if self.drdy is not None:
            self.drdy.close()
        if self.start_gpio is not None:
            self.start_gpio.close()
        self.dev.close()
