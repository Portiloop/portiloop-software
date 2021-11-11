from periphery import GPIO, PWM, SPI, I2C
from spidev import SpiDev
from time import sleep

led1_R = PWM(0, 0)
led1_B = PWM(2, 0)
led1_gnd = PWM(1, 0)

led2_R = GPIO("/dev/gpiochip0", 8, "out")
led2_B = GPIO("/dev/gpiochip0", 6, "out")
led2_gnd = GPIO("/dev/gpiochip0", 7, "out")

led3_R = GPIO("/dev/gpiochip2", 0, "out")
#led3_B = GPIO("/dev/gpiochip2", 5, "out")
led3_gnd = GPIO("/dev/gpiochip2", 8, "out")

aquisition = GPIO("/dev/gpiochip2", 20, "out")

frontend_nrst = GPIO("/dev/gpiochip2", 9, "out")
frontend_npwdn = GPIO("/dev/gpiochip2", 12, "out")
frontend_start = GPIO("/dev/gpiochip3", 29, "out")
frontend_ = SpiDev()
frontend_.open(0, 0)
frontend_.max_speed_hz = 8000000
frontend_.mode = 0b01

audio = I2C("/dev/i2c-1")
AUDIO_ADDR = 0x68

aquisition.write(True)
sleep(0.5)
aquisition.write(False)
sleep(0.5)
aquisition.write(True)

try:
  frontend_npwdn.write(True)
  frontend_start.write(False)
  sleep(0.1)
  frontend_nrst.write(False)
  sleep(0.01)
  frontend_nrst.write(True)
  sleep(0.1)
  frontend_.xfer([0x11])

  data_out = [0x20, 0x00, 0x00]
  data_in = frontend_.xfer(list(data_out))

  assert data_in[2] == 0x3E, "Wrong output"
  print("EEG Frontend responsive")
finally:
  frontend_.close()

try:
  for reg in [0x0000, 0x0002, 0x0004, 0x0006, 0x000A, 0x000E, 0x0010, 0x0014, 0x0020, 0x0022, 0x0024, 0x0026, 0x0028, 0x002A, 0x002C, 0x002E, 0x0030, 0x0032, 0x0034, 0x0036, 0x0038, 0x003A, 0x003C, 0x0100, 0x0102, 0x0104, 0x0106, 0x0108, 0x010A, 0x010C, 0x010E, 0x0110, 0x0116]:
    msgs = [I2C.Message([reg >> 8, reg & 0xFF]), I2C.Message([0x00, 0x00], read=True)]
    audio.transfer(AUDIO_ADDR, msgs)
    print("0x{:04x}: 0x{:02x} 0x{:02x}".format(reg, msgs[1].data[0], msgs[1].data[1]))
finally:
  audio.close()

state = 0
try:
  led1_R.frequency = 1e3
  led1_R.enable()
  led1_B.frequency = 1e3
  led1_B.enable()
  led1_gnd.frequency = 1e3
  led1_gnd.enable()
  led1_gnd.duty_cycle = 0.0
  for i in range(200):
    led1_R.duty_cycle = (state % 10) / 10
    led1_B.duty_cycle = (state // 10) / 10
    state = (state + 1) % 100
    sleep(0.02)
finally:
  led1_R.duty_cycle = 0.0
  led1_R.close()
  led1_B.duty_cycle = 0.0
  led1_B.close()
  led1_gnd.duty_cycle = 0.0
  led1_gnd.close()

state = 0
try:
  led2_gnd.write(False)
  for i in range(12):
    led2_R.write(state % 2 == 1)
    led2_B.write(state // 2 == 1)
    state = (state + 1) % 4
    sleep(0.2)
finally:
  led2_R.write(False)
  led2_R.close()
  led2_B.write(False)
  led2_B.close()
  led2_gnd.write(False)
  led2_gnd.close()

state = 0
try:
  led3_gnd.write(False)
  for i in range(12):
    state = 3
    led3_R.write(state % 2 == 1)
    #led3_B.write(state // 2 == 1)
    state = (state + 1) % 4
    sleep(0.2)
finally:
  led3_R.write(False)
  led3_R.close()
  #led3_B.write(False)
  #led3_B.close()
  led3_gnd.write(False)
  led3_gnd.close()
