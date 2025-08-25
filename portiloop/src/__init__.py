import io
import os


def is_coral():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            line = m.read().lower()
            if 'phanbell' in line or "coral" in line: return True
    except Exception: pass
    return False


ADS = is_coral()

# This line is to start something which seems to be necessary to make sure the sound works properly. Not sure why
os.system('aplay /home/mendel/portiloop-software/portiloop/sounds/sample1.wav')
