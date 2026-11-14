import io
import os
from pathlib import Path


def is_coral():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            line = m.read().lower()
            if 'phanbell' in line or "coral" in line:
                return True
    except Exception:
        pass
    return False


def is_rdkx5():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            line = m.read().lower()
            if 'rdk x5' in line or 'd-robotics' in line:
                return True
    except Exception:
        pass
    return False


IS_CORAL = is_coral()
IS_RDKX5 = is_rdkx5()
ADS = IS_CORAL or IS_RDKX5

# This line is to start something which seems to be necessary to make sure the sound works properly. Not sure why
_sample_wav = Path(__file__).resolve().parents[1] / 'sounds' / 'sample1.wav'
if ADS and _sample_wav.exists():
    os.system(f'aplay "{_sample_wav}"')
