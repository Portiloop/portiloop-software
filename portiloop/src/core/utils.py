import io


def get_portiloop_version():
    # Check if we are on a Portiloop V1 or V2.
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            string = m.read().lower()
            if "phanbell" in string:
                version = 1
            elif "coral" in string:
                version = 2
            else:
                version = -1
    except Exception:
        version = -1
    return version


class DummyAlsaMixer:
    def __init__(self):
        self.volume = 50

    def getvolume(self):
        return [self.volume]

    def setvolume(self, volume):
        self.volume = volume


class Dummy:
    def __getattr__(self, attr):
        return lambda *args, **kwargs: None


def get_temperature_celsius(value_microvolt):
    return (value_microvolt - 145300) / 490 + 25
