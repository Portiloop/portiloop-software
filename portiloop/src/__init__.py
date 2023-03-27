import io


def is_coral():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            string = m.read().lower()
            return any([x in string for x in ["phanbell", "coral"]])
    except Exception: pass
    return False


ADS = is_coral()
