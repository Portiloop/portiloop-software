import io


def is_coral():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            if 'phanbell' in m.read().lower(): return True
    except Exception: pass
    return False


ADS = is_coral()
