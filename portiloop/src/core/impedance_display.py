"""
HTML rendering of the electrode impedance check results, shared by the Jupyter and Simple GUIs.
"""

import math

from portiloop.src.core.hardware.impedance import reference_connected


COLORS = {
    'green': ('#43a047', 'white'),
    'yellow': ('#fdd835', 'black'),
    'red': ('#e53935', 'white'),
    'grey': ('#bdbdbd', 'black'),
}
REFERENCE = 'REF'
NOT_MEASURED = 'not measured'
DEFAULT_THRESHOLDS_KOHM = (10, 25)


def result_color(result, thresholds_kohm):
    """
    Color of an electrode tile from its ImpedanceResult (None if not measured yet).
    """
    if result is None or result.railed:
        return 'grey'
    if result.off or math.isnan(result.impedance):
        return 'red'
    low, high = thresholds_kohm
    kohm = result.impedance / 1e3
    if kohm < low:
        return 'green'
    if kohm < high:
        return 'yellow'
    return 'red'


def result_text(result):
    if result is None:
        return '-'
    if result.off:
        return 'OFF'
    if result.railed or math.isnan(result.impedance):
        return '?'
    return f"{result.impedance / 1e3:.1f} kΩ"


def reference_color_text(reference):
    """
    The impedance of the reference cannot be measured on Portiloop, only its connection.

    Args:
        reference: result of reference_connected, or NOT_MEASURED
    """
    if reference is True:
        return 'green', 'connected'
    if reference is False:
        return 'red', 'OFF'
    if reference is None:
        return 'grey', '?'
    return 'grey', '-'


def tile_html(name, color, text):
    background, foreground = COLORS[color]
    return (f"<div style='background:{background}; color:{foreground}; width:100px; height:70px; margin:4px;"
            f" border-radius:8px; display:flex; flex-direction:column; align-items:center;"
            f" justify-content:center; font-family:sans-serif;'>"
            f"<div style='font-weight:bold; font-size:16px;'>{name}</div>"
            f"<div style='font-size:14px;'>{text}</div></div>")


def legend_html(results, reference, thresholds_kohm):
    """
    Color legend, with a hint when the reference or all electrodes are off.

    Args:
        results: list of the displayed ImpedanceResult
        reference: result of reference_connected, or NOT_MEASURED
    """
    low, high = thresholds_kohm
    hint = ""
    if reference is False:
        hint = "<br><b>The reference electrode is off: reconnect it to measure the electrodes.</b>"
    elif len(results) > 0 and all(r.off for r in results):
        hint = "<br><b>All electrodes are off: check the ground (bias) electrode first.</b>"
    return (f"<span style='color:{COLORS['green'][0]}'>&#9632;</span> &lt; {low:.0f} kΩ &nbsp; "
            f"<span style='color:{COLORS['yellow'][0]}'>&#9632;</span> {low:.0f}-{high:.0f} kΩ &nbsp; "
            f"<span style='color:{COLORS['red'][0]}'>&#9632;</span> &gt; {high:.0f} kΩ or OFF{hint}")


def results_html(results, thresholds_kohm=DEFAULT_THRESHOLDS_KOHM):
    """
    Tiles and legend for the results of a single ImpedanceMeter.measure_electrodes call.

    Args:
        results: dict {channel (0-indexed): ImpedanceResult}
    """
    reference = reference_connected(results)
    tiles = [tile_html(f"ch{ch + 1}", result_color(r, thresholds_kohm), result_text(r)) for ch, r in sorted(results.items())]
    tiles.append(tile_html(REFERENCE, *reference_color_text(reference)))
    return (f"<div style='display:flex; flex-wrap:wrap;'>{''.join(tiles)}</div>"
            + legend_html(list(results.values()), reference, thresholds_kohm))
