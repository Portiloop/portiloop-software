"""
Jupyter GUI for the electrode impedance check.
"""

import logging
import math
import threading

import ipywidgets as widgets
from IPython.display import display

from portiloop.src import ADS
from portiloop.src.core.hardware.impedance import (ImpedanceMeter, SimulatedADS, PORTILOOP_SERIES_RESISTANCE,
                                                   reference_connected)
from portiloop.src.core.utils import get_hardware_version

if ADS:
    from portiloop.src.core.hardware.backend import Backend

logger = logging.getLogger(__name__)


COLORS = {
    'green': ('#43a047', 'white'),
    'yellow': ('#fdd835', 'black'),
    'red': ('#e53935', 'white'),
    'grey': ('#bdbdbd', 'black'),
}
REFERENCE = 'REF'
NOT_MEASURED = 'not measured'


class ImpedanceCheckUI:
    def __init__(self, thresholds_kohm=(10, 25), current=6e-9, duration=1.0, vref=4.5,
                 series_resistance=PORTILOOP_SERIES_RESISTANCE, simulate=None):
        """
        Args:
            thresholds_kohm: (float, float): impedances (kOhm) above which an electrode is yellow, then red
            current: float: lead-off excitation current in A (6e-9 or 24e-9)
            duration: float: measurement window in seconds
            vref: float: ADS reference voltage
            series_resistance: float: resistance in series with each electrode on the PCB (Ohm), subtracted from the result
            simulate: bool: use a simulated ADS1299 (defaults to True when not running on a Portiloop)
        """
        self.simulate = (not ADS) if simulate is None else simulate
        self.vref = vref
        self.series_resistance = series_resistance
        self.nb_channels = self._read_nb_channels()

        self._thread = None
        self._stop_event = threading.Event()
        self._results = {}  # name -> ImpedanceResult
        self._reference = NOT_MEASURED  # True if connected, False if off, None if unknown (see reference_connected)

        self.b_channels = [widgets.ToggleButton(value=True, description=f"ch{i + 1}",
                                                layout=widgets.Layout(width='60px'))
                           for i in range(self.nb_channels)]
        self.b_thresholds = widgets.FloatRangeSlider(
            value=list(thresholds_kohm), min=1, max=100, step=1, description='Yellow (kΩ):',
            style={'description_width': 'initial'}, layout=widgets.Layout(width='450px'))
        self.b_current = widgets.Dropdown(
            options=[('6 nA', 6e-9), ('24 nA', 24e-9)], value=current, description='Current:',
            layout=widgets.Layout(width='180px'))
        self.b_duration = widgets.FloatSlider(
            value=duration, min=0.5, max=4.0, step=0.5, description='Window (s):',
            style={'description_width': 'initial'})
        self.b_start = widgets.ToggleButton(value=False, description='Start', icon='play', button_style='success',
                                            tooltip="Measure continuously until stopped")
        self.b_once = widgets.Button(description='Measure once', icon='bolt', tooltip="Measure all electrodes once")
        self.tiles = {}
        self.tile_box = widgets.HBox(layout=widgets.Layout(flex_flow='row wrap'))
        self.status = widgets.HTML()
        self.legend = widgets.HTML()

        self.b_start.observe(self._on_start, 'value')
        self.b_once.on_click(self._on_once)
        self.b_thresholds.observe(lambda _: self._refresh(), 'value')
        for b in self.b_channels:
            b.observe(lambda _: self._build_tiles(), 'value')

        self._build_tiles()
        self._set_status("Ready." + (" <b>Simulated ADS1299</b> (no Portiloop detected)." if self.simulate else ""))
        display(widgets.VBox([
            widgets.HTML("<h2>Portiloop impedance check</h2>"),
            widgets.HBox([widgets.Label("Electrodes:", layout=widgets.Layout(width='80px'))] + self.b_channels,
                         layout=widgets.Layout(flex_flow='row wrap')),
            widgets.HBox([self.b_thresholds, self.b_current]),
            self.b_duration,
            widgets.HBox([self.b_start, self.b_once]),
            self.tile_box,
            self.legend,
            self.status,
        ]))

    def _read_nb_channels(self):
        if self.simulate:
            return 8
        backend = Backend(get_hardware_version())
        try:
            return backend.get_nb_channels()
        finally:
            backend.close()

    def _open_backend(self):
        if self.simulate:
            return SimulatedADS(nb_channels=self.nb_channels)
        return Backend(get_hardware_version())

    def _selected_channels(self):
        return [i for i, b in enumerate(self.b_channels) if b.value]

    def _color(self, result):
        if result is None or result.railed:
            return 'grey'
        if result.off or math.isnan(result.impedance):
            return 'red'
        low, high = self.b_thresholds.value
        kohm = result.impedance / 1e3
        if kohm < low:
            return 'green'
        if kohm < high:
            return 'yellow'
        return 'red'

    @staticmethod
    def _text(result):
        if result is None:
            return '-'
        if result.off:
            return 'OFF'
        if result.railed or math.isnan(result.impedance):
            return '?'
        return f"{result.impedance / 1e3:.1f} kΩ"

    def _build_tiles(self):
        names = [f"ch{i + 1}" for i in self._selected_channels()] + [REFERENCE]
        self.tiles = {name: widgets.HTML() for name in names}
        self.tile_box.children = list(self.tiles.values())
        self._refresh()

    def _reference_tile(self):
        """
        The impedance of the reference cannot be measured on Portiloop, only its connection.
        """
        if self._reference is True:
            return 'green', 'connected'
        if self._reference is False:
            return 'red', 'OFF'
        if self._reference is None:
            return 'grey', '?'
        return 'grey', '-'

    def _refresh(self):
        for name, tile in self.tiles.items():
            if name == REFERENCE:
                color, text = self._reference_tile()
            else:
                result = self._results.get(name)
                color, text = self._color(result), self._text(result)
            background, foreground = COLORS[color]
            tile.value = (
                f"<div style='background:{background}; color:{foreground}; width:100px; height:70px; margin:4px;"
                f" border-radius:8px; display:flex; flex-direction:column; align-items:center;"
                f" justify-content:center; font-family:sans-serif;'>"
                f"<div style='font-weight:bold; font-size:16px;'>{name}</div>"
                f"<div style='font-size:14px;'>{text}</div></div>")
        low, high = self.b_thresholds.value
        hint = ""
        measured = [r for name, r in self._results.items() if name in self.tiles]
        if self._reference is False:
            hint = "<br><b>The reference electrode is off: reconnect it to measure the electrodes.</b>"
        elif len(measured) > 0 and all(r.off for r in measured):
            hint = "<br><b>All electrodes are off: check the ground (bias) electrode first.</b>"
        self.legend.value = (
            f"<span style='color:{COLORS['green'][0]}'>&#9632;</span> &lt; {low:.0f} kΩ &nbsp; "
            f"<span style='color:{COLORS['yellow'][0]}'>&#9632;</span> {low:.0f}-{high:.0f} kΩ &nbsp; "
            f"<span style='color:{COLORS['red'][0]}'>&#9632;</span> &gt; {high:.0f} kΩ or OFF{hint}")

    def _set_status(self, text):
        self.status.value = f"<i>{text}</i>"

    def _set_running(self, running):
        for b in self.b_channels:
            b.disabled = running
        self.b_current.disabled = running
        self.b_once.disabled = running
        self.b_start.description = 'Stop' if running else 'Start'
        self.b_start.icon = 'stop' if running else 'play'
        self.b_start.button_style = 'danger' if running else 'success'

    def _on_start(self, change):
        if change['new']:
            self._launch(continuous=True)
        else:
            self._stop_event.set()

    def _on_once(self, _):
        self._launch(continuous=False)

    def _launch(self, continuous):
        if self._thread is not None and self._thread.is_alive():
            return
        channels = self._selected_channels()
        if len(channels) == 0:
            self._set_status("Select at least one electrode.")
            self.b_start.value = False
            return
        self._stop_event.clear()
        self._set_running(True)
        self._thread = threading.Thread(target=self._worker, args=(channels, continuous), daemon=True)
        self._thread.start()

    def _worker(self, channels, continuous):
        backend = None
        meter = None
        try:
            backend = self._open_backend()
            meter = ImpedanceMeter(backend, self.nb_channels, current=self.b_current.value, vref=self.vref,
                                   series_resistance=self.series_resistance)
            meter.configure(channels)
            iteration = 0
            while not self._stop_event.is_set():
                iteration += 1
                self._set_status(f"Measuring... (measurement {iteration})")
                electrodes = meter.measure_electrodes(self.b_duration.value, self._stop_event)
                if electrodes is None:
                    break
                self._results.update({f"ch{ch + 1}": r for ch, r in electrodes.items()})
                self._reference = reference_connected(electrodes)
                self._refresh()
                if not continuous:
                    break
            self._set_status("Done. The excitation current is off.")
        except Exception as e:
            logger.exception("Impedance check failed.")
            self._set_status(f"Error: {e}")
        finally:
            try:
                if meter is not None:
                    meter.close()
            finally:
                if backend is not None:
                    backend.close()
                self._set_running(False)
                self.b_start.value = False
