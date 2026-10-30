"""
Electrode impedance measurement with the ADS1299 AC lead-off excitation.

The ADS1299 injects a small square-wave current (6 nA or 24 nA) at
fCLK / 2**16 = 31.25 Hz into the positive input (INxP) of every selected channel.
The current flows through the electrode into the scalp and back through the bias
(ground) electrode. The reference electrode (SRB1) carries no current, so each
channel only sees I * Z of its own electrode. At 250 SPS the excitation is exactly
fDR / 8 (both derive from the same clock), i.e. one period spans exactly 8 samples.

Electrodes are first checked with the DC lead-off comparators of the ADS1299 (on Portiloop, an open input does
not necessarily saturate under the AC excitation). Only the connected ones are then excited: on Portiloop, the
current injected in an open input flows through its 4.7 nF filter capacitor into SRB1 and out through the
reference electrode, which would add a multiple of the reference impedance to every other channel.

The bias amplifier does not sense any channel during the measurement: it simply holds the body at mid-supply
through the bias electrode, so that floating (disconnected) inputs cannot drive it to a rail.

On Portiloop (v2.3), the impedance of the reference electrode cannot be measured: it only reaches the
ADS1299 through SRB1, which has no lead-off current source (all INxN pins are unconnected). Its connection
is inferred instead: when the reference is unplugged, SRB1 floats and the channels whose electrode passes the
DC lead-off check saturate.

Each electrode is in series with a 5 kOhm resistor on the PCB (R12 to R19), which is subtracted from the
measurements. The 4.7 nF filter capacitors between INxP and SRB1 shunt the excitation by less than 1 %
below a few hundred kOhm, and are ignored.
"""

import logging
import math
import time
from dataclasses import dataclass

import numpy as np

from portiloop.src.core.hardware.config_hardware import BACKEND_CONFIG, ADS_GAIN, mod_config
from portiloop.src.core.capture_backend import bin_to_microvolt

logger = logging.getLogger(__name__)


IMPEDANCE_DATARATE = 250  # SPS
EXCITATION_PERIOD = 8  # samples per period of the 31.25 Hz excitation at 250 SPS
EXCITATION_FREQUENCY = IMPEDANCE_DATARATE / EXCITATION_PERIOD  # 31.25 Hz
ADS_MODULATOR_FREQUENCY = 1.024e6  # fMOD = fCLK / 2 with the internal 2.048 MHz clock

# ILEAD_OFF[1:0] bits of the LOFF register for each excitation current (A):
LEADOFF_CURRENTS = {6e-9: 0b00, 24e-9: 0b01}
FLEAD_OFF_31HZ = 0b10  # AC lead-off at fCLK / 2**16

# Register addresses:
REG_LOFF = 0x04
REG_LOFF_SENSP = 0x0F
REG_LOFF_SENSN = 0x10

PORTILOOP_SERIES_RESISTANCE = 5e3  # Ohm, R12 to R19 between the electrode connector and INxP (Portiloop v2.3)
# A connected electrode cannot measure less than the series resistance: below this fraction of it
# (allowing for the +/- 20 % accuracy of the lead-off current), the excitation current is not flowing:
MIN_SERIES_FRACTION = 0.7
# Below this impedance, the electrode is within the measurement uncertainty (+/- 20 % of the series resistance):
RESOLUTION = 1e3  # Ohm
BELOW_RESOLUTION = f"<{RESOLUTION / 1e3:.0f}"  # in kOhm, for the metadata

SETTLING_TIME = 0.25  # s, discarded after each (re)start of the conversions
LEADOFF_CHECK_SAMPLES = 25  # samples waited before reading the DC lead-off comparators
LEADOFF_PRECHARGE_SAMPLES = 500  # samples of the first DC lead-off check, see ImpedanceMeter.configure
# The DC lead-off check uses 24 nA whatever the AC excitation current: an open input on Portiloop is a 4.7 nF
# filter capacitor, which the current must charge by ~2.25 V (from mid-supply to the 95 % threshold):
LEADOFF_DC_ILEAD_OFF = LEADOFF_CURRENTS[24e-9]
FILTER_CAPACITANCE = 4.7e-9  # F, between INxP and SRB1 on Portiloop v2.3 (C19 to C26)
RAIL_FRACTION = 0.95  # fraction of the full scale above which a sample is considered railed
MAINS_FREQUENCIES = (50.0, 60.0, 100.0, 120.0)  # Hz, fitted as nuisance terms


@dataclass
class SignalFit:
    excitation: float  # peak amplitude of the 31.25 Hz excitation (uV)
    mains: float  # peak amplitude of the strongest mains fundamental (50 or 60 Hz) (uV)
    residual: float  # standard deviation of what the model does not explain (uV)


@dataclass
class ImpedanceResult:
    impedance: float  # Ohm (nan if unknown)
    off: bool  # True if the electrode is disconnected (DC lead-off comparator)
    railed: bool = False  # True if the electrode is connected but the channel saturates (reference off)

    def below_resolution(self):
        return not (self.off or self.railed or math.isnan(self.impedance)) and self.impedance < RESOLUTION


def impedance_config(channels, nb_channels, current=6e-9):
    """
    ADS1299 register configuration (from address 0x00 to CONFIG4) for impedance measurement.
    Lead-off current sources are left off, they are switched by ImpedanceMeter.

    Args:
        channels: list of int: 0-indexed channels to measure
        nb_channels: int: number of channels of the ADS
        current: float: lead-off excitation current in A (6e-9 or 24e-9)
    """
    if current not in LEADOFF_CURRENTS:
        raise ValueError(f"Unsupported lead-off current: {current}. Possible values: {list(LEADOFF_CURRENTS)}")
    channel_states = ['simple' if i in channels else 'disabled' for i in range(nb_channels)]
    config = mod_config(list(BACKEND_CONFIG), IMPEDANCE_DATARATE, channel_states)
    config[2] = 0xC0  # CONFIG2: no internal test signal
    config[13] = 0x00  # BIAS_SENSP: the bias amplifier holds the body at mid-supply without feedback
    config[14] = 0x00  # BIAS_SENSN
    config[REG_LOFF] = LEADOFF_CURRENTS[current] << 2 | FLEAD_OFF_31HZ
    config[REG_LOFF_SENSP] = 0x00
    config[REG_LOFF_SENSN] = 0x00
    config += [
        0x00,  # MISC2
        0x02,  # CONFIG4: lead-off comparators enabled
    ]
    return config


def sinc3_gain(frequency, datarate):
    """
    Gain of the ADS1299 sinc3 decimation filter at a given frequency.
    """
    n = ADS_MODULATOR_FREQUENCY / datarate
    x = math.pi * frequency / ADS_MODULATOR_FREQUENCY
    return abs(math.sin(n * x) / (n * math.sin(x))) ** 3


def volts_per_ohm(current):
    """
    Peak amplitude (V) of the 31.25 Hz component measured per Ohm of electrode impedance.
    The fundamental of a +/- I square wave has a peak amplitude of 4 / pi * I.
    """
    return current * 4 / math.pi * sinc3_gain(EXCITATION_FREQUENCY, IMPEDANCE_DATARATE)


def fit_signal(indices, signal, datarate=IMPEDANCE_DATARATE):
    """
    Fits the 31.25 Hz excitation and the mains interference in a signal.

    A least-squares fit is used instead of an FFT so that dropped samples (gaps in indices) are handled.
    The model contains a quadratic drift, the harmonics of the 8-sample excitation period, and mains interference.

    Args:
        indices: array of int: sample index of each value (gaps are allowed)
        signal: array of float: signal values
        datarate: float: actual sampling rate (Hz), used for the mains terms
    Returns:
        SignalFit, amplitudes in the unit of signal
    """
    n = np.asarray(indices, dtype=float)
    t = (n - n.mean()) / max(np.ptp(n), 1.0)
    columns = [np.ones_like(n), t, t ** 2]
    for k in (1, 2, 3):
        w = 2 * np.pi * k / EXCITATION_PERIOD
        columns += [np.cos(w * n), np.sin(w * n)]
    columns.append(np.cos(np.pi * n))  # k = 4
    for f in MAINS_FREQUENCIES:
        if f < datarate / 2:
            w = 2 * np.pi * f / datarate
            columns += [np.cos(w * n), np.sin(w * n)]
    design = np.column_stack(columns)
    signal = np.asarray(signal, dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, signal, rcond=None)
    mains = [float(np.hypot(coefficients[i], coefficients[i + 1]))
             for i, f in zip(range(10, len(coefficients), 2), [f for f in MAINS_FREQUENCIES if f < datarate / 2])
             if f in (50.0, 60.0)]
    return SignalFit(excitation=float(np.hypot(coefficients[3], coefficients[4])),
                     mains=max(mains, default=0.0),
                     residual=float(np.std(signal - design @ coefficients)))


class ImpedanceMeter:
    def __init__(self, backend, nb_channels, current=6e-9, vref=4.5, series_resistance=PORTILOOP_SERIES_RESISTANCE):
        """
        Args:
            backend: Backend: opened ADS1299 backend (or SimulatedADS)
            nb_channels: int: number of channels of the ADS
            current: float: lead-off excitation current in A (6e-9 or 24e-9)
            vref: float: ADS reference voltage
            series_resistance: float: resistance in series with each electrode on the PCB (Ohm), subtracted from the result
        """
        self.backend = backend
        self.nb_channels = nb_channels
        self.current = current
        self.vref = vref
        self.series_resistance = series_resistance
        self.full_scale_uv = vref / ADS_GAIN * 1e6
        self.channels = []
        self._statp = 0x00  # LOFF_STATP of the last DC lead-off check
        self.last_fits = {}  # {channel: (SignalFit, railed)} of the last electrode measurement

    def configure(self, channels):
        """
        Writes the impedance configuration for the given 0-indexed channels.

        A first, longer DC lead-off check charges the filter capacitors of the open inputs to the comparator
        threshold (up to ~1.2 s at 24 nA from the negative rail): with the short checks of measure_electrodes only, open inputs
        would pass the check (and be excited) during the first measurements. Open inputs then stay at the rail,
        as they are not excited anymore.
        """
        self.channels = sorted(channels)
        self.backend.reset()
        self.backend.stop_conversion()
        self.backend.write_regs(0x00, impedance_config(self.channels, self.nb_channels, self.current))
        self._statp = self._leads_off(sum(1 << ch for ch in self.channels), LEADOFF_PRECHARGE_SAMPLES)

    def _set_excitation(self, sensp):
        self.backend.stop_conversion()
        self.backend.write_regs(REG_LOFF_SENSP, [sensp, 0x00])  # LOFF_SENSP, LOFF_SENSN

    def _wait_samples(self, n_samples):
        drdy = self.backend.drdy
        while drdy.poll(0):  # drop stale events
            drdy.read_event()
        self.backend.start_conversion()
        for _ in range(n_samples):
            if not drdy.poll(1.0):
                raise TimeoutError("No DRDY signal from the ADS1299.")
            drdy.read_event()
        reading = self.backend.read()
        self.backend.stop_conversion()
        return reading

    def _leads_off(self, sensp, n_samples=LEADOFF_CHECK_SAMPLES):
        """
        Checks the connection of the given positive inputs with the DC lead-off comparators.

        Returns:
            int: LOFF_STATP (bit set = input off)
        """
        loff_ac = impedance_config([], self.nb_channels, self.current)[REG_LOFF]
        self.backend.stop_conversion()
        self.backend.write_regs(REG_LOFF, [(loff_ac & ~0x0F) | LEADOFF_DC_ILEAD_OFF << 2])  # DC lead-off
        self._set_excitation(sensp)
        try:
            reading = self._wait_samples(n_samples)
        finally:
            self._set_excitation(0x00)
            self.backend.write_regs(REG_LOFF, [loff_ac])
        return reading.loff_statp & sensp

    def _acquire(self, duration, stop_event=None):
        """
        Starts the conversions and reads samples on DRDY for SETTLING_TIME + duration seconds.
        Samples read while DRDY events were pending are discarded, as RDATA returns the latest conversion.

        Returns:
            indices: array of int (n,): sample index of each sample
            data: array of float (n, nb_channels): samples in microvolts
            datarate: float: sampling rate measured from the DRDY timestamps
        """
        drdy = self.backend.drdy
        while drdy.poll(0):  # drop stale events
            drdy.read_event()
        self.backend.start_conversion()
        n_settle = int(SETTLING_TIME * IMPEDANCE_DATARATE)
        n_total = n_settle + int(duration * IMPEDANCE_DATARATE)
        timestamps, data = [], []
        n_events = 0
        while n_events < n_total:
            if stop_event is not None and stop_event.is_set():
                break
            if not drdy.poll(1.0):
                raise TimeoutError("No DRDY signal from the ADS1299.")
            event = drdy.read_event()
            n_events += 1
            if drdy.poll(0):  # we are late, the data we would read belongs to a later event
                continue
            reading = self.backend.read()
            if n_events > n_settle:
                timestamps.append(event.timestamp * 1e-9)
                data.append(reading.channels())
        self.backend.stop_conversion()
        if len(timestamps) < 4 * EXCITATION_PERIOD:
            return None
        timestamps = np.array(timestamps)
        period = np.median(np.diff(timestamps))
        steps = np.maximum(1, np.round(np.diff(timestamps) / period)).astype(int)
        indices = np.concatenate([[0], np.cumsum(steps)])
        data = bin_to_microvolt(np.array(data), self.vref)
        return indices, data, 1.0 / period

    def _channel_fits(self, acquisition):
        """
        Returns {channel: (SignalFit in uV, railed)} for the configured channels, railed meaning saturated.
        """
        indices, data, datarate = acquisition
        fits = {}
        for ch in self.channels:
            x = data[:, ch]
            railed = bool(np.mean(np.abs(x) >= RAIL_FRACTION * self.full_scale_uv) > 0.05)
            fits[ch] = (fit_signal(indices, x, datarate), railed)
        return fits

    def _to_impedance(self, amplitude_uv):
        """
        Returns nan when the measured impedance is implausibly below the series resistance (no excitation current).
        """
        total = amplitude_uv * 1e-6 / volts_per_ohm(self.current)
        if total < MIN_SERIES_FRACTION * self.series_resistance:
            return math.nan
        return max(0.0, total - self.series_resistance)

    def measure_electrodes(self, duration=1.0, stop_event=None):
        """
        Measures the impedance of the electrode connected to INxP of each configured channel.
        Only the electrodes that pass the DC lead-off check are excited (see the module docstring).

        Returns:
            dict {channel (0-indexed): ImpedanceResult}, or None if interrupted
        """
        sensp = sum(1 << ch for ch in self.channels)
        statp = self._leads_off(sensp)
        if self._statp & ~statp:
            # Inputs that were off now pass the check: this also happens to open inputs when a step on SRB1
            # (e.g., reconnecting the reference) is coupled through the filter capacitors, so check longer
            statp = self._leads_off(sensp, LEADOFF_PRECHARGE_SAMPLES)
        self._statp = statp
        connected = sensp & ~statp
        if connected == 0:
            self.last_fits = {}
            return {ch: ImpedanceResult(math.nan, True) for ch in self.channels}
        self._set_excitation(connected)
        acquisition = self._acquire(duration, stop_event)
        self._set_excitation(0x00)
        if acquisition is None:
            return None
        results = {}
        self.last_fits = self._channel_fits(acquisition)
        for ch, (fit, railed) in self.last_fits.items():
            off = not connected >> ch & 0x01
            valid = not (off or railed)
            results[ch] = ImpedanceResult(self._to_impedance(fit.excitation) if valid else math.nan, off, railed and not off)
        return results

    def close(self):
        """
        Switches the excitation current off and stops the conversions.
        """
        try:
            self._set_excitation(0x00)
            self.backend.write_regs(REG_LOFF, [0x00])
        finally:
            self.backend.stop_conversion()


def reference_connected(results):
    """
    Infers the connection of the reference electrode from the results of ImpedanceMeter.measure_electrodes.

    Returns:
        True if at least one electrode is connected and does not saturate,
        False if all the connected electrodes saturate (SRB1 is floating),
        None if all electrodes are off (the bias or all electrodes may be unplugged)
    """
    connected = [r for r in results.values() if not r.off]
    if len(connected) == 0:
        return None
    return not all(r.railed for r in connected)


NOT_CHECKED = '?'


def results_to_metadata(results, nb_channels):
    """
    JSON-serializable summary of an impedance check, for the recording metadata.

    Args:
        results: dict {channel (0-indexed): ImpedanceResult} from ImpedanceMeter.measure_electrodes,
            or None if no check was performed
        nb_channels: int: number of channels of the ADS
    Returns:
        dict {"ch1": impedance in kOhm (rounded to 0.1) or "<1" or "OFF" or "?", ..., "REF": "connected" or "OFF" or "?"}
    """
    metadata = {}
    for ch in range(nb_channels):
        r = results.get(ch) if results is not None else None
        if r is None or r.railed or (not r.off and math.isnan(r.impedance)):
            metadata[f"ch{ch + 1}"] = NOT_CHECKED
        elif r.off:
            metadata[f"ch{ch + 1}"] = "OFF"
        elif r.below_resolution():
            metadata[f"ch{ch + 1}"] = BELOW_RESOLUTION
        else:
            metadata[f"ch{ch + 1}"] = round(r.impedance / 1e3, 1)
    reference = reference_connected(results) if results else None
    metadata["REF"] = {True: "connected", False: "OFF"}.get(reference, NOT_CHECKED)
    return metadata


class _SimulatedReading:
    def __init__(self, channels, loff_statp=0x00, loff_statn=0x00):
        self._channels = channels
        self.loff_statp = loff_statp
        self.loff_statn = loff_statn

    def channels(self):
        return self._channels


class _SimulatedEvent:
    def __init__(self, timestamp):
        self.timestamp = timestamp


class _SimulatedDRDY:
    def __init__(self, ads):
        self.ads = ads

    def poll(self, timeout=None):
        return self.ads._poll(timeout)

    def read_event(self):
        return self.ads._read_event()


class SimulatedADS:
    """
    Minimal stand-in for Backend, used to try the impedance check without a Portiloop.
    It follows the lead-off registers to synthesize the electrode signals.
    """
    def __init__(self, nb_channels=8, impedances=None, reference_impedance=8e3, realtime=True,
                 noise_uv=2.0, mains_uv=50.0, drop_probability=0.01, seed=None):
        """
        Args:
            impedances: list of float: impedance of each electrode (Ohm), math.inf for a disconnected electrode
            reference_impedance: float: math.inf for a disconnected reference (the connected channels then saturate)
        """
        self.nb_channels = nb_channels
        self.rng = np.random.default_rng(seed)
        if impedances is None:
            impedances = list(self.rng.choice([3e3, 7e3, 15e3, 40e3, math.inf], size=nb_channels))
        self.impedances = impedances
        self.reference_impedance = reference_impedance
        self.realtime = realtime
        self.noise_uv = noise_uv
        self.mains_uv = mains_uv
        self.drop_probability = drop_probability
        self.regs = [0x00] * 0x18
        self.running = False
        self.n = 0
        self.t0 = 0.0
        self.pending = 0
        self.drdy = _SimulatedDRDY(self)
        self.open_voltages = [0.0] * nb_channels  # V from mid-supply on the filter capacitor of each open input

    def get_nb_channels(self):
        return self.nb_channels

    def reset(self):
        self.regs = [0x00] * 0x18
        self.running = False

    def write_regs(self, start, values):
        self.regs[start:start + len(values)] = values

    def start_conversion(self):
        self.running = True
        self.n = 0
        self.pending = 0
        self.t0 = time.time()

    def stop_conversion(self):
        self.running = False

    def close(self):
        self.running = False

    def _poll(self, timeout):
        if not self.running:
            return False
        if self.pending > 0:
            return True
        if self.realtime:
            wait = self.t0 + (self.n + 1) / IMPEDANCE_DATARATE - time.time()
            if timeout is not None and wait > timeout:
                return False
            time.sleep(max(0.0, wait))
        elif timeout is not None and timeout <= 0:
            return False
        self.pending = 2 if self.rng.random() < self.drop_probability else 1
        return True

    def _read_event(self):
        self.pending -= 1
        if self.regs[REG_LOFF] & 0x03 == 0:  # DC lead-off: the current sources charge the open inputs
            current = [6e-9, 24e-9, 6e-6, 24e-6][(self.regs[REG_LOFF] >> 2) & 0x03]
            for ch in range(self.nb_channels):
                if self.regs[REG_LOFF_SENSP] >> ch & 0x01 and math.isinf(self.impedances[ch]):
                    self.open_voltages[ch] += current / FILTER_CAPACITANCE / IMPEDANCE_DATARATE
        for ch in range(self.nb_channels):
            if not math.isinf(self.impedances[ch]):  # a connected electrode holds its input at the body potential
                self.open_voltages[ch] = 0.0
        self.n += 1
        return _SimulatedEvent(int((self.t0 + self.n / IMPEDANCE_DATARATE) * 1e9))

    def read(self):
        n = self.n
        current = [6e-9, 24e-9, 6e-6, 24e-6][(self.regs[REG_LOFF] >> 2) & 0x03]
        # Only the fundamental of the square wave matters: the sinc3 filter removes the aliased harmonics
        excitation = 0.0
        if self.regs[REG_LOFF] & 0x03:
            excitation = volts_per_ohm(current) * math.cos(2 * math.pi * (n - 1.5) / EXCITATION_PERIOD)
        sensp = self.regs[REG_LOFF_SENSP]
        full_scale = 4.5 / ADS_GAIN
        common = self.mains_uv * 1e-6 * math.sin(2 * math.pi * 60 * n / IMPEDANCE_DATARATE)
        reference_off = math.isinf(self.reference_impedance)
        n_open_excited = sum(1 for ch in range(self.nb_channels) if sensp >> ch & 0x01 and math.isinf(self.impedances[ch]))
        common -= n_open_excited * excitation * (self.reference_impedance + PORTILOOP_SERIES_RESISTANCE)
        values = []
        for ch in range(self.nb_channels):
            v = common + self.noise_uv * 1e-6 * self.rng.standard_normal() + 20e-6 * n / IMPEDANCE_DATARATE
            z = self.impedances[ch]
            if math.isinf(z) or reference_off:
                v = full_scale  # open input, or floating SRB1
            elif sensp >> ch & 0x01:
                v += excitation * (z + PORTILOOP_SERIES_RESISTANCE)
            code = int(round(max(-full_scale, min(full_scale * (1 - 2 ** -23), v)) / full_scale * 2 ** 23))
            values.append(code & 0xFFFFFF)
        statp = sum(1 << ch for ch in range(self.nb_channels) if self.open_voltages[ch] >= 2.25)
        return _SimulatedReading(values, statp)
