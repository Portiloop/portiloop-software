"""
Electrode impedance measurement with the ADS1299 AC lead-off excitation.

The ADS1299 can inject a small square-wave current (6 nA or 24 nA) at
fCLK / 2**16 = 31.25 Hz into the positive (INxP) or negative (INxN) input of
each channel. The current flows through the electrode into the scalp and back
through the bias (ground) electrode. The electrode on the other side of the
differential pair carries no current, so the channel only sees I * Z of the
excited electrode.

Electrodes reported as off by the DC lead-off comparators of the ADS1299 are displayed as such
(on Portiloop, an open input does not necessarily saturate under the AC excitation).

The measurement is done in two phases:
1. the current is injected in INxP of every selected channel at once:
   each channel measures the impedance of its own electrode;
2. the current is injected in INxN of a single channel: on Portiloop, all INxN
   are tied to the reference electrode through SRB1, so every channel measures
   the impedance of the reference electrode.

At 250 SPS the excitation is exactly fDR / 8 (both derive from the same clock),
i.e. one excitation period spans exactly 8 samples.
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

SETTLING_TIME = 0.25  # s, discarded after each (re)start of the conversions
LEADOFF_CHECK_SAMPLES = 25  # samples waited before reading the DC lead-off comparators
RAIL_FRACTION = 0.95  # fraction of the full scale above which a sample is considered railed
MAINS_FREQUENCIES = (50.0, 60.0, 100.0, 120.0)  # Hz, fitted as nuisance terms


@dataclass
class ImpedanceResult:
    impedance: float  # Ohm (nan if unknown)
    off: bool  # True if the electrode is disconnected (DC lead-off comparator or saturated input)


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


def excitation_amplitude(indices, signal, datarate=IMPEDANCE_DATARATE):
    """
    Peak amplitude of the 31.25 Hz excitation in a signal.

    A least-squares fit is used instead of an FFT so that dropped samples (gaps in indices) are handled.
    The model contains a quadratic drift, the harmonics of the 8-sample excitation period, and mains interference.

    Args:
        indices: array of int: sample index of each value (gaps are allowed)
        signal: array of float: signal values
        datarate: float: actual sampling rate (Hz), used for the mains terms
    Returns:
        float: peak amplitude of the 31.25 Hz component, in the unit of signal
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
    coefficients, *_ = np.linalg.lstsq(np.column_stack(columns), np.asarray(signal, dtype=float), rcond=None)
    return float(np.hypot(coefficients[3], coefficients[4]))


class ImpedanceMeter:
    def __init__(self, backend, nb_channels, current=6e-9, vref=4.5, series_resistance=0.0):
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
        self._off_channels = set()  # channels whose INxP electrode was off at the last measurement

    def configure(self, channels):
        """
        Writes the impedance configuration for the given 0-indexed channels.
        """
        self.channels = sorted(channels)
        self.backend.stop_conversion()
        self.backend.write_regs(0x00, impedance_config(self.channels, self.nb_channels, self.current))

    def _set_excitation(self, sensp, sensn):
        self.backend.stop_conversion()
        self.backend.write_regs(REG_LOFF_SENSP, [sensp, sensn])

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

    def _leads_off(self, sensp, sensn):
        """
        Checks the connection of the given inputs with the DC lead-off comparators.

        Returns:
            (int, int): LOFF_STATP and LOFF_STATN (bit set = input off)
        """
        loff_ac = impedance_config([], self.nb_channels, self.current)[REG_LOFF]
        self.backend.stop_conversion()
        self.backend.write_regs(REG_LOFF, [loff_ac & ~0x03])  # same current, DC lead-off
        self._set_excitation(sensp, sensn)
        try:
            reading = self._wait_samples(LEADOFF_CHECK_SAMPLES)
        finally:
            self._set_excitation(0x00, 0x00)
            self.backend.write_regs(REG_LOFF, [loff_ac])
        return reading.loff_statp & sensp, reading.loff_statn & sensn

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

    def _channel_amplitudes(self, acquisition):
        """
        Returns {channel: (amplitude in uV, railed)} for the configured channels, railed meaning saturated.
        """
        indices, data, datarate = acquisition
        amplitudes = {}
        for ch in self.channels:
            x = data[:, ch]
            railed = bool(np.mean(np.abs(x) >= RAIL_FRACTION * self.full_scale_uv) > 0.05)
            amplitudes[ch] = (excitation_amplitude(indices, x, datarate), railed)
        return amplitudes

    def _to_impedance(self, amplitude_uv):
        return max(0.0, amplitude_uv * 1e-6 / volts_per_ohm(self.current) - self.series_resistance)

    def measure_electrodes(self, duration=1.0, stop_event=None):
        """
        Measures the impedance of the electrode connected to INxP of each configured channel.

        Returns:
            dict {channel (0-indexed): ImpedanceResult}, or None if interrupted
        """
        sensp = sum(1 << ch for ch in self.channels)
        statp, _ = self._leads_off(sensp, 0x00)
        self._set_excitation(sensp, 0x00)
        acquisition = self._acquire(duration, stop_event)
        self._set_excitation(0x00, 0x00)
        if acquisition is None:
            return None
        results = {}
        for ch, (amplitude, railed) in self._channel_amplitudes(acquisition).items():
            off = railed or bool(statp >> ch & 0x01)
            results[ch] = ImpedanceResult(math.nan if off else self._to_impedance(amplitude), off)
        self._off_channels = {ch for ch, r in results.items() if r.off}
        return results

    def measure_reference(self, duration=1.0, stop_event=None):
        """
        Measures the impedance of the reference electrode (SRB1, shared by all INxN).
        The excitation is injected through the first configured channel and measured on all of them,
        except those whose own electrode was off at the last call to measure_electrodes.

        Returns:
            ImpedanceResult, or None if interrupted
        """
        sensn = 1 << self.channels[0]
        _, statn = self._leads_off(0x00, sensn)
        if statn:
            return ImpedanceResult(math.nan, True)
        self._set_excitation(0x00, sensn)
        acquisition = self._acquire(duration, stop_event)
        self._set_excitation(0x00, 0x00)
        if acquisition is None:
            return None
        amplitudes = [a for ch, (a, railed) in self._channel_amplitudes(acquisition).items()
                      if not railed and ch not in self._off_channels]
        if len(amplitudes) == 0:  # no electrode to measure the reference against
            return ImpedanceResult(math.nan, False)
        return ImpedanceResult(self._to_impedance(float(np.median(amplitudes))), False)

    def close(self):
        """
        Switches the excitation current off and stops the conversions.
        """
        try:
            self._set_excitation(0x00, 0x00)
            self.backend.write_regs(REG_LOFF, [0x00])
        finally:
            self.backend.stop_conversion()


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

    def get_nb_channels(self):
        return self.nb_channels

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
        self.n += 1
        return _SimulatedEvent(int((self.t0 + self.n / IMPEDANCE_DATARATE) * 1e9))

    def read(self):
        n = self.n
        current = [6e-9, 24e-9, 6e-6, 24e-6][(self.regs[REG_LOFF] >> 2) & 0x03]
        # Only the fundamental of the square wave matters: the sinc3 filter removes the aliased harmonics
        excitation = 0.0
        if self.regs[REG_LOFF] & 0x03:
            excitation = volts_per_ohm(current) * math.cos(2 * math.pi * (n - 1.5) / EXCITATION_PERIOD)
        sensp, sensn = self.regs[REG_LOFF_SENSP], self.regs[REG_LOFF_SENSN]
        full_scale = 4.5 / ADS_GAIN
        common = self.mains_uv * 1e-6 * math.sin(2 * math.pi * 60 * n / IMPEDANCE_DATARATE)
        if sensn:
            common -= excitation * self.reference_impedance
        values = []
        for ch in range(self.nb_channels):
            v = common + self.noise_uv * 1e-6 * self.rng.standard_normal() + 20e-6 * n / IMPEDANCE_DATARATE
            if sensp >> ch & 0x01:
                z = self.impedances[ch]
                v = full_scale if math.isinf(z) else v + excitation * z
            elif math.isinf(self.impedances[ch]):
                v = full_scale
            code = int(round(max(-full_scale, min(full_scale * (1 - 2 ** -23), v)) / full_scale * 2 ** 23))
            values.append(code & 0xFFFFFF)
        statp = sum(1 << ch for ch in range(self.nb_channels) if math.isinf(self.impedances[ch]))
        statn = 0xFF if math.isinf(self.reference_impedance) else 0x00
        return _SimulatedReading(values, statp, statn)
