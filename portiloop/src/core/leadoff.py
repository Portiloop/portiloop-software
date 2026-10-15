"""
Lead-off detection during a recording, without injecting current in the connected electrodes.

On Portiloop, a fallen electrode leaves its input floating. As every recorded channel is part of the bias loop
(BIAS_SENSP/N), a floating input drives the bias amplifier and degrades all the other channels. The LeadoffMonitor
removes fallen electrodes from the bias loop, from the capture process:

1. At the start of the recording, a DC lead-off check of all the channels (the bias amplifier sensing no channel
   in the meantime) removes the channels without electrode from the bias loop, before any data is recorded.
2. Screening: every second, the last seconds of each channel are checked for the signature of a floating input
   (strong mains interference, or saturation). Movement artifacts are mostly below 10 Hz and do not trigger it.
3. Confirmation: when a channel has been suspect for a few seconds, a short DC lead-off check (24 nA) is run on the
   suspect channels only, which are out of the bias loop during the check. The current returns through the bias
   electrode as a common mode, so the other channels are not affected.
4. Confirmed channels are removed from the bias loop for the rest of the recording. Channels that pass the check
   but pick up extreme mains interference are removed too: their electrode still has a DC path to the body (e.g.,
   through hair or a fingertip), but megohms of contact impedance feed the interference to the bias loop as well.

When all the channels in use look floating at once, the reference (or bias) electrode is probably off: nothing is
disabled, and no check is run for a while. The reference is off when the channels look floating while their
electrodes pass the check, the bias is off when all the electrodes (at least 2) fail it at once.
"""

import time

import numpy as np

from portiloop.src.core.hardware.config_hardware import ADS_GAIN
from portiloop.src.core.hardware.impedance import REG_LOFF, REG_LOFF_SENSP, LEADOFF_CURRENTS

REG_BIAS_SENSP = 0x0D  # followed by BIAS_SENSN
REG_CONFIG4 = 0x17
CONFIG4_LOFF_COMP = 0x02  # lead-off comparators enabled
LOFF_DC_24NA = LEADOFF_CURRENTS[24e-9] << 2  # comparator threshold 95 %, 24 nA, DC lead-off

# An open input is a 4.7 nF filter capacitor: from the negative rail, 24 nA (+/- 20 %) needs up to ~1.2 s to charge it
# past the comparator threshold
INITIAL_CHECK_DURATION = 2.0  # s
SCREEN_WINDOW = 2.0  # s of signal screened every second
MAINS_FREQUENCIES = (50.0, 60.0)
MAINS_SUSPECT_UV = 800.0  # peak mains amplitude above which a channel looks floating (connected: < 300 uV, fallen: > 5 mV)
MAINS_DISABLE_UV = 5000.0  # peak mains amplitude above which a channel is disabled even if it passes the check
RAILED_SUSPECT_FRACTION = 0.2  # fraction of saturated samples above which a channel looks floating
RAIL_FRACTION = 0.95  # fraction of the full scale above which a sample is saturated
SUSPECT_SECONDS = 3  # consecutive suspect screenings before a confirmation check
CHECK_DURATION = 1.5  # s of DC lead-off check, the status is read at the end
NOT_CONFIRMED_COOLDOWN = 60.0  # s before a channel that passed a check can be checked again
REFERENCE_COOLDOWN = 600.0  # s without check after a suspected reference or bias fault
RECOVERY_SECONDS = 10  # consecutive screenings without suspect channel to end a reference or bias fault

# Reasons of the DISABLED events:
REASON_OPEN = "open"  # failed the DC lead-off check
REASON_MAINS = "mains"  # passed the check, but extreme mains interference

# Event types:
NOT_CONNECTED_AT_START = "not_connected_at_start"
DISABLED = "disabled"
SUSPECT_NOT_CONFIRMED = "suspect_not_confirmed"
REFERENCE_OR_BIAS_OFF = "reference_or_bias_off"
REFERENCE_OR_BIAS_RECOVERED = "reference_or_bias_recovered"


def _channels(mask, nb_channels):
    return [ch for ch in range(nb_channels) if mask >> ch & 0x01]


class LeadoffMonitor:
    def __init__(self, backend, frequency, in_use, nb_channels, vref=4.5, clock=time.monotonic):
        """
        Args:
            backend: Backend: ADS1299 backend, conversions running
            frequency: int: number of samples passed to update per second
            in_use: int: bitmask of the channels in the bias loop (BIAS_SENSP of the recording configuration)
            nb_channels: int: number of channels of the ADS
            vref: float: ADS reference voltage
            clock: function returning a time in seconds (for the cooldowns)
        """
        self.backend = backend
        self.frequency = int(frequency)
        self.in_use = in_use
        self.nb_channels = nb_channels
        self.uv_per_code = vref / ADS_GAIN / 2 ** 23 * 1e6
        self.full_scale_code = RAIL_FRACTION * 2 ** 23
        self.clock = clock

        self.window = int(SCREEN_WINDOW * self.frequency)
        self.buffer = np.zeros((self.window, nb_channels))
        self.n = 0  # samples received
        self.suspect_counts = [0] * nb_channels
        self.cooldown_until = [0.0] * nb_channels
        self.reference_fault = False
        self.reference_cooldown_until = 0.0
        self.clean_count = 0  # consecutive screenings without suspect channel
        self.check = None  # ongoing confirmation check
        self.last_screening = None  # {channel: (mains amplitude in uV, saturated fraction)}, for diagnostics

    def _event(self, event, channels=(), reason=None):
        event = {"event": event, "channels": [ch + 1 for ch in channels], "sample": self.n, "time": time.time()}
        if reason is not None:
            event["reason"] = reason
        return event

    def _set_bias(self, mask):
        self.backend.write_regs(REG_BIAS_SENSP, [mask, mask])

    def _set_dc_leadoff(self, mask):
        if mask:
            self.backend.write_regs(REG_LOFF, [LOFF_DC_24NA])
            self.backend.write_regs(REG_LOFF_SENSP, [mask, 0x00])
        else:
            self.backend.write_regs(REG_LOFF_SENSP, [0x00, 0x00])
            self.backend.write_regs(REG_LOFF, [0x00])

    def initial_check(self):
        """
        DC lead-off check of all the channels in use. Must be called with the conversions running, before recording.

        Returns:
            list of events
        """
        mask = self.in_use
        if mask == 0:
            return []
        self.backend.write_regs(REG_CONFIG4, [CONFIG4_LOFF_COMP])
        self._set_bias(0x00)  # floating inputs would drive the bias amplifier and fail the connected ones
        self._set_dc_leadoff(mask)
        time.sleep(INITIAL_CHECK_DURATION)
        statp = self.backend.read().loff_statp & mask
        self._set_dc_leadoff(0x00)
        events = []
        if statp == mask:  # nothing connected: reference or bias off (or no electrode at all)
            events.append(self._event(REFERENCE_OR_BIAS_OFF, _channels(mask, self.nb_channels)))
        elif statp:
            self.in_use = mask & ~statp
            events.append(self._event(NOT_CONNECTED_AT_START, _channels(statp, self.nb_channels)))
        self._set_bias(self.in_use)
        drdy = getattr(self.backend, 'drdy', None)
        while drdy is not None and drdy.poll(0):  # DRDY events queued during the check
            drdy.read_event()
        return events

    def update(self, reading):
        """
        To be called for each recorded sample.

        Args:
            reading: Reading: ADS1299 reading (channel codes and lead-off status)
        Returns:
            list of events
        """
        self.buffer[self.n % self.window] = reading.channels()[:self.nb_channels]
        self.n += 1
        if self.check is not None:
            return self._step_check(reading)
        if self.n % self.frequency == 0 and self.n >= self.window:
            return self._screen()
        return []

    def _screen(self):
        codes = np.roll(self.buffer, -(self.n % self.window), axis=0)  # chronological order
        codes = np.where(codes >= 2 ** 23, codes - 2 ** 24, codes)  # two's complement
        saturated = np.mean(np.abs(codes) >= self.full_scale_code, axis=0)
        x = (codes - codes.mean(axis=0)) * self.uv_per_code
        w = np.hanning(self.window)
        t = np.arange(self.window) / self.frequency
        mains = np.max([np.abs((w * np.exp(-2j * np.pi * f * t)) @ x) * 2 / w.sum() for f in MAINS_FREQUENCIES], axis=0)
        self.last_screening = {ch: (float(mains[ch]), float(saturated[ch])) for ch in range(self.nb_channels)}

        in_use = _channels(self.in_use, self.nb_channels)
        for ch in in_use:
            suspect = mains[ch] > MAINS_SUSPECT_UV or saturated[ch] > RAILED_SUSPECT_FRACTION
            self.suspect_counts[ch] = self.suspect_counts[ch] + 1 if suspect else 0
        self.clean_count = self.clean_count + 1 if all(self.suspect_counts[ch] == 0 for ch in in_use) else 0
        now = self.clock()
        if self.reference_fault:
            if self.clean_count >= RECOVERY_SECONDS:
                self.reference_fault = False
                return [self._event(REFERENCE_OR_BIAS_RECOVERED, in_use)]
            if now < self.reference_cooldown_until:
                return []
        candidates = [ch for ch in in_use if self.suspect_counts[ch] >= SUSPECT_SECONDS and now >= self.cooldown_until[ch]]
        if candidates:
            self._start_check(sum(1 << ch for ch in candidates))
        return []

    def _start_check(self, mask):
        self._set_bias(self.in_use & ~mask)
        self._set_dc_leadoff(mask)
        n = max(2, int(CHECK_DURATION * self.frequency))
        self.check = {"mask": mask, "left": n, "votes": [0] * self.nb_channels, "n_votes": 0, "vote_from": max(1, n // 10)}

    def _step_check(self, reading):
        check = self.check
        check["left"] -= 1
        if check["left"] < check["vote_from"]:  # the end of the check, once open inputs have charged
            check["n_votes"] += 1
            for ch in _channels(check["mask"], self.nb_channels):
                check["votes"][ch] += reading.loff_statp >> ch & 0x01
        if check["left"] > 0:
            return []
        self._set_dc_leadoff(0x00)
        self.check = None
        mask = check["mask"]
        checked = _channels(mask, self.nb_channels)
        confirmed = [ch for ch in checked if check["votes"][ch] * 2 > check["n_votes"]]
        for ch in checked:
            self.suspect_counts[ch] = 0
        in_use = _channels(self.in_use, self.nb_channels)
        all_off = len(in_use) > 1 and set(confirmed) == set(in_use)
        if all_off or (len(confirmed) == 0 and set(checked) == set(in_use)):
            # All the channels in use are off at once (bias off), or all look floating while connected (reference off).
            # A single channel in use that is off is disabled: whether its electrode or the bias fell, nothing else
            # depends on it
            self._set_bias(self.in_use)
            self.reference_fault = True
            self.reference_cooldown_until = self.clock() + REFERENCE_COOLDOWN
            return [self._event(REFERENCE_OR_BIAS_OFF, in_use)]
        events = []
        extreme = [ch for ch in checked if ch not in confirmed and self.last_screening[ch][0] > MAINS_DISABLE_UV]
        for reason, channels in ((REASON_OPEN, confirmed), (REASON_MAINS, extreme)):
            if channels:
                self.in_use &= ~sum(1 << ch for ch in channels)
                events.append(self._event(DISABLED, channels, reason))
        not_confirmed = [ch for ch in checked if ch not in confirmed and ch not in extreme]
        if not_confirmed:
            for ch in not_confirmed:
                self.cooldown_until[ch] = self.clock() + NOT_CONFIRMED_COOLDOWN
            events.append(self._event(SUSPECT_NOT_CONFIRMED, not_confirmed))
        self._set_bias(self.in_use)
        return events
