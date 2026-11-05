"""
Lead-off detection during a recording, without injecting current in the connected electrodes.

On Portiloop, a fallen electrode leaves its input floating. As every recorded channel is part of the bias loop
(BIAS_SENSP/N), a floating input drives the bias amplifier and degrades all the other channels. Fallen electrodes are
removed from the bias loop by a LeadoffMonitor in the parent process, which screens the recorded signal and makes the
decisions, and a LeadoffCheck in the capture process, which owns the ADS1299 and runs the checks (so that the
screening does not delay the capture of the samples):

1. At the start of the recording, a DC lead-off check of all the channels (the bias amplifier sensing no channel
   in the meantime) removes the channels without electrode from the bias loop, before any data is recorded.
2. Screening: every 10 seconds, the last seconds of each channel are checked for the signature of a floating input
   (strong mains interference, or saturation). Movement artifacts are mostly below 10 Hz and do not trigger it.
3. Confirmation: when a channel has been suspect for two screenings, a short DC lead-off check (24 nA) is run on the
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

from portiloop.src.core.hardware.config_hardware import ADS_LSB
from portiloop.src.core.hardware.impedance import REG_LOFF, REG_LOFF_SENSP, LEADOFF_CURRENTS

REG_BIAS_SENSP = 0x0D  # followed by BIAS_SENSN
REG_CONFIG4 = 0x17
CONFIG4_LOFF_COMP = 0x02  # lead-off comparators enabled
LOFF_DC_24NA = LEADOFF_CURRENTS[24e-9] << 2  # comparator threshold 95 %, 24 nA, DC lead-off

# An open input is a 4.7 nF filter capacitor: from the negative rail, 24 nA (+/- 20 %) needs up to ~1.2 s to charge it
# past the comparator threshold
INITIAL_CHECK_DURATION = 2.0  # s
SCREEN_INTERVAL = 10.0  # s between screenings (a fallen electrode degrades all the channels until it is removed)
SCREEN_WINDOW = 2.0  # s of signal screened, at the end of each interval
MAINS_FREQUENCIES = (50.0, 60.0)
MAINS_SUSPECT_UV = 800.0  # peak mains amplitude above which a channel looks floating (connected: < 300 uV, fallen: > 5 mV)
MAINS_DISABLE_UV = 5000.0  # peak mains amplitude above which a channel is disabled even if it passes the check
RAILED_SUSPECT_FRACTION = 0.2  # fraction of saturated samples above which a channel looks floating
RAIL_FRACTION = 0.95  # fraction of the full scale above which a sample is saturated
SUSPECT_SCREENINGS = 2  # consecutive suspect screenings before a confirmation check
CHECK_DURATION = 1.5  # s of DC lead-off check, the status is read at the end
NOT_CONFIRMED_COOLDOWN = 60.0  # s before a channel that passed a check can be checked again
REFERENCE_COOLDOWN = 600.0  # s without check after a suspected reference or bias fault
RECOVERY_SCREENINGS = 2  # consecutive screenings without suspect channel to end a reference or bias fault

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


def _event(event, channels, sample, reason=None):
    event = {"event": event, "channels": [ch + 1 for ch in channels], "sample": sample, "time": time.time()}
    if reason is not None:
        event["reason"] = reason
    return event


class LeadoffCheck:
    """
    The part of the lead-off detection that needs the ADS1299, in the capture process: the initial check, and the
    confirmation checks and changes of the bias loop requested by the LeadoffMonitor of the parent process.
    """
    def __init__(self, backend, frequency, in_use, nb_channels):
        """
        Args:
            backend: Backend: ADS1299 backend, conversions running
            frequency: int: number of samples passed to update per second
            in_use: int: bitmask of the channels in the bias loop (BIAS_SENSP of the recording configuration)
            nb_channels: int: number of channels of the ADS
        """
        self.backend = backend
        self.frequency = int(frequency)
        self.in_use = in_use
        self.nb_channels = nb_channels
        self.check = None  # ongoing confirmation check

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
            events.append(_event(REFERENCE_OR_BIAS_OFF, _channels(mask, self.nb_channels), 0))
        elif statp:
            self.in_use = mask & ~statp
            events.append(_event(NOT_CONNECTED_AT_START, _channels(statp, self.nb_channels), 0))
        self._set_bias(self.in_use)
        drdy = getattr(self.backend, 'drdy', None)
        while drdy is not None and drdy.poll(0):  # DRDY events queued during the check
            drdy.read_event()
        return events

    def set_in_use(self, mask):
        """
        Sets the channels of the bias loop ("LOFF_IN_USE" message of the LeadoffMonitor, after each check).
        """
        self.in_use = mask
        self._set_bias(mask)

    def start_check(self, mask):
        """
        Starts a confirmation check of the given channels ("LOFF_CHECK" message of the LeadoffMonitor).
        """
        self._set_bias(self.in_use & ~mask)
        self._set_dc_leadoff(mask)
        n = max(2, int(CHECK_DURATION * self.frequency))
        self.check = {"mask": mask, "left": n, "votes": [0] * self.nb_channels, "n_votes": 0, "vote_from": max(1, n // 10)}

    def update(self, reading):
        """
        To be called for each recorded sample.

        Args:
            reading: Reading: ADS1299 reading (channel codes and lead-off status)
        Returns:
            dict or None: at the end of a confirmation check, its result (for LeadoffMonitor.check_done)
        """
        check = self.check
        if check is None:
            return None
        check["left"] -= 1
        if check["left"] < check["vote_from"]:  # the end of the check, once open inputs have charged
            check["n_votes"] += 1
            for ch in _channels(check["mask"], self.nb_channels):
                check["votes"][ch] += reading.loff_statp >> ch & 0x01
        if check["left"] > 0:
            return None
        self._set_dc_leadoff(0x00)
        self.check = None  # the checked channels stay out of the bias loop until the decision of the LeadoffMonitor
        return {"mask": check["mask"], "votes": check["votes"], "n_votes": check["n_votes"]}


class LeadoffMonitor:
    """
    The screening and the decisions of the lead-off detection, in the parent process: the confirmation checks and the
    changes of the bias loop are sent to the LeadoffCheck of the capture process.
    """
    def __init__(self, frequency, nb_channels, send, vref=4.5, clock=time.monotonic):
        """
        Args:
            frequency: int: number of samples passed to update per second
            nb_channels: int: number of channels of the ADS
            send: function sending a message to the capture process, ("LOFF_CHECK", mask) or ("LOFF_IN_USE", mask)
            vref: float: ADS reference voltage
            clock: function returning a time in seconds (for the cooldowns)
        """
        self.send = send
        self.frequency = int(frequency)
        self.nb_channels = nb_channels
        self.in_use = 0x00  # channels in the bias loop, sent by the capture process after the initial check
        self.full_scale_uv = RAIL_FRACTION * 2 ** 23 * 1e6 * vref * ADS_LSB  # same scale as bin_to_microvolt
        self.clock = clock

        self.window = int(SCREEN_WINDOW * self.frequency)
        self.interval = int(SCREEN_INTERVAL * self.frequency)
        self.buffer = np.zeros((self.window, nb_channels))
        # Hann-windowed mains kernels, scaled to peak amplitudes (computed once)
        w = np.hanning(self.window)
        t = np.arange(self.window) / self.frequency
        self.mains_kernels = np.array([w * np.exp(-2j * np.pi * f * t) * 2 / w.sum() for f in MAINS_FREQUENCIES])
        self.n = 0  # samples received
        self.suspect_counts = [0] * nb_channels
        self.cooldown_until = [0.0] * nb_channels
        self.reference_fault = False
        self.reference_cooldown_until = 0.0
        self.clean_count = 0  # consecutive screenings without suspect channel
        self.checking = False  # whether a confirmation check is running in the capture process
        self.last_screening = None  # {channel: (mains amplitude in uV, saturated fraction)}, for diagnostics

    def set_in_use(self, mask):
        """
        Sets the channels in the bias loop after the initial check ("LOFF_IN_USE" message of the LeadoffCheck).
        """
        self.in_use = mask

    def update(self, points):
        """
        To be called with all the recorded samples, in order.

        Args:
            points: list of samples, each a list of the channel values in uV
        Returns:
            list of events
        """
        events = []
        for point in points:
            self.buffer[self.n % self.window] = point[:self.nb_channels]
            self.n += 1
            if not self.checking and self.n % self.interval == 0 and self.n >= self.window:
                events += self._screen()
        return events

    def _screen(self):
        x = np.roll(self.buffer, -(self.n % self.window), axis=0)  # chronological order
        saturated = np.mean(np.abs(x) >= self.full_scale_uv, axis=0)
        mains = np.abs(self.mains_kernels @ (x - x.mean(axis=0))).max(axis=0)
        self.last_screening = {ch: (float(mains[ch]), float(saturated[ch])) for ch in range(self.nb_channels)}

        in_use = _channels(self.in_use, self.nb_channels)
        for ch in in_use:
            suspect = mains[ch] > MAINS_SUSPECT_UV or saturated[ch] > RAILED_SUSPECT_FRACTION
            self.suspect_counts[ch] = self.suspect_counts[ch] + 1 if suspect else 0
        self.clean_count = self.clean_count + 1 if all(self.suspect_counts[ch] == 0 for ch in in_use) else 0
        now = self.clock()
        if self.reference_fault:
            if self.clean_count >= RECOVERY_SCREENINGS:
                self.reference_fault = False
                return [_event(REFERENCE_OR_BIAS_RECOVERED, in_use, self.n)]
            if now < self.reference_cooldown_until:
                return []
        candidates = [ch for ch in in_use if self.suspect_counts[ch] >= SUSPECT_SCREENINGS and now >= self.cooldown_until[ch]]
        if candidates:
            self.checking = True
            self.send(("LOFF_CHECK", sum(1 << ch for ch in candidates)))
        return []

    def check_done(self, result):
        """
        Decides from the result of a confirmation check ("LOFF_RESULT" message of the LeadoffCheck).

        Args:
            result: dict: result returned by LeadoffCheck.update
        Returns:
            list of events
        """
        self.checking = False
        mask, votes, n_votes = result["mask"], result["votes"], result["n_votes"]
        checked = _channels(mask, self.nb_channels)
        confirmed = [ch for ch in checked if votes[ch] * 2 > n_votes]
        for ch in checked:
            self.suspect_counts[ch] = 0
        in_use = _channels(self.in_use, self.nb_channels)
        all_off = len(in_use) > 1 and set(confirmed) == set(in_use)
        if all_off or (len(confirmed) == 0 and set(checked) == set(in_use)):
            # All the channels in use are off at once (bias off), or all look floating while connected (reference off).
            # A single channel in use that is off is disabled: whether its electrode or the bias fell, nothing else
            # depends on it
            self.reference_fault = True
            self.reference_cooldown_until = self.clock() + REFERENCE_COOLDOWN
            self.send(("LOFF_IN_USE", self.in_use))
            return [_event(REFERENCE_OR_BIAS_OFF, in_use, self.n)]
        events = []
        extreme = [ch for ch in checked if ch not in confirmed and self.last_screening[ch][0] > MAINS_DISABLE_UV]
        for reason, channels in ((REASON_OPEN, confirmed), (REASON_MAINS, extreme)):
            if channels:
                self.in_use &= ~sum(1 << ch for ch in channels)
                events.append(_event(DISABLED, channels, self.n, reason))
        not_confirmed = [ch for ch in checked if ch not in confirmed and ch not in extreme]
        if not_confirmed:
            for ch in not_confirmed:
                self.cooldown_until[ch] = self.clock() + NOT_CONFIRMED_COOLDOWN
            events.append(_event(SUSPECT_NOT_CONFIRMED, not_confirmed, self.n))
        self.send(("LOFF_IN_USE", self.in_use))
        return events
