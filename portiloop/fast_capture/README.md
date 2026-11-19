# Fast acquisition (2 to 16 kSPS)

The "Acquisition only (fast)" pipeline of the Simple GUI records the 8 channels at 2000, 4000, 8000 or 16000 Hz.
Python cannot read the ADS1299 faster than about 1 kSPS, so the capture runs in a C program, `adscapture.c`, which
reads each conversion by driving the SPI controller's registers directly through a small kernel module, `spimap`.
`portiloop/src/core/fast_capture.py` starts it from the GUI. The pipeline only appears on the Portiloops where it is
installed.

## Installing

On the Portiloop, as mendel, once:

```
bash ~/portiloop-software/portiloop/fast_capture/install.sh
sudo systemctl restart simplegui
```

The script builds `adscapture` (gcc on the Portiloop), and installs:
- `spimap/spimap.ko` in `/lib/modules/4.19.125-mtk/extra`, loaded at boot (`/etc/modules-load.d/portiloop-spimap.conf`);
- `spimap.rules` as `/etc/udev/rules.d/99-portiloop-spimap.rules`: `/dev/spimap` for the group `input` (as
  `/dev/spidev0.0`), and the SPI controller kept powered (`power/control` = `on`, no runtime suspend).

The module is the one built on red-ibis for the Mendel kernel 4.19.125-mtk (`linux-mtk 4-2`). The script checks that
the Portiloop runs this kernel and that the module's vermagic is the kernel's (`CONFIG_MODVERSIONS` is off and modules
are not signed, so the same file loads on every Portiloop with this kernel).

To remove it: delete these three files, run `sudo depmod -a`, and reboot (the module cannot be unloaded:
`CONFIG_MODULE_UNLOAD` is off).

## Recordings

Same folder, names and JSON files as the other pipelines (`_metadata.json`, `_stats.json`). The CSV has the columns
of the "Acquisition only" pipeline: `raw_ch1` to `raw_ch8` in uV with 3 decimals (the LSB is 0.022 uV at gain 24)
and `stimulation_on` (always 0). It is 75 to 100 bytes per row (78 with the test signal): about 2.3 to 2.8 GB per
hour at 8 kSPS, twice that at 16 kSPS.

- **Parts.** A new file starts every 28 800 000 rows (at most about 2.8 GB, below the 4 GB limit of FAT32):
  `<name>.csv`, then `<name>_part2.csv`, ..., each with a header. One hour fits in one file up to 8 kSPS; at
  16 kSPS, there is a new file every 30 minutes.
- **One row per conversion.** Each conversion gets its index from the time it was read, against a reference that
  follows the ADS's clock (its period is re-estimated every second). A missed conversion, a frame with a bad status
  byte, or one lost because the writer was too slow, repeats the previous row, and its row index is listed in
  `lost_samples` of `_stats.json`, as in the other pipelines.
- **Free space.** The recording stops when less than 200 MB are left (without an SD card, it goes to the home
  folder of the eMMC).
- **Statistics.** `_stats.json` has the usual fields (`os_time`, `nb_steps`, `os_sampling_frequency`,
  `nb_lost_samples`, `lost_samples`) and a `fast_capture` section: why the recording stopped, the number of rows
  and files, the lost conversions by cause, the read timeouts, the longest gap between reads, and the ADS period as
  per the Coral's clock.

The GUI display and the LSL stream ("Portiloop Raw Data") get a decimated copy of the signal: means of blocks of
samples, at 250 Hz. It is only meant for viewing.

Not available in this pipeline: filtering, detection, stimulation, and lead-off detection during the recording
(its checks need register commands, which would interrupt the reads). The electrodes OFF at the last impedance check
are disabled, as in the other pipelines.

## How it works

- The ADS is configured through the kernel driver (spidev): reset, the registers of `mod_config()`, read back and
  checked, START. Register commands stay at 1 MHz with 4 us between bytes (they fail from 8 MHz).
- One read through the kernel at 16 MHz sets the controller's clock, mode and chip-select timing. The program then
  disables the controller's interrupts and reads each conversion itself (RDATA and 27 bytes, about 18 us), and gives
  the controller back to the kernel at the end (also after a crash signal). The Python side then stops the ADS
  through the kernel, which also restores the controller if the program was killed.
- The reader polls DRDY on CPU 3 with `SCHED_FIFO` priority 10 (`LimitRTPRIO=10` in `simplegui.service`). A writer
  thread on the other CPUs writes the CSV and the decimated signal (to the Python process, through a pipe), through
  a buffer of 524 288 conversions (32 s at 16 kSPS), so that storage stalls do not delay the reads.
- The program stops on SIGTERM (the GUI's Stop), at the end of the duration, when the disk is nearly full, or if the
  ADS stops converting (no DRDY for 1 s), and if its parent process dies.
- With `-e` (the FFR pipeline), the writer thread also reads the triggers on stdin and writes them in the rows (see
  "FFR paradigm").

## FFR paradigm

The "FFR (fast)" pipeline is a fast acquisition during which the Portiloop plays the stimuli of the FFR protocol
(`FFR_Timeline_and_parameters.pdf`), with a trigger code in the recording at the onset of each one. The paradigm and
its parameters are in `portiloop/src/core/ffr.py`, the stimuli in `portiloop/sounds/ffr`. In short:

- 4 conditions, 120 ms each, with their trigger codes: `+` /da/ (1), `-` /da/ x -1 (2), `+N` /da/ + noise (3),
  `-N` -/da/ + noise (4). The noise is one of 4 tokens (drawn at random), at +10 dB SNR (RMS over the 120 ms).
- One Start is one run: 250 blocks of the 4 conditions shuffled within each block (1000 trials), SOA uniformly random
  between 195 and 205 ms, 1 s of silence before the first trial and 0.5 s after the last one (about 3.4 min). A
  session is 4 runs, so 4 recordings. The recording stops at the end of the run.
- The random parameters come from a seed drawn for each run (`ffr.seed` in the metadata).
- The volume of the GUI applies (calibrate it for 80 dB SPL with "Test Sound", which plays a positive /da/ every
  200 ms for 5 s). With "Cond A", the run is the same, with its triggers, in silence.

Its recordings have, besides the usual files:
- the CSV's last column, `trigger` (instead of `stimulation_on`): the trigger code in the row of each onset, 0
  elsewhere;
- `_trials.csv`: one line per trial (block, condition, code, noise token, onset frame in the sound, SOA to the next
  trial), with the row of its trigger, `offset_us` (onset - time of that row) and `in_csv`;
- `_triggers.csv`: the triggers as written by `adscapture` (trial, code, row, onset time, `offset_us`, `in_csv`);
- the `ffr` section of `_metadata.json` (the parameters, the seed, the SHA-256 of the stimulus files) and the
  `ffr_playback` and `fast_capture.triggers` sections of `_stats.json` (underruns, late triggers).

### Timing

The whole run is rendered as one sound at 48 kHz (the stimuli are resampled once, with `resample_poly`, so that ALSA
does not resample them), so the SOAs are exact to the frame (1/48 ms). `ffrplay.c` plays it through `softvol` and,
after each write of 5 ms, reads the frame being played (`snd_pcm_delay()`) with the time (CLOCK_MONOTONIC). It fits a
line to the last 2 s of these (the clock of the sound output), and sends the time of each onset to `adscapture -e`,
on its stdin, when it writes it (about 100 ms before it is played). `adscapture` writes the code in the row of the
conversion nearest to it, as per the time of each row (the expected time of its DRDY, which the reader already
follows). Through the time, and not a number of samples, the triggers stay right although the ADS's clock is not the
Coral's.

On bored-yarn (18 Nov 2026):
- The playback position moves by 4 frames (83 us) and lies within about 1 frame (24 us RMS) of the fitted line; the
  onset times of a run lie on a straight line in the onset frames to within 2 to 3 us RMS (14 us at most).
- The time of the rows follows the DRDYs read to within 1 to 7 us (checked against all the read times of a run). Over a
  run, the ADS's clock moves by up to 2.6 ms against the Coral's (0.25 % slower than nominal, varying by a few ppm):
  counting samples from the first trigger would put the last ones up to 40 rows off at 16 kSPS.
- Every trigger is in the row nearest to its onset, to within the 2 % by which the reader's reference can move per
  conversion (`offset_us` gives the rest). Runs of 1000 trials at 8 and 16 kSPS and of 100 trials at 16 kSPS: no
  underrun, all the triggers in the CSV, 9 missed conversions (filled) in the run of 1000 trials at 16 kSPS, none at
  8 kSPS; with the test signal, every half-period had its rows (4096 at 8 kSPS, 8192 at 16 kSPS) while the sound
  played.

Not included in the onset time: the latency of the codec, amplifier and earphones after the audio DMA, and the delay of
the ADS1299's filter. Both are constant; measure them once (for example, the sound output through a divider into a
channel) to place the onsets in absolute time.

The PulseAudio daemon keeps the sound card for about 5 s after a sound played through it (ALSA's `default`, as the
`aplay` run when `portiloop.src` is imported); `ffrplay` waits for it (the silence before the first trial is then
longer).

## Tests (red-ibis, 5 Nov 2026)

Internal test signal on all channels (a square wave at 0.977 Hz: each half-period must have exactly 0.512 x rate
rows when every conversion has its row):

| Rate | Duration | Lost | Half-periods with the wrong number of rows |
|---|---|---|---|
| 2000 | 30 s | 0 | 0 of 57 |
| 4000 | 30 s | 0 | 0 of 57 |
| 8000 | 20 s | 0 | 0 of 38 |
| 16000 | 2 min | 2 (missed, filled) | 0 of 232 |
| 16000 | 10 min | 14 (missed, filled: 1.5 per million) | 0 of 1167 |

With the reader forced to share its CPU with a busy loop at normal priority (8 kSPS, half of the conversions
missed, gaps up to 20 ms), the edges of the test signal stayed on the expected grid of rows to within one gap,
without drift. Through the GUI's start and stop (8 kSPS, 40 s): no loss, about 250 decimated samples per second on
the display queue and on LSL.

## Rebuilding the module

The module must be built against the exact running kernel, on the Portiloop itself (there is no aarch64 cross
compiler in the usual setups). For 4.19.125-mtk (package `linux-mtk (4-2)`), the source is the Mendel kernel
checkout (`mendel-linux/linux`, commit `d51d089ea374`, not part of this repository). The steps used on red-ibis:

1. **Copy the kernel source to the Portiloop** (about 300 MB), without `.git`, `Documentation`, `tools`, `samples`
   and the other architectures. Then add back every `Kconfig*` file and `tools/include`, which `olddefconfig` and
   `prepare` need:
   ```
   cd mendel-linux/linux
   tar czf - --exclude=.git --exclude=Documentation --exclude='./arch/[!a]*' --exclude='./arch/a[!r]*' \
       --exclude='./arch/arc' --exclude='./arch/arm' --exclude='./drivers/*.c' --exclude='./fs/*.c' \
       --exclude='./net/*.c' --exclude='./sound/*.c' --exclude='./tools' --exclude='./samples' --exclude='*.o' . \
     | ssh mendel@<portiloop> 'mkdir -p ~/kmod/linux && tar xzf - -C ~/kmod/linux'
   find . -name 'Kconfig*' -not -path './.git/*' | tar czf - -T - | ssh mendel@<portiloop> 'tar xzf - -C ~/kmod/linux'
   tar czf - tools/include | ssh mendel@<portiloop> 'tar xzf - -C ~/kmod/linux'
   ```
2. **Generate the parsers on another machine**, since Mendel has no bison or flex (for example with
   `pixi exec --spec bison --spec flex`): `scripts/kconfig/zconf.tab.c` (`bison -t -l scripts/kconfig/zconf.y`),
   `scripts/kconfig/zconf.lex.c` (`flex -L scripts/kconfig/zconf.l`), `scripts/dtc/dtc-parser.tab.c` and `.tab.h`
   (`bison -t -l scripts/dtc/dtc-parser.y`, with `--defines` for the header) and `scripts/dtc/dtc-lexer.lex.c`
   (`flex -L scripts/dtc/dtc-lexer.l`). Copy them to the same paths on the Portiloop.
3. **Install the `bc` stand-in**, since Mendel has no `bc` either. `kernel_build/timeconst.h` is the output of
   `echo 250 | bc -q kernel/time/timeconst.bc` (CONFIG_HZ is 250), and `kernel_build/bc` prints it. Copy
   `kernel_build/` to `~/kmod/bin` on the Portiloop. For a kernel with another CONFIG_HZ, regenerate `timeconst.h`.
4. **Prepare the tree** with the running kernel's configuration, on the Portiloop:
   ```
   cd ~/kmod/linux
   cp /boot/config-4.19.125-mtk .config
   make LOCALVERSION=-mtk YACC=: LEX=: olddefconfig
   PATH=$HOME/kmod/bin:$PATH make LOCALVERSION=-mtk YACC=: LEX=: -j4 prepare
   ```
   On red-ibis the resulting `.config` was identical to `/boot/config-4.19.125-mtk`.
5. **Build the module**, on the Portiloop, in this folder:
   ```
   make -C ~/kmod/linux M=$PWD/spimap LOCALVERSION=-mtk YACC=: LEX=: modules
   strip --strip-debug spimap/spimap.ko
   /sbin/modinfo -F vermagic spimap/spimap.ko   # must be the kernel's own ("4.19.125-mtk SMP preempt aarch64"
                                                # for 4.19.125-mtk, as for spidev.ko)
   ```
   `install.sh` checks the kernel version and the vermagic before installing it.

## Why direct register access

Measured on red-ibis on 23 Oct 2026, before this pipeline existed, with no electrodes and the internal test signal
on all channels ("lost" counted from the data):

| Capture | 8 kSPS | 16 kSPS |
|---|---|---|
| Python, best version tested (lead-off on) | 90 % or more lost (tops out at about 1650 reads/s) | did not finish |
| C, kernel spidev, DRDY interrupt | about 0.5 % lost | about 50 % lost |
| C, kernel spidev, DRDY polled on CPU 3 | 7 to 13 per million (20 min runs) | about 21 % lost (SPI too slow) |
| **C, direct register access (`spimap`), DRDY polled on CPU 3** | **2 in 9 576 676 (20 min)** | **2 in 10 784 968 (11 min)** |

- **SPI reads without the kernel.** A 28-byte read (RDATA and 27 bytes) through spidev takes 65 to 130 us, almost all
  of it kernel overhead. Driving the controller's registers directly takes about 18 us.
- **DRDY polled.** The kernel's GPIO event path drops about 1 DRDY edge in 180 at 8 kSPS, without a trace. The
  reader spins on the pin's level instead, on a CPU of its own.
- **Completion read from register 0x20** (bit 0: 1 when idle, 0 during a transfer). Polling the read-to-clear
  `STATUS0` flag instead missed the end of a transfer about 1 time in 500.

The prototype (its capture loop, `spiprobe.c`, the first cautious test of the direct reads, the test scripts and the
log of the long runs) is kept under the tag `high-rate-capture-prototype`, in `experiments/high_rate_capture/`.

## Security

Any process that can open `/dev/spimap` (root and the group `input`) can drive the SPI controller, including its DMA
addresses, which could be used to read or write the physical memory. On the Portiloops, mendel is in that group and already has
sudo.
