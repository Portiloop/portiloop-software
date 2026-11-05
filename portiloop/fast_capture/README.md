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

For another kernel, build `spimap/` against that kernel's source, on the Portiloop (there is no aarch64 cross
compiler in the usual setups). The steps used for 4.19.125-mtk (the Mendel kernel source, prepared with the running
kernel's `.config`, with the parsers generated elsewhere since Mendel has no bison or flex, and a stand-in for `bc`)
are in `experiments/high_rate_capture/README.md` on the `high-rate-capture` branch. Then:

```
make -C <prepared kernel tree> M=$PWD/spimap LOCALVERSION=-mtk YACC=: LEX=: modules
strip --strip-debug spimap/spimap.ko
```

## Security

Any process that can open `/dev/spimap` (root and the group `input`) can drive the SPI controller, including its DMA
addresses, which could be used to read or write the physical memory. On the Portiloops, mendel is in that group and already has
sudo.
