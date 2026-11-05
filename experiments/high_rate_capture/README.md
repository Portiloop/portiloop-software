# High-rate capture experiment (8 and 16 kSPS)

Prototype of an ADS1299 capture at the chip's highest data rates, tested on one Portiloop (red-ibis) on
23 Oct 2026. It is **not** used by the Portiloop software: nothing here is installed, built or loaded by `setup.py`,
the Makefile or `update.sh`.

## Results

30 s to 20 min runs on red-ibis, no electrodes, the ADS's internal test signal on all channels, "lost" counted from
the data (each half-period of the square wave must have exactly 4096 samples at 8 kSPS, 8192 at 16 kSPS):

| Capture | 8 kSPS | 16 kSPS |
|---|---|---|
| Python, best version tested (lead-off on) | 90 % or more lost (tops out at about 1650 reads/s) | did not finish |
| C, kernel spidev, DRDY interrupt | about 0.5 % lost | about 50 % lost |
| C, kernel spidev, DRDY polled on CPU 3 | 7 to 13 per million (20 min runs) | about 21 % lost (SPI too slow) |
| **C, direct register access (`spimap`), DRDY polled on CPU 3** | **2 in 9 576 676 (20 min)** | **2 in 10 784 968 (11 min, stopped early)** |

The log of the last row is in `results/2026-10-23_red-ibis_long.txt`. In that run there were no read timeouts and no
bad status bytes; the longest gap between reads was 114 µs at 16 kSPS (the period is 62.5 µs).

What made the difference:
- **SPI reads without the kernel.** A 28-byte read (RDATA and 27 bytes) through spidev takes 65 to 130 µs, almost all
  of it kernel overhead. Driving the controller's registers directly takes about 18 µs.
- **DRDY polled.** The kernel's GPIO event path drops about 1 DRDY edge in 180 at 8 kSPS, without a trace. The
  capture spins on the pin's level instead, on a core of its own (CPU 3, `SCHED_FIFO`).
- **Completion read from register 0x20** (bit 0: 1 when idle, 0 during a transfer). Polling the read-to-clear
  `STATUS0` flag instead missed the end of a transfer about 1 time in 500.

## Files

- `spimap/spimap.c`, `spimap/Kbuild`: the kernel module. It creates `/dev/spimap` (root only, mode 0600), whose only
  operation maps the 4 KB register page of SPI controller 0 (physical address 0x1100c000) into the caller.
- `capture/spiprobe.c`: the first, cautious test. At 250 SPS it reads 5 frames directly, compares them with kernel
  reads, checks that no SPI interrupt fired, and checks that the kernel driver still works afterwards.
- `capture/adsfast.c`: the capture loop. It configures the ADS through spidev, does one kernel read at 16 MHz so
  that the kernel sets the controller's clock, mode and chip-select timing, disables the controller's interrupts,
  then reads each conversion directly. It writes the 24 data bytes of each frame to a file and restores the
  controller's command register at the end. Usage:
  `[ADS_PRIO=n] adsfast <seconds> <out_file> <config bytes in hex, from register 0x01>`.
- `capture/fast.sh` (2 min at 8 and 16 kSPS) and `capture/long.sh` (20 min each): the test runs, with the loss count
  from the test signal.
- `kernel_build/bc`, `kernel_build/timeconst.h`: a stand-in for `bc`, which the Coral does not have (see below).

## Building

Everything is built on the Coral itself (there is no aarch64 cross compiler here). The module must be built against
the exact running kernel: 4.19.125-mtk, package `linux-mtk (4-2)`, whose source is the Mendel kernel checkout
(`mendel-linux/linux`, commit `d51d089ea374`, untracked in this repo).

1. **Copy the kernel source to the Coral** (about 300 MB), without `.git`, `Documentation`, `tools`, `samples` and the
   other architectures. Then add back every `Kconfig*` file and `tools/include`, which `olddefconfig` and
   `prepare` need:
   ```
   cd mendel-linux/linux
   tar czf - --exclude=.git --exclude=Documentation --exclude='./arch/[!a]*' --exclude='./arch/a[!r]*' \
       --exclude='./arch/arc' --exclude='./arch/arm' --exclude='./drivers/*.c' --exclude='./fs/*.c' \
       --exclude='./net/*.c' --exclude='./sound/*.c' --exclude='./tools' --exclude='./samples' --exclude='*.o' . \
     | ssh mendel@<coral> 'mkdir -p ~/kmod/linux && tar xzf - -C ~/kmod/linux'
   find . -name 'Kconfig*' -not -path './.git/*' | tar czf - -T - | ssh mendel@<coral> 'tar xzf - -C ~/kmod/linux'
   tar czf - tools/include | ssh mendel@<coral> 'tar xzf - -C ~/kmod/linux'
   ```
2. **Generate the parsers here**, since the Coral has no bison or flex (`pixi exec --spec bison --spec flex`):
   `scripts/kconfig/zconf.tab.c` (`bison -t -l scripts/kconfig/zconf.y`), `scripts/kconfig/zconf.lex.c`
   (`flex -L scripts/kconfig/zconf.l`), `scripts/dtc/dtc-parser.tab.c` and `.tab.h`
   (`bison -t -l scripts/dtc/dtc-parser.y`, with `--defines` for the header) and `scripts/dtc/dtc-lexer.lex.c`
   (`flex -L scripts/dtc/dtc-lexer.l`), copied to the same paths on the Coral.
3. **Install the `bc` stand-in.** `timeconst.h` is the output of `echo 250 | bc -q kernel/time/timeconst.bc`
   (CONFIG_HZ is 250), computed here. Copy `kernel_build/` to `~/kmod/bin` on the Coral.
4. **Prepare the tree** with the running kernel's configuration, on the Coral:
   ```
   cd ~/kmod/linux
   cp /boot/config-4.19.125-mtk .config
   make LOCALVERSION=-mtk YACC=: LEX=: olddefconfig
   PATH=$HOME/kmod/bin:$PATH make LOCALVERSION=-mtk YACC=: LEX=: -j4 prepare
   ```
   The resulting `.config` was identical to `/boot/config-4.19.125-mtk`.
5. **Build the module and the programs**, on the Coral, from a copy of this folder:
   ```
   make -C ~/kmod/linux M=$PWD/spimap LOCALVERSION=-mtk YACC=: LEX=: modules
   /sbin/modinfo -F vermagic spimap/spimap.ko   # "4.19.125-mtk SMP preempt aarch64", identical to the
                                                # kernel's own modules (e.g. spidev.ko)
   gcc -O2 -Wall -o capture/spiprobe capture/spiprobe.c
   gcc -O2 -Wall -o capture/adsfast capture/adsfast.c
   ```

## Running

Before anything, check that no process holds the ADS (the Simple GUI or a notebook with `/dev/spidev0.0` or
`/dev/gpiochip0` open), since the programs reconfigure it.

```
P=/sys/devices/platform/soc/1100c000.spi/power/control
echo on | sudo tee $P            # keep the controller powered while it is driven directly
sudo /sbin/insmod spimap/spimap.ko
sudo capture/spiprobe            # first
sudo capture/fast.sh             # then; it sets power/control back to auto at exit
```

- **The module cannot be removed** once loaded (`CONFIG_MODULE_UNLOAD` is off). It stays until the next reboot.
- **While a program drives the controller, the kernel must not use it.** The programs disable the controller's
  interrupts (`FINISH_IE`, `PAUSE_IE`) and restore the command register when they exit, and the kernel driver
  works again afterwards (checked by reading the ADS's ID register, 0x3E). A program killed with SIGKILL would leave
  the interrupts disabled until the next kernel transfer rewrites the register.
- **Register commands to the ADS go through the kernel**, at 1 MHz with 4 µs between bytes: from 8 MHz they fail,
  because the datasheet requires 4 tCLK between command bytes. Data reads were checked to be clean up to 20 MHz.
- **A bug at this level can hang or reboot the Coral.** Nothing is written to its storage by these programs.

## Not done

1. **A capture core:** `adsfast` only writes raw frames to a file. It has to become a process feeding Python
   through shared memory, with the lead-off checks still working (they use register commands).
2. **Decimation for the online pipeline:** the Python filters and detectors cannot run at 8 kSPS.
3. **A recording format:** an 8 h night at 8 kSPS is about 230 million samples, about 5.5 GB as raw 24-bit data but
   several times more as CSV.
4. **Deployment:** building the module for the kernel of every Portiloop and loading it at boot. Any root process can
   drive the SPI controller through `/dev/spimap`.
5. **Tests with electrodes and over full nights.** The corrupted frames seen with electrodes (about 3 per hour)
   are a separate, unexplained issue.
