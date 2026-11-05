#!/bin/bash
# 2 min at 8 and 16 kSPS with direct SPI register access, test signal; losses counted from the data.
cd "$(dirname "$0")"
P=/sys/devices/platform/soc/1100c000.spi/power/control
PY=/home/mendel/miniforge3/envs/portiloop/bin/python
trap 'echo auto > $P; rm -f /dev/shm/t.bin; echo "restored: SPI PM $(cat $P)"' EXIT
echo on > $P
for dr in 91 90; do
    ADS_PRIO=70 taskset -c 3 ./adsfast 120 /dev/shm/t.bin $dr d0 fc 00 65 65 65 65 65 65 65 65 00 00 00 00 00 00 00 00 20
    $PY - /dev/shm/t.bin $dr <<PYEOF
import sys, numpy as np
raw = np.fromfile(sys.argv[1], np.uint8).reshape(-1, 24)
c = (raw[:, 0].astype(np.int64) << 16) | (raw[:, 1].astype(np.int64) << 8) | raw[:, 2]
c = np.where(c >= 2**23, c - 2**24, c)
hi = c > (np.percentile(c, 5) + np.percentile(c, 95)) / 2
runs = np.diff(np.where(hi[1:] != hi[:-1])[0] + 1)
half = {"91": 4096, "90": 8192}[sys.argv[2]]
lost = int((half - runs).clip(0).sum()); extra = int((runs - half).clip(0).sum())
vals, counts = np.unique(runs, return_counts=True)
print("   from the data: %d samples, lost %d, duplicated %d, %d/%d half-periods complete; run lengths %s" % (len(c), lost, extra, int((runs == half).sum()), len(runs), dict(zip(vals.tolist()[:10], counts.tolist()[:10]))))
PYEOF
done
