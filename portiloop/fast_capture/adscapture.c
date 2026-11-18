/* Fast acquisition of the ADS1299 (2 to 16 kSPS), for the "Acquisition only (fast)" pipeline
 * (portiloop/src/core/fast_capture.py starts it). See README.md in this folder.
 *
 * Usage: adscapture [options] <config bytes in hex, from register 0x01>
 *   -r rate     sampling rate in Hz (must match the DR bits of CONFIG1)
 *   -c n        number of channels in the CSV (default 8)
 *   -k scale    uV per ADC code
 *   -o prefix   CSV files <prefix>.csv, <prefix>_part2.csv, ... (no CSV without -o)
 *   -n rows     rows per CSV file (default 28800000)
 *   -s path     statistics (JSON), written at the end
 *   -d n        decimated output on stdout: means of n samples, as float32 uV (0: none, the default)
 *   -t seconds  duration (0: until SIGTERM or SIGINT, the default)
 *   -m MB       stop when the free space of the CSV folder falls below this (default 200)
 *   -p prio     SCHED_FIFO priority of the reader (default 10), -u cpu: its CPU (default 3)
 *   -e          triggers on stdin (lines "<trial> <code> <CLOCK_MONOTONIC time in s>", in increasing time, received
 *               before the conversion at that time is read, as ffrplay sends them): each code is written in the
 *               row of the conversion nearest to its time, in the last column (then named "trigger" instead of
 *               "stimulation_on"), and listed in <prefix>_triggers.csv
 *
 * The ADS is configured through spidev, then each conversion is read by driving the SPI controller's registers
 * directly (/dev/spimap, about 18 us per read, DRDY polled on its own CPU). A writer thread on the other CPUs writes
 * the CSV, so that SD card stalls do not delay the reads. Each conversion gets its index from the time it was read,
 * so the rows of the CSV stay one per conversion: a missed conversion (or a frame with a bad status byte, or one
 * lost because the buffer was full) repeats the previous row, and its index is listed in the statistics.
 *
 * The time of each row (for the triggers) is the expected time of its DRDY as per the reference of the reader
 * (a + index * p below), in CLOCK_MONOTONIC, the clock of ffrplay's onset times. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/gpio.h>
#include <linux/spi/spidev.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>

/* MediaTek SPI controller registers */
#define CFG1 0x04
#define TX_DATA 0x10
#define RX_DATA 0x14
#define CMD 0x18
#define STATUS0 0x1c
#define BUSY 0x20  /* bit 0: 1 when idle, 0 during a transfer */
#define CMD_ACT (1u << 0)
#define CMD_RESUME (1u << 1)
#define CMD_RST (1u << 2)
#define CMD_PAUSE_EN (1u << 4)
#define CMD_FINISH_IE (1u << 16)
#define CMD_PAUSE_IE (1u << 17)
#define CFG1_LEN_MASK 0x3ff0000u
#define CFG1_LOOP_MASK 0xff00u
#define R(off) regs[(off) / 4]

#define SPI_POWER_CONTROL "/sys/devices/platform/soc/1100c000.spi/power/control"
#define DRDY_LINE 45
#define START_LINE 37

#define RING_BITS 19  /* 524288 frames: 32 s at 16 kSPS */
#define RING_SIZE (1u << RING_BITS)
#define RING_MASK (RING_SIZE - 1)
#define MAX_LOST_LISTED 100000
#define LATE_FRACTION 0.9   /* a conversion read up to 0.9 period late is not counted as missed */
#define SLEW_FRACTION 0.02  /* the time reference follows late reads by at most 2 % of a period per conversion */

struct frame {
    double t;        /* expected time of its DRDY (CLOCK_MONOTONIC) */
    uint32_t index;  /* conversion index (CSV row) */
    uint8_t bad;     /* bad status byte */
    uint8_t data[24];
};

static int spi = -1;
static volatile uint32_t *regs;
static uint32_t cmd0, cmd;
static volatile sig_atomic_t stop;
static atomic_int reader_done;

static struct frame *ring;
static atomic_uint_fast64_t head, tail;  /* written by the reader, by the writer */

/* settings */
static double rate, scale;
static int nch = 8, decimation, cpu = 3, prio = 10;
static uint64_t part_rows = 28800000;
static const char *prefix, *stats_path;
static double duration, min_free_mb = 200;
static int events_on;  /* -e */

/* reader statistics */
static uint64_t missed, overflows, timeouts, retried_ok, gaps;
static double max_gap, max_read, t_first, t_last, period_est;
/* writer statistics */
static uint64_t rows, bad_frames, decim_dropped, nb_lost;
static uint32_t *lost;
static size_t lost_cap;
static int nb_parts;
static char stop_reason[128] = "stopped";
static int write_failed, failed;  /* failed: setup or capture error */

static void logmsg(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void logmsg(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fputs("adscapture: ", stderr); vfprintf(stderr, fmt, ap); fputc('\n', stderr);
    va_end(ap);
}
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static void on_stop(int s) { (void)s; stop = 1; }
static void on_fatal(int s) {  /* give the controller back to the kernel driver */
    if (regs) R(CMD) = cmd0;
    signal(s, SIG_DFL);
    raise(s);
}

/* ---------- kernel SPI (configuration) ---------- */

/* one byte per transfer at 1 MHz with 4 us between bytes: register commands fail from 8 MHz (4 tCLK between bytes) */
static int command(const uint8_t *tx, uint8_t *rx, int n) {
    struct spi_ioc_transfer t[32];
    memset(t, 0, sizeof t);
    for (int i = 0; i < n; i++) {
        t[i].tx_buf = (uintptr_t)&tx[i]; t[i].rx_buf = rx ? (uintptr_t)&rx[i] : 0; t[i].len = 1;
        t[i].speed_hz = 1000000; t[i].bits_per_word = 8; t[i].delay_usecs = 4;
    }
    return ioctl(spi, SPI_IOC_MESSAGE(n), t) < 0 ? -1 : 0;
}
static int command1(uint8_t b) { return command(&b, NULL, 1); }

/* ---------- direct reads ---------- */

/* as mtk_spi_reset(): set RST, read the register back (which also lengthens the pulse), clear RST */
static void soft_reset(void) {
    R(CMD) = cmd | CMD_RST;
    (void)R(CMD);
    R(CMD) = cmd;
    (void)R(CMD);
}

/* one RDATA + 27 NOP frame through the FIFO; returns 0 on timeout (after a soft reset) */
static int direct_read_once(uint8_t *rx) {
    for (int i = 0; i < 7; i++) R(TX_DATA) = i == 0 ? 0x12 : 0;
    R(CMD) = cmd | CMD_ACT;
    double t0 = now();
    /* completion from the busy register (polling the read-to-clear STATUS0 can miss its finish flag):
     * wait for busy, then idle again */
    while (R(BUSY) & 1)
        if (now() - t0 > 10e-6) break;  /* not started yet: checked below */
    while (!(R(BUSY) & 1))
        if (now() - t0 > 60e-6) break;
    (void)R(STATUS0);  /* clear the finish flag */
    if (!(R(BUSY) & 1)) {
        soft_reset();
        return 0;
    }
    uint32_t w[7];
    for (int i = 0; i < 7; i++) w[i] = R(RX_DATA);
    memcpy(rx, w, 28);
    soft_reset();  /* after each packet, as mtk_spi_set_cs() does */
    return 1;
}

static int direct_read(uint8_t *rx) {
    if (direct_read_once(rx)) return 1;
    if (direct_read_once(rx)) { retried_ok++; return 1; }  /* the conversion is still there */
    return 0;
}

/* ---------- writer ---------- */

static int csv_fd = -1;
static char *wbuf;
#define WBUF_SIZE (1 << 20)
static size_t wlen;
static char part_name[4096];
static int out_dec = -1;  /* decimated output (stdout), -1 when off */

static void fail_write(const char *what) {
    if (!write_failed) {
        write_failed = 1;
        snprintf(stop_reason, sizeof stop_reason, "%s: %s", what, strerror(errno));
        logmsg("%s", stop_reason);
    }
    stop = 1;
}

static void check_free_space(void) {
    struct statvfs s;
    if (fstatvfs(csv_fd, &s) == 0 && (double)s.f_bavail * s.f_frsize < min_free_mb * 1e6) {
        if (!stop) {
            snprintf(stop_reason, sizeof stop_reason, "disk full (less than %.0f MB free)", min_free_mb);
            logmsg("%s: stopping", stop_reason);
        }
        stop = 1;
    }
}

#define FREE_SPACE_CHECK_BYTES (16u << 20)  /* checked every 16 MB written */
static size_t written_since_check;

static void flush_csv(void) {
    size_t off = 0;
    while (off < wlen && !write_failed) {
        ssize_t n = write(csv_fd, wbuf + off, wlen - off);
        if (n < 0) { if (errno == EINTR) continue; fail_write("CSV write"); break; }
        off += n;
    }
    written_since_check += off;
    wlen = 0;
    if (written_since_check >= FREE_SPACE_CHECK_BYTES) {
        written_since_check = 0;
        check_free_space();
    }
}

static void close_part(void) {
    if (csv_fd < 0) return;
    flush_csv();
    if (fsync(csv_fd) < 0 && !write_failed) fail_write("CSV fsync");
    close(csv_fd);
    csv_fd = -1;
}

static void open_part(void) {
    nb_parts++;
    if (nb_parts == 1) snprintf(part_name, sizeof part_name, "%s.csv", prefix);
    else snprintf(part_name, sizeof part_name, "%s_part%d.csv", prefix, nb_parts);
    csv_fd = open(part_name, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (csv_fd < 0) { fail_write(part_name); return; }
    for (int c = 0; c < nch; c++) wlen += sprintf(wbuf + wlen, "raw_ch%d,", c + 1);
    wlen += sprintf(wbuf + wlen, events_on ? "trigger\n" : "stimulation_on\n");
    logmsg("writing %s", part_name);
}

/* uV with 3 decimals (the LSB is about 0.02 uV), without printf */
static char *put_uv(char *p, int32_t code) {
    long long m = llround(code * scale * 1000.0);
    if (m < 0) { *p++ = '-'; m = -m; }
    char tmp[24]; int n = 0;
    long long ip = m / 1000; int fp = (int)(m % 1000);
    do { tmp[n++] = '0' + ip % 10; ip /= 10; } while (ip);
    while (n) *p++ = tmp[--n];
    *p++ = '.'; *p++ = '0' + fp / 100; *p++ = '0' + fp / 10 % 10; *p++ = '0' + fp % 10;
    return p;
}

static int32_t code_of(const uint8_t *d) {
    int32_t v = (d[0] << 16) | (d[1] << 8) | d[2];
    return v & 0x800000 ? v - 0x1000000 : v;
}

/* ---------- triggers (-e) ---------- */

struct event { double t; uint32_t trial; int code; };
#define EV_CAP 4096  /* pending triggers: ffrplay sends them about 0.1 s ahead, 5 per second */
static struct event evq[EV_CAP];
static uint64_t ev_head, ev_tail;
static char evbuf[4096];
static size_t evlen;
static int ev_eof;
static uint64_t ev_received, ev_in_csv, ev_late, ev_same_row, ev_malformed, ev_overflow;
static FILE *trig_f;

/* reads the triggers that arrived on stdin (non-blocking) */
static void read_events(void) {
    if (!events_on || ev_eof) return;
    for (;;) {
        ssize_t n = read(STDIN_FILENO, evbuf + evlen, sizeof evbuf - 1 - evlen);
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno != EAGAIN) { logmsg("triggers input: %s", strerror(errno)); ev_eof = 1; }
            return;
        }
        if (n == 0) { ev_eof = 1; return; }  /* the player has exited */
        evlen += n;
        char *s = evbuf, *nl;
        while ((nl = memchr(s, '\n', evbuf + evlen - s))) {
            *nl = 0;
            unsigned trial; int code; double t;
            if (sscanf(s, "%u %d %lf", &trial, &code, &t) != 3 || code <= 0) ev_malformed++;
            else if (ev_head - ev_tail >= EV_CAP) ev_overflow++;
            else { evq[ev_head++ % EV_CAP] = (struct event){ t, trial, code }; ev_received++; }
            s = nl + 1;
        }
        evlen -= s - evbuf;
        memmove(evbuf, s, evlen);
        if (evlen == sizeof evbuf - 1) { ev_malformed++; evlen = 0; }  /* a line too long */
    }
}

/* The code of the triggers of the row 'row' expected at t_row: those from the end of the previous row (half a period
 * after its time) to half a period after t_row (the rows follow one another without gaps, even when the reference
 * of the reader moves), and the late ones (received after their row was written: listed with their row, but not in
 * the CSV). */
static double row_end = -INFINITY;  /* of the previous row */

static int take_triggers(uint64_t row, double t_row) {
    int code = 0;
    double half = 0.5 / rate;
    while (ev_tail != ev_head && evq[ev_tail % EV_CAP].t < t_row + half) {
        struct event *e = &evq[ev_tail++ % EV_CAP];
        double off = e->t - t_row;
        uint64_t r = row;
        int in_csv = 1;
        if (e->t < row_end) {  /* its row was already written */
            ev_late++;
            in_csv = 0;
            double back = floor(-off * rate + 0.5);
            r = back <= (double)row ? row - (uint64_t)back : 0;
            off += back / rate;
        } else if (code) {
            ev_same_row++;  /* two triggers in one row: the first code is kept in the CSV */
            in_csv = 0;
        } else {
            code = e->code;
            ev_in_csv++;
        }
        if (trig_f) fprintf(trig_f, "%u,%d,%llu,%.7f,%.1f,%d\n", e->trial, e->code, (unsigned long long)r, e->t,
                            off * 1e6, in_csv);
    }
    row_end = t_row + half;
    return code;
}

static double dec_sum[8];
static int dec_n;

static void emit(const uint8_t *data, double t_row) {
    int code = events_on ? take_triggers(rows, t_row) : 0;
    if (prefix && !write_failed) {
        if (csv_fd < 0 || (rows > 0 && rows % part_rows == 0)) {
            close_part();
            if (!write_failed) open_part();
        }
        if (wlen > WBUF_SIZE - 512) flush_csv();
        char *p = wbuf + wlen;
        for (int c = 0; c < nch; c++) { p = put_uv(p, code_of(data + 3 * c)); *p++ = ','; }
        p += sprintf(p, "%d\n", code);
        wlen = p - wbuf;
    }
    rows++;
    if (out_dec >= 0) {
        for (int c = 0; c < nch; c++) dec_sum[c] += code_of(data + 3 * c) * scale;
        if (++dec_n == decimation) {
            float f[8];
            for (int c = 0; c < nch; c++) { f[c] = (float)(dec_sum[c] / decimation); dec_sum[c] = 0; }
            dec_n = 0;
            if (write(out_dec, f, sizeof(float) * nch) < 0) {
                if (errno == EAGAIN) decim_dropped++;  /* the reader is late: display only */
                else { logmsg("decimated output closed (%s)", strerror(errno)); out_dec = -1; }
            }
        }
    }
}

static void note_lost(uint64_t index) {
    nb_lost++;
    if (nb_lost > MAX_LOST_LISTED) return;
    if (nb_lost > lost_cap) {
        lost_cap = lost_cap ? 2 * lost_cap : 1024;
        lost = realloc(lost, lost_cap * sizeof *lost);
        if (!lost) { logmsg("out of memory"); exit(3); }
    }
    lost[nb_lost - 1] = (uint32_t)index;
}

static void *writer(void *arg) {
    (void)arg;
    uint8_t prev[24] = {0};
    uint64_t t = 0;
    double t_flush = now();
    for (;;) {
        uint64_t h = atomic_load_explicit(&head, memory_order_acquire);
        read_events();  /* after head: the triggers of all these rows were sent before they were read */
        if (t == h) {
            if (atomic_load(&reader_done)) break;
            if (prefix && wlen && now() - t_flush > 1.0) {  /* at least once a second */
                flush_csv();
                if (trig_f) fflush(trig_f);
                t_flush = now();
            }
            usleep(2000);
            continue;
        }
        while (t != h) {
            struct frame *f = &ring[t & RING_MASK];
            while (rows < f->index) {  /* missed conversions */
                emit(prev, f->t - (f->index - rows) / rate);
                note_lost(rows - 1);
            }
            if (f->bad) { bad_frames++; emit(prev, f->t); note_lost(rows - 1); }
            else { emit(f->data, f->t); memcpy(prev, f->data, 24); }
            t++;
            atomic_store_explicit(&tail, t, memory_order_release);
        }
    }
    close_part();
    if (trig_f && fclose(trig_f) && !write_failed) fail_write("triggers file");
    trig_f = NULL;
    return NULL;
}

/* ---------- statistics ---------- */

static void json_str(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') fputc('\\', f);
        if ((unsigned char)*s >= 0x20) fputc(*s, f);
    }
    fputc('"', f);
}

static void write_stats(void) {
    FILE *f = fopen(stats_path, "w");
    if (!f) { logmsg("cannot write %s: %s", stats_path, strerror(errno)); return; }
    double os_time = t_last - t_first;
    uint64_t steps = rows ? rows - 1 : 0;
    fprintf(f, "{\n    \"os_time\": %.6f,\n    \"nb_steps\": %llu,\n", os_time, (unsigned long long)steps);
    fprintf(f, "    \"os_step_duration\": %.9g,\n    \"os_sampling_frequency\": %.6f,\n",
            steps ? os_time / steps : 0, os_time > 0 ? steps / os_time : 0);
    fprintf(f, "    \"nb_lost_samples\": %llu,\n    \"lost_samples\": [", (unsigned long long)nb_lost);
    uint64_t listed = nb_lost < MAX_LOST_LISTED ? nb_lost : MAX_LOST_LISTED;
    for (uint64_t i = 0; i < listed; i++) fprintf(f, "%s%u", i ? ", " : "", lost[i]);
    fprintf(f, "],\n    \"fast_capture\": {\n");
    fprintf(f, "        \"stop_reason\": "); json_str(f, stop_reason);
    fprintf(f, ",\n        \"rows\": %llu,\n        \"csv_files\": %d,\n", (unsigned long long)rows, nb_parts);
    fprintf(f, "        \"lost_samples_listed\": %llu,\n", (unsigned long long)listed);
    fprintf(f, "        \"missed_conversions\": %llu,\n        \"bad_status_frames\": %llu,\n",
            (unsigned long long)missed, (unsigned long long)bad_frames);
    fprintf(f, "        \"buffer_overflows\": %llu,\n        \"read_timeouts\": %llu,\n",
            (unsigned long long)overflows, (unsigned long long)timeouts);
    fprintf(f, "        \"reads_recovered_by_retry\": %llu,\n", (unsigned long long)retried_ok);
    fprintf(f, "        \"gaps_over_1.5_periods\": %llu,\n        \"longest_gap_us\": %.1f,\n",
            (unsigned long long)gaps, max_gap * 1e6);
    fprintf(f, "        \"longest_read_us\": %.1f,\n        \"ads_period_us\": %.6f,\n", max_read * 1e6, period_est * 1e6);
    fprintf(f, "        \"decimated_samples_dropped\": %llu", (unsigned long long)decim_dropped);
    if (events_on)
        fprintf(f, ",\n        \"triggers\": {\"received\": %llu, \"in_csv\": %llu, \"late\": %llu, \"same_row\": %llu, "
                   "\"after_the_end\": %llu, \"malformed\": %llu, \"overflows\": %llu}",
                (unsigned long long)ev_received, (unsigned long long)ev_in_csv, (unsigned long long)ev_late,
                (unsigned long long)ev_same_row, (unsigned long long)(ev_head - ev_tail),
                (unsigned long long)ev_malformed, (unsigned long long)ev_overflow);
    fprintf(f, "\n    }\n}\n");
    fclose(f);
}

/* ---------- main ---------- */

static void usage(void) {
    fprintf(stderr, "usage: adscapture -r rate -k scale -s stats [-c n] [-o prefix] [-n rows] [-d n] [-t s] [-m MB] "
                    "[-p prio] [-u cpu] [-e] config bytes (hex, from register 0x01)\n");
    exit(2);
}

static int check_power_control(void) {
    char s[16] = {0};
    FILE *f = fopen(SPI_POWER_CONTROL, "r");
    if (!f || !fgets(s, sizeof s, f)) { logmsg("cannot read %s", SPI_POWER_CONTROL); if (f) fclose(f); return -1; }
    fclose(f);
    if (strncmp(s, "on", 2)) {
        logmsg("%s is not \"on\": the SPI controller could be suspended while it is driven directly "
               "(the udev rule of install.sh sets it)", SPI_POWER_CONTROL);
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    int opt;
    while ((opt = getopt(argc, argv, "r:c:k:o:n:s:d:t:m:p:u:e")) != -1) {
        switch (opt) {
        case 'r': rate = atof(optarg); break;
        case 'c': nch = atoi(optarg); break;
        case 'k': scale = atof(optarg); break;
        case 'o': prefix = optarg; break;
        case 'n': part_rows = strtoull(optarg, NULL, 10); break;
        case 's': stats_path = optarg; break;
        case 'd': decimation = atoi(optarg); break;
        case 't': duration = atof(optarg); break;
        case 'm': min_free_mb = atof(optarg); break;
        case 'p': prio = atoi(optarg); break;
        case 'u': cpu = atoi(optarg); break;
        case 'e': events_on = 1; break;
        default: usage();
        }
    }
    int ncfg = argc - optind;
    if (rate <= 0 || scale <= 0 || !stats_path || nch < 1 || nch > 8 || part_rows < 1 || decimation < 0
            || ncfg < 1 || ncfg > 23)
        usage();
    uint8_t cfg[2 + 23];
    cfg[0] = 0x40 | 0x01; cfg[1] = ncfg - 1;
    for (int i = 0; i < ncfg; i++) cfg[2 + i] = strtol(argv[optind + i], NULL, 16);
    double cfg_rate = 16000.0 / (1 << (cfg[2] & 0x07));
    if (fabs(cfg_rate - rate) > 0.5) { logmsg("-r %.0f does not match CONFIG1 (%.0f SPS)", rate, cfg_rate); return 2; }
    double period = 1.0 / rate;

    prctl(PR_SET_PDEATHSIG, SIGTERM);  /* stop if the Python process dies */
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_stop;  /* no SA_RESTART */
    sigaction(SIGINT, &sa, NULL); sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);
    if (check_power_control()) return 1;

    /* ---- configuration through the kernel ---- */
    spi = open("/dev/spidev0.0", O_RDWR); if (spi < 0) { logmsg("/dev/spidev0.0: %s", strerror(errno)); return 1; }
    uint8_t mode = SPI_MODE_1; ioctl(spi, SPI_IOC_WR_MODE, &mode);
    uint32_t base = 1000000; ioctl(spi, SPI_IOC_WR_MAX_SPEED_HZ, &base);
    int chip = open("/dev/gpiochip0", O_RDWR); if (chip < 0) { logmsg("/dev/gpiochip0: %s", strerror(errno)); return 1; }
    struct gpiohandle_request start_line, drdy;
    memset(&start_line, 0, sizeof start_line); memset(&drdy, 0, sizeof drdy);
    start_line.lineoffsets[0] = START_LINE; start_line.lines = 1; start_line.flags = GPIOHANDLE_REQUEST_OUTPUT;
    strcpy(start_line.consumer_label, "adscapture");  /* held low, as Backend does */
    if (ioctl(chip, GPIO_GET_LINEHANDLE_IOCTL, &start_line) < 0) { logmsg("START line: %s", strerror(errno)); return 1; }
    drdy.lineoffsets[0] = DRDY_LINE; drdy.lines = 1; drdy.flags = GPIOHANDLE_REQUEST_INPUT;
    strcpy(drdy.consumer_label, "adscapture-drdy");
    if (ioctl(chip, GPIO_GET_LINEHANDLE_IOCTL, &drdy) < 0) { logmsg("DRDY line: %s", strerror(errno)); return 1; }
    struct gpiohandle_data v;

    /* as capture_process(): reset, configure, start */
    if (command1(0x0A) || command1(0x11) || command1(0x06)) { logmsg("SPI command: %s", strerror(errno)); return 1; }
    usleep(1000);  /* 18 tCLK to execute RESET */
    if (command1(0x11) || command(cfg, NULL, 2 + ncfg)) { logmsg("SPI command: %s", strerror(errno)); return 1; }
    uint8_t rreg[2 + 23] = {0x20 | 0x01, ncfg - 1}, back[2 + 23];
    if (command(rreg, back, 2 + ncfg)) { logmsg("SPI command: %s", strerror(errno)); return 1; }
    for (int i = 0; i < ncfg; i++) {
        int reg = 1 + i;
        if (reg == 0x12 || reg == 0x13) continue;  /* LOFF_STATP, LOFF_STATN: read-only */
        if (reg == 0x03) {  /* CONFIG3: BIAS_STAT (bit 0) is read-only */
            if ((back[2 + i] & 0xFE) != (cfg[2 + i] & 0xFE)) goto mismatch;
            continue;
        }
        if (reg == 0x14) {  /* GPIO: the input bits (GPIOD) read the pins */
            if ((back[2 + i] & 0x0F) != (cfg[2 + i] & 0x0F)) goto mismatch;
            continue;
        }
        if (back[2 + i] != cfg[2 + i]) {
        mismatch:
            logmsg("register 0x%02x reads 0x%02x after writing 0x%02x: ADS not configured", reg, back[2 + i], cfg[2 + i]);
            return 1;
        }
    }
    if (command1(0x08)) { logmsg("SPI command: %s", strerror(errno)); return 1; }  /* START */
    /* last kernel transfer: sets the controller's 16 MHz dividers, mode and CS timing for the direct reads */
    uint8_t tx[28] = {0x12}, rx[28];
    struct spi_ioc_transfer k;
    memset(&k, 0, sizeof k);
    k.tx_buf = (uintptr_t)tx; k.rx_buf = (uintptr_t)rx; k.len = 28; k.speed_hz = 16000000; k.bits_per_word = 8;
    if (ioctl(spi, SPI_IOC_MESSAGE(1), &k) < 0) { logmsg("SPI read: %s", strerror(errno)); return 1; }

    /* ---- buffers and writer ---- */
    ring = malloc(RING_SIZE * sizeof *ring);
    wbuf = malloc(WBUF_SIZE);
    if (!ring || !wbuf) { logmsg("out of memory"); return 1; }
    memset(ring, 0, RING_SIZE * sizeof *ring);  /* no page faults in the reader */
    if (events_on) {
        fcntl(STDIN_FILENO, F_SETFL, fcntl(STDIN_FILENO, F_GETFL) | O_NONBLOCK);
        if (prefix) {
            char name[4096];
            snprintf(name, sizeof name, "%s_triggers.csv", prefix);
            trig_f = fopen(name, "wx");
            if (!trig_f) { logmsg("%s: %s", name, strerror(errno)); return 1; }
            fputs("trial,code,row,onset_time,offset_us,in_csv\n", trig_f);
        }
    }
    if (decimation > 0) {
        out_dec = STDOUT_FILENO;
        fcntl(out_dec, F_SETFL, fcntl(out_dec, F_GETFL) | O_NONBLOCK);
    }
    pthread_t wt;
    cpu_set_t others;
    CPU_ZERO(&others);
    long nb_cpus = sysconf(_SC_NPROCESSORS_CONF);
    for (int c = 0; c < nb_cpus && c < CPU_SETSIZE; c++) if (c != cpu) CPU_SET(c, &others);
    pthread_attr_t wa;
    pthread_attr_init(&wa);
    pthread_attr_setaffinity_np(&wa, sizeof others, &others);
    if (pthread_create(&wt, &wa, writer, NULL)) { logmsg("writer thread: %s", strerror(errno)); return 1; }

    /* ---- reader: its own CPU, real-time priority ---- */
    cpu_set_t one;
    CPU_ZERO(&one); CPU_SET(cpu, &one);
    if (pthread_setaffinity_np(pthread_self(), sizeof one, &one)) logmsg("CPU %d: %s (continuing)", cpu, strerror(errno));
    struct sched_param sp = { .sched_priority = prio };
    int e = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
    if (e) logmsg("SCHED_FIFO %d: %s (continuing at normal priority)", prio, strerror(e));

    int fd = open("/dev/spimap", O_RDWR | O_SYNC);
    if (fd < 0) { logmsg("/dev/spimap: %s", strerror(errno)); stop = 1; }
    else {
        void *m = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) { logmsg("mmap: %s", strerror(errno)); stop = 1; }
        else regs = m;
    }
    if (regs) {
        signal(SIGSEGV, on_fatal); signal(SIGBUS, on_fatal); signal(SIGILL, on_fatal);
        signal(SIGFPE, on_fatal); signal(SIGABRT, on_fatal);
        cmd0 = R(CMD);
        cmd = cmd0 & ~(CMD_FINISH_IE | CMD_PAUSE_IE | CMD_PAUSE_EN | CMD_ACT | CMD_RESUME);
        R(CMD) = cmd;
        if (R(CMD) & (CMD_FINISH_IE | CMD_PAUSE_IE)) { logmsg("interrupt enables still set: aborting"); R(CMD) = cmd0; stop = 1; }
        (void)R(STATUS0);
        R(CFG1) = (R(CFG1) & ~(CFG1_LEN_MASK | CFG1_LOOP_MASK)) | (27u << 16);
    }
    if (stop) { snprintf(stop_reason, sizeof stop_reason, "setup failed"); failed = 1; }
    else logmsg("capturing at %.0f SPS", rate);

    /* Index of each conversion: conversion k is expected at a + k * p. A read later than LATE_FRACTION of a
     * period after the expected time of the next conversion means that conversions were missed. The reference a
     * follows earlier reads at once and later ones by at most SLEW_FRACTION of a period per conversion (the
     * latency of the reads varies), and p is re-estimated every second (the ADS's clock is not the Coral's). */
    double a = 0, p = period, a_cal = 0, t_prev = 0, t_end = 0;
    uint64_t idx = 0, idx_cal = 0;
    int first = 1;
    while (!stop) {
        double t_wait = 0;
        for (unsigned polls = 1; ; polls++) {
            ioctl(drdy.fd, GPIOHANDLE_GET_LINE_VALUES_IOCTL, &v);
            if (!v.values[0] || stop) break;
            if (polls % 4096 == 0) {  /* the ADS should convert at least 250 times a second */
                double tw = now();
                if (!t_wait) t_wait = tw;
                else if (tw - t_wait > 1.0) {
                    snprintf(stop_reason, sizeof stop_reason, "no conversion (DRDY) for 1 s");
                    logmsg("%s", stop_reason);
                    failed = stop = 1;
                }
            }
        }
        if (stop) break;
        double t = now();
        if (t_end && t >= t_end) { snprintf(stop_reason, sizeof stop_reason, "duration reached"); break; }
        if (!direct_read(rx)) { timeouts++; continue; }
        double dr = now() - t;
        if (dr > max_read) max_read = dr;
        if (first) {
            first = 0;
            a = a_cal = t_first = t;
            if (duration > 0) t_end = t + duration;
        } else {
            double g = t - t_prev;
            if (g > max_gap) max_gap = g;
            if (g > 1.5 * period) gaps++;
            long n = (long)floor((t - a) / p - (double)idx + (1.0 - LATE_FRACTION));
            if (n < 1) n = 1;
            missed += n - 1;
            idx += n;
            double r = t - (a + idx * p);
            a += r < 0 ? r : fmin(r, SLEW_FRACTION * p);
            if (idx - idx_cal >= (uint64_t)rate) {
                double slope = (a - a_cal) / (double)(idx - idx_cal);
                p += slope;
                a -= idx * slope;  /* same reference time for conversion idx */
                a_cal = a; idx_cal = idx;
            }
        }
        t_prev = t_last = t;
        uint64_t h = atomic_load_explicit(&head, memory_order_relaxed);
        if (h - atomic_load_explicit(&tail, memory_order_acquire) >= RING_SIZE) { overflows++; continue; }
        struct frame *f = &ring[h & RING_MASK];
        f->t = a + idx * p;
        f->index = (uint32_t)idx;
        f->bad = (rx[1] & 0xF0) != 0xC0;
        memcpy(f->data, rx + 4, 24);
        atomic_store_explicit(&head, h + 1, memory_order_release);
    }
    period_est = p;

    /* ---- back to the kernel driver ---- */
    if (regs) {
        (void)R(STATUS0);
        R(CMD) = cmd0;
        munmap((void *)regs, 4096);
        regs = NULL;
    }
    if (fd >= 0) close(fd);
    command1(0x0A);  /* STOP */
    atomic_store(&reader_done, 1);
    pthread_join(wt, NULL);
    if (overflows) logmsg("%llu conversions lost: the CSV writer was too slow", (unsigned long long)overflows);
    write_stats();
    logmsg("%s: %llu rows in %d file(s), %llu lost (%llu missed, %llu bad status, %llu buffer overflows), "
           "%llu read timeouts, longest gap %.0f us",
           stop_reason, (unsigned long long)rows, nb_parts, (unsigned long long)nb_lost, (unsigned long long)missed,
           (unsigned long long)bad_frames, (unsigned long long)overflows, (unsigned long long)timeouts, max_gap * 1e6);
    close(drdy.fd); close(start_line.fd); close(chip); close(spi);
    return write_failed || failed ? 3 : 0;
}
