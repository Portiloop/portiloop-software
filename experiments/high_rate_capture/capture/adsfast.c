/* ADS1299 capture with direct register access to the SPI controller (/dev/spimap) and a polled DRDY.
 * Usage: [ADS_PRIO=n] adsfast <seconds> <out_file> <config bytes in hex, from register 0x01>
 * The ADS is configured through spidev, one spidev read at 16 MHz sets the controller's clock and mode, then the
 * controller's interrupts are disabled and each conversion is read directly (about 18 us). The command register
 * is restored at the end. Losses are counted from the data (test signal); gaps between reads are reported. */
#include <fcntl.h>
#include <linux/gpio.h>
#include <linux/spi/spidev.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define CFG1 0x04
#define TX_DATA 0x10
#define RX_DATA 0x14
#define CMD 0x18
#define STATUS0 0x1c
#define CMD_ACT (1u << 0)
#define CMD_RESUME (1u << 1)
#define CMD_RST (1u << 2)
#define CMD_PAUSE_EN (1u << 4)
#define CMD_FINISH_IE (1u << 16)
#define CMD_PAUSE_IE (1u << 17)
#define CFG1_LEN_MASK 0x3ff0000u
#define CFG1_LOOP_MASK 0xff00u

static int spi;
static volatile uint32_t *regs;
static uint32_t cmd0;
static volatile sig_atomic_t stop;
#define R(off) regs[(off) / 4]

static void kxfer(uint8_t *tx, uint8_t *rx, int n, uint32_t hz) {
    struct spi_ioc_transfer t = {0};
    t.tx_buf = (uintptr_t)tx; t.rx_buf = (uintptr_t)rx; t.len = n; t.speed_hz = hz; t.bits_per_word = 8;
    if (ioctl(spi, SPI_IOC_MESSAGE(1), &t) < 0) { perror("spi"); exit(1); }
}
static void command(uint8_t *bytes, int n) {
    struct spi_ioc_transfer t[32] = {0};
    for (int i = 0; i < n; i++) {
        t[i].tx_buf = (uintptr_t)&bytes[i]; t[i].len = 1; t[i].speed_hz = 1000000; t[i].bits_per_word = 8; t[i].delay_usecs = 4;
    }
    if (ioctl(spi, SPI_IOC_MESSAGE(n), t) < 0) { perror("spi command"); exit(1); }
}
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static void on_signal(int s) { (void)s; stop = 1; }

/* as mtk_spi_reset(): set RST, read the register back (which also lengthens the pulse), clear RST */
static void soft_reset(uint32_t cmd) {
    R(CMD) = cmd | CMD_RST;
    (void)R(CMD);
    R(CMD) = cmd;
    (void)R(CMD);
}

static long probe20_n, probe20_low, probe20_high, probe20_st, probe20_st_first;
static double probe20_high_us, probe20_st_us;
static long timeout_log_n;
static uint32_t timeout_log[8][4];  /* STATUS0 seen, CMD, CFG1, attempt */
static long timeout_valid;  /* timeouts where the RX FIFO nevertheless holds a valid frame */
static uint32_t reg20_idle, reg20_timeout[8];

/* one RDATA + 27 NOP frame through the FIFO; returns 0 on timeout (after a soft reset) */
static int direct_read_once(uint32_t cmd, uint8_t *rx, int attempt) {
    for (int i = 0; i < 7; i++) R(TX_DATA) = i == 0 ? 0x12 : 0;
    R(CMD) = cmd | CMD_ACT;
    double t0 = now();
    uint32_t st;
    if (getenv("ADS_PROBE20") && probe20_n < 2000) {  /* register 0x20 over one transfer, against STATUS0 */
        double t_low = -1, t_high = -1, t_st = -1;
        while (now() - t0 < 60e-6) {
            uint32_t r = regs[0x20 / 4]; double t = now() - t0;
            if (!(r & 1) && t_low < 0) t_low = t;
            if ((r & 1) && t_low >= 0 && t_high < 0) t_high = t;
            if (t_st < 0 && (R(STATUS0) & 1)) t_st = t;
            if (t_high >= 0 && t_st >= 0) break;
        }
        probe20_n++;
        if (t_low >= 0) probe20_low++;
        if (t_high >= 0) { probe20_high++; probe20_high_us += t_high * 1e6; }
        if (t_st >= 0) { probe20_st++; probe20_st_us += t_st * 1e6; }
        if (t_high >= 0 && t_st >= 0 && t_st < t_high) probe20_st_first++;
        st = 1;  /* finished (or given up after 60 us): read the FIFO */
    } else {
        /* Completion: register 0x20 bit 0 is 1 when idle, 0 during the transfer (polling the read-to-clear
         * STATUS0 instead can miss its finish flag). Wait for busy, then idle again. */
        while (regs[0x20 / 4] & 1)
            if (now() - t0 > 10e-6) break;  /* not started yet: fall through to the STATUS0 check below */
        while (!(regs[0x20 / 4] & 1))
            if (now() - t0 > 60e-6) break;
        (void)R(STATUS0);  /* clear the finish flag */
        st = (regs[0x20 / 4] & 1) ? 1 : 0;
    }
    while (!(st & 1))
        if (now() - t0 > 50e-6) {
            if (timeout_log_n < 8) {
                uint32_t *l = timeout_log[timeout_log_n];
                l[0] = st; l[1] = R(CMD); l[2] = R(CFG1); l[3] = attempt;
                reg20_timeout[timeout_log_n] = regs[0x20 / 4];
            }
            uint32_t w0 = R(RX_DATA);  /* first word: the byte clocked with RDATA, then the 3 status bytes */
            if (((w0 >> 8) & 0xF0) == 0xC0) timeout_valid++;
            timeout_log_n++;
            soft_reset(cmd);
            return 0;
        }
    uint32_t w[7];
    for (int i = 0; i < 7; i++) w[i] = R(RX_DATA);
    memcpy(rx, w, 28);
    soft_reset(cmd);  /* soft reset after each packet, as mtk_spi_set_cs() does */
    return 1;
}

static long retried_ok;
static int direct_read(uint32_t cmd, uint8_t *rx) {
    if (direct_read_once(cmd, rx, 0)) return 1;
    if (direct_read_once(cmd, rx, 1)) { retried_ok++; return 1; }  /* the conversion is still there */
    return 0;
}

int main(int argc, char **argv) {
    int ncfg = argc - 3;
    if (ncfg < 1 || ncfg > 23) { fprintf(stderr, "usage: adsfast seconds out_file config bytes\n"); return 2; }
    double seconds = atof(argv[1]);
    uint8_t cfg[2 + 23]; cfg[0] = 0x40 | 0x01; cfg[1] = ncfg - 1;
    for (int i = 0; i < ncfg; i++) cfg[2 + i] = strtol(argv[3 + i], NULL, 16);
    double rate = 16000.0 / (1 << (cfg[2] & 0x07)), period = 1.0 / rate;

    int prio = getenv("ADS_PRIO") ? atoi(getenv("ADS_PRIO")) : 10;
    struct sched_param sp = { .sched_priority = prio };
    if (sched_setscheduler(0, SCHED_FIFO, &sp)) perror("SCHED_FIFO (continuing at normal priority)");
    signal(SIGINT, on_signal); signal(SIGTERM, on_signal);

    spi = open("/dev/spidev0.0", O_RDWR); if (spi < 0) { perror("spidev"); return 1; }
    uint8_t mode = SPI_MODE_1; ioctl(spi, SPI_IOC_WR_MODE, &mode);
    uint32_t base = 1000000; ioctl(spi, SPI_IOC_WR_MAX_SPEED_HZ, &base);
    int chip = open("/dev/gpiochip0", O_RDWR); if (chip < 0) { perror("gpiochip0"); return 1; }
    struct gpiohandle_request start = {0};
    start.lineoffsets[0] = 37; start.lines = 1; start.flags = GPIOHANDLE_REQUEST_OUTPUT; strcpy(start.consumer_label, "adsfast");
    ioctl(chip, GPIO_GET_LINEHANDLE_IOCTL, &start);
    struct gpiohandle_request drdy = {0};
    drdy.lineoffsets[0] = 45; drdy.lines = 1; drdy.flags = GPIOHANDLE_REQUEST_INPUT; strcpy(drdy.consumer_label, "adsfast-drdy");
    if (ioctl(chip, GPIO_GET_LINEHANDLE_IOCTL, &drdy) < 0) { perror("DRDY line"); return 1; }
    struct gpiohandle_data v;

    uint8_t b = 0x0A; command(&b, 1); b = 0x11; command(&b, 1); command(cfg, 2 + ncfg);
    b = 0x08; command(&b, 1);  /* START: every kernel transfer must happen before the direct reads take over */
    uint8_t tx[28] = {0x12}, rx[28];
    kxfer(tx, rx, 28, 16000000);  /* last kernel transfer: sets the controller's 16 MHz dividers, mode and CS timing */

    int fd = open("/dev/spimap", O_RDWR | O_SYNC); if (fd < 0) { perror("/dev/spimap"); return 1; }
    regs = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (regs == MAP_FAILED) { perror("mmap"); return 1; }
    cmd0 = R(CMD);
    uint32_t cmd = cmd0 & ~(CMD_FINISH_IE | CMD_PAUSE_IE | CMD_PAUSE_EN | CMD_ACT | CMD_RESUME);
    R(CMD) = cmd;
    if (R(CMD) & (CMD_FINISH_IE | CMD_PAUSE_IE)) { fprintf(stderr, "interrupt enables still set: aborting\n"); R(CMD) = cmd0; return 1; }
    (void)R(STATUS0);
    reg20_idle = regs[0x20 / 4];
    R(CFG1) = (R(CFG1) & ~(CFG1_LEN_MASK | CFG1_LOOP_MASK)) | (27u << 16);

    FILE *out = fopen(argv[2], "wb"); if (!out) { perror("out"); R(CMD) = cmd0; return 1; }
    static char obuf[4096]; setvbuf(out, obuf, _IOFBF, sizeof obuf);

    long frames = 0, gaps = 0, timeouts = 0, bad = 0;
    double t_end = now() + seconds, t_prev = 0, max_gap = 0, max_read = 0, t0 = now();
    while (!stop) {
        do ioctl(drdy.fd, GPIOHANDLE_GET_LINE_VALUES_IOCTL, &v); while (v.values[0] && !stop);
        double t = now();
        if (t >= t_end) break;
        if (!direct_read(cmd, rx)) { timeouts++; continue; }
        double dr = now() - t; if (dr > max_read) max_read = dr;
        if ((rx[1] & 0xF0) != 0xC0) bad++;
        fwrite(rx + 4, 1, 24, out);
        if (t_prev) { double g = t - t_prev; if (g > max_gap) max_gap = g; if (g > 1.5 * period) gaps++; }
        t_prev = t; frames++;
    }
    fclose(out);
    (void)R(STATUS0);
    R(CMD) = cmd0;  /* back to the kernel driver */
    munmap((void *)regs, 4096); close(fd);
    b = 0x0A; command(&b, 1);  /* STOP, through the kernel again */
    for (long i = 0; i < timeout_log_n && i < 8; i++)
        printf("   timeout %ld: STATUS0 %x, CMD %08x, CFG1 %08x, attempt %u\n", i + 1, timeout_log[i][0], timeout_log[i][1], timeout_log[i][2], timeout_log[i][3]);
    printf("   %ld first attempts timed out, %ld recovered by the retry; %ld had a valid frame in the RX FIFO anyway\n",
           timeout_log_n, retried_ok, timeout_valid);
    printf("   register 0x20 over %ld transfers: went to 0 in %ld, back to 1 in %ld (after %.1f us on average); "
           "STATUS0 seen done in %ld (after %.1f us), before 0x20 returned to 1 in %ld\n",
           probe20_n, probe20_low, probe20_high, probe20_high ? probe20_high_us / probe20_high : 0,
           probe20_st, probe20_st ? probe20_st_us / probe20_st : 0, probe20_st_first);
    printf("   register 0x20: %08x when idle; at the first timeouts:", reg20_idle);
    for (long i = 0; i < timeout_log_n && i < 8; i++) printf(" %08x", reg20_timeout[i]);
    printf("\n");
    printf("%5.0f SPS direct: %.1f s, %ld frames, gaps > 1.5 periods %ld, longest gap %.0f us, longest read %.0f us, "
           "read timeouts %ld, bad status %ld\n", rate, now() - t0, frames, gaps, max_gap * 1e6, max_read * 1e6, timeouts, bad);
    return 0;
}
