/* Probe of direct register access to the MediaTek SPI controller (through /dev/spimap), with the ADS1299.
 * Compares a few frames read directly with frames read through spidev, and checks that no SPI interrupt fires. */
#include <fcntl.h>
#include <linux/gpio.h>
#include <linux/spi/spidev.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define CFG0 0x00
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
static long spi_irqs(void) {
    FILE *f = fopen("/proc/interrupts", "r"); char line[512]; long total = -1;
    while (f && fgets(line, sizeof line, f)) {
        if (strstr(line, "1100c000.spi")) { long a, b, c, d; if (sscanf(line, " %*d: %ld %ld %ld %ld", &a, &b, &c, &d) == 4) total = a + b + c + d; }
    }
    if (f) fclose(f);
    return total;
}
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static void print_frame(const char *label, uint8_t *rx) {
    printf("%s: status %02x%02x%02x | ch1 %02x%02x%02x ch2 %02x%02x%02x ... ch8 %02x%02x%02x\n", label,
           rx[1], rx[2], rx[3], rx[4], rx[5], rx[6], rx[7], rx[8], rx[9], rx[25], rx[26], rx[27]);
}

int main(void) {
    spi = open("/dev/spidev0.0", O_RDWR); if (spi < 0) { perror("spidev"); return 1; }
    uint8_t mode = SPI_MODE_1; ioctl(spi, SPI_IOC_WR_MODE, &mode);
    uint32_t base = 1000000; ioctl(spi, SPI_IOC_WR_MAX_SPEED_HZ, &base);
    int chip = open("/dev/gpiochip0", O_RDWR);
    struct gpiohandle_request start = {0};
    start.lineoffsets[0] = 37; start.lines = 1; start.flags = GPIOHANDLE_REQUEST_OUTPUT; strcpy(start.consumer_label, "spiprobe");
    ioctl(chip, GPIO_GET_LINEHANDLE_IOCTL, &start);

    /* ADS: 250 SPS, internal test signal on all channels, conversions started */
    uint8_t cfg[] = {0x40 | 0x01, 20, 0x96, 0xd0, 0xfc, 0x00, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65,
                     0, 0, 0, 0, 0, 0, 0, 0, 0x20};
    uint8_t b = 0x0A; command(&b, 1); b = 0x11; command(&b, 1); command(cfg, sizeof cfg); b = 0x08; command(&b, 1);
    usleep(100000);

    uint8_t tx[28] = {0x12}, rx[28];
    kxfer(tx, rx, 28, 16000000);  /* sets the 16 MHz clock dividers, the mode and CS timing in the controller */
    print_frame("kernel read    ", rx);

    int fd = open("/dev/spimap", O_RDWR | O_SYNC); if (fd < 0) { perror("/dev/spimap"); return 1; }
    regs = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (regs == MAP_FAILED) { perror("mmap"); return 1; }
    uint32_t cmd0 = R(CMD), cfg1 = R(CFG1);
    printf("registers: CFG0 %08x CFG1 %08x CMD %08x STATUS0 %08x\n", R(CFG0), cfg1, cmd0, R(STATUS0));

    long irq0 = spi_irqs();
    uint32_t cmd = cmd0 & ~(CMD_FINISH_IE | CMD_PAUSE_IE | CMD_PAUSE_EN | CMD_ACT | CMD_RESUME);
    R(CMD) = cmd;
    uint32_t back = R(CMD);
    printf("CMD with interrupts disabled: wrote %08x, read back %08x\n", cmd, back);
    if (back & (CMD_FINISH_IE | CMD_PAUSE_IE)) { printf("interrupt enables still set: aborting\n"); R(CMD) = cmd0; return 1; }
    (void)R(STATUS0);  /* clear a stale status */

    R(CFG1) = (cfg1 & ~(CFG1_LEN_MASK | CFG1_LOOP_MASK)) | (27u << 16);  /* one packet of 28 bytes */
    for (int k = 0; k < 5; k++) {
        usleep(4000);  /* the next conversion (250 SPS) */
        for (int i = 0; i < 7; i++) R(TX_DATA) = i == 0 ? 0x12 : 0;  /* RDATA then 27 NOPs, little endian */
        double t0 = now();
        R(CMD) = cmd | CMD_ACT;
        uint32_t st, seen[8]; int nseen = 0; long polls = 0;
        do {
            st = R(STATUS0); polls++;
            if (nseen < 8 && (nseen == 0 || st != seen[nseen - 1])) seen[nseen++] = st;
        } while (!(st & 1) && now() - t0 < 1e-3);
        double dt = now() - t0;
        uint32_t w[7];
        for (int i = 0; i < 7; i++) w[i] = R(RX_DATA);
        memcpy(rx, w, 28);
        R(CMD) = cmd | CMD_RST; R(CMD) = cmd;  /* soft reset after each packet, as mtk_spi_set_cs() does */
        printf("direct read %d: %.1f us, %ld status polls, STATUS0 values:", k + 1, dt * 1e6, polls);
        for (int i = 0; i < nseen; i++) printf(" %x", seen[i]);
        printf(", CMD now %08x\n", R(CMD));
        print_frame("               ", rx);
    }
    (void)R(STATUS0);
    long irq1 = spi_irqs();
    R(CMD) = cmd0;  /* the kernel driver rewrites CMD at its next message anyway */
    printf("SPI interrupts during the direct reads: %ld\n", irq1 - irq0);
    munmap((void *)regs, 4096); close(fd);

    kxfer(tx, rx, 28, 16000000);
    print_frame("kernel read after", rx);
    b = 0x0A; command(&b, 1);
    return 0;
}
