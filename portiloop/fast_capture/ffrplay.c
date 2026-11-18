/* Sound of an FFR run (portiloop/src/core/ffr.py starts it), and the time of each stimulus onset for adscapture -e.
 *
 * Usage: ffrplay [options] <audio> <schedule>
 *   audio       the whole run, S16_LE interleaved frames at the rate of -r (silences included)
 *   schedule    lines "<trial> <code> <frame of the onset in the audio>", in increasing frames (may be empty)
 *   -D device   ALSA PCM (default "softvol", as the stimulators: the GUI's volume applies)
 *   -r rate     frames per second (default 48000: the rate of dmix, so that ALSA does not resample)
 *   -c n        channels (default 2)
 *   -l us       ALSA latency (default 100000)
 *   -s path     statistics (JSON), written at the end
 *   -p prio     SCHED_FIFO priority (default 5, below adscapture's reader), -u cpu: a CPU not to use (default 3,
 *               adscapture's reader)
 *
 * stdout: "<trial> <code> <time>\n" for each onset, <time> in s of CLOCK_MONOTONIC, sent when the onset is written
 * to ALSA, about the latency before it is played.
 *
 * Time of a frame: after each write, the frame being played (frames written - snd_pcm_delay()) is read with the time
 * of the query. The frames played and the times follow a straight line (the DAC's clock), fitted on the last
 * FIT_POINTS queries (about 2 s); the time of an onset is that line at its frame. On the Portiloop, the playback
 * position moves by 4 frames (83 us at 48 kHz) and lies within about 1 frame of the line, so the fitted times are
 * within a few us. This is the time at which the audio DMA reads the frame: the latency of the codec, amplifier and
 * earphones comes on top of it (constant).
 *
 * After an underrun the line is fitted again; the onsets written before enough queries are sent late (adscapture
 * lists them, without writing them in the CSV). Stops at the end of the audio (after it is played), or on SIGTERM. */
#define _GNU_SOURCE
#include <alsa/asoundlib.h>
#include <errno.h>
#include <math.h>
#include <sched.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>

#define CHUNK 240            /* frames per write: 5 ms at 48 kHz */
#define FIT_POINTS 400       /* queries in the fit: 2 s */
#define MIN_FIT_POINTS 20    /* before sending onsets: 0.1 s after the start of the stream */
#define MAX_QUERY 50e-6      /* queries longer than this (preempted) are not used */
#define MAX_XRUNS_LISTED 1000
#define OPEN_TIMEOUT 10.0    /* s, while the device is busy */

static volatile sig_atomic_t stop;
static void on_stop(int s) { (void)s; stop = 1; }
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

static void logmsg(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void logmsg(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fputs("ffrplay: ", stderr); vfprintf(stderr, fmt, ap); fputc('\n', stderr);
    va_end(ap);
}

/* fit of the frames played against the time, on the last FIT_POINTS queries */
static double fit_t[FIT_POINTS], fit_f[FIT_POINTS], t_base;
static int fit_n, fit_i;

static void fit_add(double t, double f) {
    if (fit_n == 0 && fit_i == 0) t_base = t;
    fit_t[fit_i] = t - t_base; fit_f[fit_i] = f;
    fit_i = (fit_i + 1) % FIT_POINTS;
    if (fit_n < FIT_POINTS) fit_n++;
}
static void fit_reset(void) { fit_n = fit_i = 0; }

/* frame = slope * (t - t_base - tm) + fm; returns the RMS residual (s) */
static double fit(double *slope, double *tm, double *fm) {
    double st = 0, sf = 0;
    for (int i = 0; i < fit_n; i++) { st += fit_t[i]; sf += fit_f[i]; }
    *tm = st / fit_n; *fm = sf / fit_n;
    double stt = 0, stf = 0;
    for (int i = 0; i < fit_n; i++) {
        double dt = fit_t[i] - *tm;
        stt += dt * dt; stf += dt * (fit_f[i] - *fm);
    }
    *slope = stt > 0 ? stf / stt : 0;
    double ss = 0;
    for (int i = 0; i < fit_n; i++) {
        double r = fit_f[i] - (*slope * (fit_t[i] - *tm) + *fm);
        ss += r * r;
    }
    return *slope > 0 ? sqrt(ss / fit_n) / *slope : 0;
}

struct onset { unsigned trial; int code; uint64_t frame; };

static void quiet(const char *file, int line, const char *function, int err, const char *fmt, ...) {
    (void)file; (void)line; (void)function; (void)err; (void)fmt;
}

static void json_str(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') fputc('\\', f);
        if ((unsigned char)*s >= 0x20) fputc(*s, f);
    }
    fputc('"', f);
}

int main(int argc, char **argv) {
    const char *device = "softvol", *stats_path = NULL;
    unsigned rate = 48000, latency = 100000;
    int channels = 2, prio = 5, cpu_excluded = 3, opt;
    while ((opt = getopt(argc, argv, "D:r:c:l:s:p:u:")) != -1) {
        switch (opt) {
        case 'D': device = optarg; break;
        case 'r': rate = atoi(optarg); break;
        case 'c': channels = atoi(optarg); break;
        case 'l': latency = atoi(optarg); break;
        case 's': stats_path = optarg; break;
        case 'p': prio = atoi(optarg); break;
        case 'u': cpu_excluded = atoi(optarg); break;
        default: goto usage;
        }
    }
    if (argc - optind != 2 || rate < 8000 || channels < 1 || channels > 8) {
    usage:
        fprintf(stderr, "usage: ffrplay [-D device] [-r rate] [-c channels] [-l latency_us] [-s stats.json] "
                        "[-p prio] [-u cpu] audio.raw schedule.txt\n");
        return 2;
    }
    prctl(PR_SET_PDEATHSIG, SIGTERM);  /* stop if the Python process dies */
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_stop;  /* no SA_RESTART: interrupts the blocking writes */
    sigaction(SIGINT, &sa, NULL); sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);  /* adscapture has stopped: play to the end anyway */

    /* ---- audio and schedule ---- */
    FILE *fa = fopen(argv[optind], "rb");
    if (!fa) { logmsg("%s: %s", argv[optind], strerror(errno)); return 1; }
    fseek(fa, 0, SEEK_END);
    long bytes = ftell(fa);
    fseek(fa, 0, SEEK_SET);
    size_t frame_bytes = 2 * channels;
    uint64_t nframes = bytes / frame_bytes;
    int16_t *audio = malloc(nframes * frame_bytes + 1);
    if (!audio || fread(audio, frame_bytes, nframes, fa) != nframes) { logmsg("cannot read the audio"); return 1; }
    fclose(fa);

    FILE *fs = fopen(argv[optind + 1], "r");
    if (!fs) { logmsg("%s: %s", argv[optind + 1], strerror(errno)); return 1; }
    size_t nons = 0, cap = 1024;
    struct onset *ons = malloc(cap * sizeof *ons);
    unsigned trial; int code; unsigned long long fr;
    while (ons && fscanf(fs, "%u %d %llu", &trial, &code, &fr) == 3) {
        if (fr >= nframes || (nons && fr < ons[nons - 1].frame)) { logmsg("onset frame %llu out of order", fr); return 1; }
        if (nons == cap) ons = realloc(ons, (cap *= 2) * sizeof *ons);
        if (ons) ons[nons++] = (struct onset){ trial, code, fr };
    }
    if (!ons) { logmsg("out of memory"); return 1; }
    fclose(fs);

    /* ---- real time: SCHED_FIFO, off adscapture's CPU ----
     * (no mlockall: the Portiloop has no swap, so the audio, read above, stays in memory; with MCL_FUTURE, ALSA's
     * dmix buffers would exceed the memlock limit) */
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    long nb_cpus = sysconf(_SC_NPROCESSORS_CONF);
    for (int c = 0; c < nb_cpus && c < CPU_SETSIZE; c++) if (c != cpu_excluded) CPU_SET(c, &cpus);
    if (sched_setaffinity(0, sizeof cpus, &cpus)) logmsg("CPU affinity: %s (continuing)", strerror(errno));
    struct sched_param sp = { .sched_priority = prio };
    if (prio > 0 && sched_setscheduler(0, SCHED_FIFO, &sp)) logmsg("SCHED_FIFO %d: %s (continuing)", prio, strerror(errno));

    /* ---- ALSA ---- */
    /* PulseAudio keeps the card for about 5 s after a sound played through it ("default", as the aplay at the import
     * of portiloop.src), and dmix cannot open it meanwhile: wait for it (the silence before the first onset is longer) */
    snd_pcm_t *pcm;
    int e;
    double t_open = now();
    snd_lib_error_set_handler(quiet);  /* dmix prints an error at each attempt */
    while ((e = snd_pcm_open(&pcm, device, SND_PCM_STREAM_PLAYBACK, 0)) == -EBUSY && !stop && now() - t_open < OPEN_TIMEOUT) {
        if (now() - t_open < 0.2) logmsg("%s is busy (PulseAudio?): waiting up to %.0f s", device, OPEN_TIMEOUT);
        usleep(200000);
    }
    snd_lib_error_set_handler(NULL);
    if (e < 0) { logmsg("%s: %s", device, snd_strerror(e)); return 1; }
    if (now() - t_open > 0.1) logmsg("%s opened after %.1f s", device, now() - t_open);
    /* no resampling by ALSA: the audio is at the rate of the device */
    e = snd_pcm_set_params(pcm, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED, channels, rate, 0, latency);
    if (e < 0) { logmsg("%s at %u Hz, %d channels: %s", device, rate, channels, snd_strerror(e)); return 1; }
    snd_pcm_uframes_t buffer_size, period_size;
    snd_pcm_get_params(pcm, &buffer_size, &period_size);
    logmsg("playing %llu frames (%.1f s) with %zu onsets on %s (buffer %lu, period %lu frames)",
           (unsigned long long)nframes, (double)nframes / rate, nons, device, buffer_size, period_size);

    uint64_t written = 0, xruns = 0, sent = 0, sent_late = 0, slow_queries = 0, queries = 0;
    uint64_t xrun_frames[MAX_XRUNS_LISTED];
    double max_resid = 0, sum_resid = 0, slope_last = 0, max_query = 0;
    size_t next = 0;  /* next onset to send */
    char stop_reason[64] = "end of the audio";
    while (written < nframes) {
        if (stop) { snprintf(stop_reason, sizeof stop_reason, "stopped (SIGTERM)"); break; }
        uint64_t n = nframes - written < CHUNK ? nframes - written : CHUNK;
        /* the onsets in this chunk (and those not sent yet, after an underrun): once the line is fitted */
        while (next < nons && ons[next].frame < written + n && fit_n >= MIN_FIT_POINTS) {
            double slope, tm, fm;
            double resid = fit(&slope, &tm, &fm);
            double t = t_base + tm + (ons[next].frame - fm) / slope;
            if (resid > max_resid) max_resid = resid;
            sum_resid += resid;
            slope_last = slope;
            if (t < now()) sent_late++;
            printf("%u %d %.9f\n", ons[next].trial, ons[next].code, t);
            fflush(stdout);
            sent++; next++;
        }
        snd_pcm_sframes_t w = snd_pcm_writei(pcm, audio + written * channels, n);
        if (w < 0) {
            if (w == -EINTR) continue;
            if (w == -EPIPE) {
                if (xruns < MAX_XRUNS_LISTED) xrun_frames[xruns] = written;
                xruns++;
                logmsg("underrun at frame %llu", (unsigned long long)written);
                fit_reset();
            }
            if ((e = snd_pcm_recover(pcm, w, 1)) < 0) {
                snprintf(stop_reason, sizeof stop_reason, "ALSA error: %s", snd_strerror(e));
                logmsg("%s", stop_reason);
                break;
            }
            continue;
        }
        written += w;
        if (snd_pcm_state(pcm) == SND_PCM_STATE_RUNNING) {
            snd_pcm_sframes_t delay;
            double a = now();
            e = snd_pcm_delay(pcm, &delay);
            double b = now();
            queries++;
            if (b - a > max_query) max_query = b - a;
            if (e == 0 && b - a <= MAX_QUERY) fit_add(0.5 * (a + b), (double)written - delay);
            else slow_queries++;
        }
    }
    if (stop || written < nframes) snd_pcm_drop(pcm);
    else snd_pcm_drain(pcm);
    snd_pcm_close(pcm);
    if (next < nons) logmsg("%zu onsets not played", nons - next);
    logmsg("%s: %llu onsets sent (%llu late), %llu underruns, DAC at %.4f Hz, position scatter up to %.1f us RMS",
           stop_reason, (unsigned long long)sent, (unsigned long long)sent_late, (unsigned long long)xruns, slope_last,
           max_resid * 1e6);

    if (stats_path) {
        FILE *f = fopen(stats_path, "w");
        if (!f) { logmsg("cannot write %s: %s", stats_path, strerror(errno)); return 1; }
        fprintf(f, "{\n    \"stop_reason\": "); json_str(f, stop_reason);
        fprintf(f, ",\n    \"device\": "); json_str(f, device);
        fprintf(f, ",\n    \"rate\": %u,\n    \"channels\": %d,\n    \"buffer_frames\": %lu,\n    \"period_frames\": %lu,\n",
                rate, channels, buffer_size, period_size);
        fprintf(f, "    \"frames\": %llu,\n    \"frames_written\": %llu,\n", (unsigned long long)nframes,
                (unsigned long long)written);
        fprintf(f, "    \"onsets\": %zu,\n    \"onsets_sent\": %llu,\n    \"onsets_sent_late\": %llu,\n", nons,
                (unsigned long long)sent, (unsigned long long)sent_late);
        fprintf(f, "    \"underruns\": %llu,\n    \"underrun_frames\": [", (unsigned long long)xruns);
        for (uint64_t i = 0; i < xruns && i < MAX_XRUNS_LISTED; i++)
            fprintf(f, "%s%llu", i ? ", " : "", (unsigned long long)xrun_frames[i]);
        fprintf(f, "],\n    \"dac_rate_hz\": %.6f,\n", slope_last);
        /* scatter of the playback positions around the fitted line (not the error of the onset times: the line averages
         * FIT_POINTS of them) */
        fprintf(f, "    \"position_scatter_us_rms_mean\": %.3f,\n    \"position_scatter_us_rms_max\": %.3f,\n",
                sent ? sum_resid / sent * 1e6 : 0, max_resid * 1e6);
        fprintf(f, "    \"position_queries\": %llu,\n    \"position_queries_too_slow\": %llu,\n"
                   "    \"longest_query_us\": %.1f\n}\n",
                (unsigned long long)queries, (unsigned long long)slow_queries, max_query * 1e6);
        fclose(f);
    }
    return written < nframes && !stop ? 3 : 0;
}
