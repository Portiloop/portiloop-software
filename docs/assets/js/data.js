/*
 * data.js: loads the tutorial night (assets/data/*.json, made by docs/tools/build_assets.py) for the widgets.
 * Load after portiloop.js. Every function returns a promise; each file is fetched once and cached.
 *
 *   PL.data.night()     -> night.json, with spectrogram.db = Float32Array (n_epochs * n_freqs, dB) and spectrogram.nFreqs
 *                          columns: { stage, stims, sound, cz_rail, cz_delta, cz_theta, cz_alpha, cz_sigma, cz_beta,
 *                                     cz_ptp, eog1_ptp, emg_ptp, emg_beta, eog_slow } (one value per 30 s epoch)
 *   PL.data.excerpts()  -> excerpts.json, each excerpt with signals = [8 Float32Array in uV, NaN = missing]
 *   PL.data.epochs()    -> epochs.json, with time (Float32Array, s), slow and sigma (Float32Array, uV, NaN = missing,
 *                          n_events * n_times) and row(kind, i) returning one event's epoch ("slow" or "sigma")
 *
 * Pages under tutorials/ load the files from "../assets/data/"; set PL.data.base to change it.
 */
(function () {
  "use strict";
  const PL = window.PL || (window.PL = {});
  const cache = {};

  function bytes(b64) {
    const s = atob(b64);
    const out = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
    return out;
  }

  /** base64 little-endian int16 in units of 1/scale -> Float32Array, -32768 -> NaN. */
  function int16(b64, scale) {
    const u8 = bytes(b64);
    const view = new DataView(u8.buffer);
    const n = u8.length / 2;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const v = view.getInt16(2 * i, true);
      out[i] = v === -32768 ? NaN : v / scale;
    }
    return out;
  }

  function fetchJSON(name) {
    const base = PL.data.base || (document.body.dataset.root ? document.body.dataset.root + "/assets/data/" : "../assets/data/");
    if (!cache[name]) {
      cache[name] = fetch(base + name).then((r) => {
        if (!r.ok) throw new Error("Cannot load " + base + name + " (" + r.status + ")");
        return r.json();
      });
    }
    return cache[name];
  }

  PL.data = {
    base: null,
    decodeInt16: int16,

    night: function () {
      if (!cache._night) {
        cache._night = fetchJSON("night.json").then((j) => {
          const sp = j.spectrogram;
          const u8 = bytes(sp.data);
          const db = new Float32Array(u8.length);
          for (let i = 0; i < u8.length; i++) db[i] = sp.db_min + (u8[i] / 255) * (sp.db_max - sp.db_min);
          sp.db = db;
          sp.nFreqs = sp.freqs_hz.length;
          return j;
        });
      }
      return cache._night;
    },

    excerpts: function () {
      if (!cache._excerpts) {
        cache._excerpts = fetchJSON("excerpts.json").then((j) => {
          const n = j.fs * j.seconds;
          j.excerpts.forEach((ex) => {
            const all = int16(ex.data, 10);
            ex.signals = j.channels.map((_, c) => all.subarray(c * n, (c + 1) * n));
          });
          return j;
        });
      }
      return cache._excerpts;
    },

    epochs: function () {
      if (!cache._epochs) {
        cache._epochs = fetchJSON("epochs.json").then((j) => {
          j.slow = int16(j.slow, 10);
          j.sigma = int16(j.sigma, 10);
          j.time = new Float32Array(j.n_times);
          for (let k = 0; k < j.n_times; k++) j.time[k] = j.t0_s + k / j.fs;
          j.row = (kind, i) => j[kind].subarray(i * j.n_times, (i + 1) * j.n_times);
          return j;
        });
      }
      return cache._epochs;
    },
  };
})();
