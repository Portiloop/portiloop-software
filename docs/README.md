# Portiloop tutorials (GitHub Pages site)

Interactive tutorials on running a closed-loop auditory stimulation night with the Portiloop and processing the recordings. Static HTML, CSS and JavaScript: no build step.

## Preview locally

The pages load data files, so they must be served over HTTP (opening them from disk does not work):

```bash
python3 -m http.server 8000 -d docs
# then open http://localhost:8000
```

## Publish on GitHub Pages

Settings → Pages → Source: Deploy from a branch → the branch that holds `docs/`, folder `/docs`. The site is then at `https://portiloop.github.io/portiloop-software/`.

## Layout

```
docs/
├── index.html                  home page (cards are generated from the table of contents in nav.js)
├── tutorials/NN-name.html      one page per tutorial; _template.html shows every component
├── assets/css/style.css        shared styles (light and dark themes), same design as the training tutorials
├── assets/js/nav.js            top bar, sidebar, prev/next, footer; the table of contents lives here
├── assets/js/portiloop.js      shared helpers: plotting (PL.plot), UI, filters
├── assets/js/data.js           loader of the tutorial night (PL.data.night/excerpts/epochs)
├── assets/data/                summaries of the tutorial night for the widgets (built by tools/build_assets.py)
├── assets/img/                 logo and icons; protocol/ holds the protocol illustration and screenshots
└── tools/                      scripts that make the dataset and the data files
```

## The tutorial night

The processing tutorials use one real night recorded with the lab's CLAS protocol. `tools/make_tutorial_dataset.py` turns the original recording into the public dataset (new file names, metadata without the device path and recording name, values written with 3 decimals) and a zip archive; `tools/stage_night.py` (if present) stages it automatically with the Portiloop's NanoSleepNet model; `tools/build_assets.py` writes the summaries in `assets/data/` that the widgets load:

```bash
python docs/tools/make_tutorial_dataset.py ORIGINAL.csv OUT --name clas-tutorial --stages STAGES.csv
python docs/tools/build_assets.py OUT/clas-tutorial_<date>_<time>_CONDB
```

Before publishing a recording, make sure the participant consented to the public sharing of their data and that nothing in the files (names, dates, notes) identifies them.

## Adding a tutorial

1. Copy `tutorials/_template.html` to `tutorials/NN-name.html`, set `data-page="NN-name"` on `<body>`.
2. Add `["NN-name", "Title"]` to `TOC` in `assets/js/nav.js` and a blurb in `index.html`.
3. Widgets: a `div.widget` with a canvas drawn with `PL.plot`, redrawn on resize and on the `pl-theme-change` event.

## Logo

The logo files in `assets/img/` are vector tracings of `figures/portiloop-logo.png` from the [portiloop-hardware](https://github.com/Portiloop/portiloop-hardware) repository (see the training tutorials' `docs/README.md`).
