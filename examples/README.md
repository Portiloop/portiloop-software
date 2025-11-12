# Portiloop Python library (examples)

This folder contains:
- `custom_pipeline_example.py`: an example of custom filtering/detection/stimulation pipeline. In this example, we customize the stimulator used in the default closed-loop sleep spindles stimulation pipeline so that, instead of always playing the same sound file, the stimulator samples a random sound file from a set.
- `Jupyter_UI.ipynb`: this jupyter notebook enables using both the default pipelines and the custom pipeline described above.
- `custom_ui_example.py`: this file launches a web GUI server on port `8083`. This GUI is a modded version of the default "Simple GUI". It reads the content of `sound_sets.json`, which contains mappings from keys to sets of files from the `sounds` folder. The GUI allows the user to select a set of sounds from all the sets defined in `sound_sets.json`. It then feeds this set to the custom stimulator described above.