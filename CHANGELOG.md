# Changelog

Versions stay 0.x until the first official release (1.0).

## Unreleased

- Download: the radius search is now the only method. `QQ_dl_stations.py` and `QQ_dl_data.py` (box search with fixed networks and channels) were removed, and so were the driver options `station_search`, `networks` and `channels`. The default data center is `EARTHSCOPE_FEDERATOR`.
- New driver option `station_excluded_stations`: stations to leave out, as `"NET.STA"`.
- `examples/pavlof_2021/`: radius search within 0.6 degrees of the volcano (the AV network is within 0.51 degrees).
- README: HypoInverse is compiled from HypoXPy (`dependencies/HypoXPy/external/hypoinverse1.40`), whose makefile has the compiler options this old Fortran code needs; added the macOS fix for `ld: library 'System' not found`.
- Removed `dependencies/hyp1.40`: a second copy of the HypoInverse source with a makefile without those options (on the Pavlof example its HypoInverse catalog had 56 instead of 57 good locations), plus 47 compiled `.o` files.
- Added `environment_mac.yml` (conda environment used on macOS).

## v0.5 (2026-10-15)

With the default settings, results are the same as the previous development code
(commit b9f0e9c, "v4.6"): on the Pavlof example (2021-07-29) the GaMMA catalog and
picks, the HypoInverse and HypoDD catalogs and the QC catalogs are byte-identical
to the ones from those scripts.

### Driver (`quickquake/QQ_driver.py`)
- Same layout: settings at the top (each `RUN_...` switch followed by the options of that step), then the code that runs each step.
- New header with instructions: copy the driver into a project folder, set `QUICKQUAKE_DIR` (new, at the top), edit the settings, run it. Results go to a `data/` folder next to the driver.
- `location_binpath` is `~/bin` instead of a personal path.
- Example driver: `examples/pavlof_2021/` (Alaska).

### Settings that were fixed in the code are now in the driver
- Download: station search method, retries, wait time, minimum sample rate, minimum coverage, excluded networks, search radius, backup data centers.
- PhaseNet: batch size and high-pass filter.
- GaMMA: P velocity, Vp/Vs, depth range, use of amplitudes, minimum P/S picks.
- GaMMA: choice between one job on 1 CPU (`gamma_use_dbscan = False`, default, original method) and DBSCAN groups on several CPUs (`True`, with `gamma_dbscan_eps` and `gamma_ncpu`).
- Location: templates folder, `location_min_nsta` (4, as before), velocity models given as a path to your own `.cre` file, `location_combine_net_sta`.
- QC: waveform window, bandpass, signal velocity band, neighbor-chunk padding, geometric correction, number of events.
- QC figures: `qc_max_plots` defaults to 20; `None` plots every event.

### Station search
- New radius search (`QQ_dl_stations_radius.py`, `QQ_dl_data_radius.py`): all seismic networks within `station_maxradius` of the center, the best channel set of each station chosen from the data actually available, and backup data centers when the EarthScope federator has no data. It is the default (`station_search = "radius"`); the original box search (`"box"`) is still available and is used by the Pavlof example.

### Location fixes (`QQ_location.py`), needed for large multi-network regions
- Location codes (e.g. `DK.GDH.00`) are removed from the station ids given to HypoXPy; HypoInverse rejected those phase lines.
- GaMMA magnitude 999 (no amplitudes) is given to HypoDD as 0.0; 999 stopped HypoDD.
- Results of an earlier run are deleted before a new run, so a failed step cannot silently reuse old files.
- `location_combine_net_sta = "auto"` uses station codes only when `NET.STA` is longer than HypoDD's 7 characters.

### Other scripts
- `QQ_merge_stationxml.py` takes `--data_root` instead of a fixed path.
- Plot titles in `qq_helpers.py` take a `region_name` instead of "Pavlof"; each seismic network has a fixed colour in maps.

### Figures
- QC figures: larger figure and text (10 x 7 in), thicker axes and ticks.
- `qq_helpers.plot_event_moveout_pretty`: moveout figure with control of figure size, text sizes, tick spacing and line widths, and optional saving; `visualization.ipynb` uses it.
- New notebook `catalog_visualization_clean.ipynb`: publication figures of the final catalog, saved in `data/images/` of the project.

### Language and documentation
- All comments, messages and notebooks are in English. `visualizacion.ipynb` is now `visualization.ipynb`, and `qc copy.ipynb` is now `qc_inspection.ipynb`.
- Notebooks have a few lines at the top for the project's data folder instead of absolute paths; their stored outputs were cleared.
- One README covers installation, the repository contents, project setup, outputs, tips for large networks, troubleshooting and the scripts.

### Installation
- `environment.yml` pins the versions known to work (Python 3.10, numpy 1.26, ObsPy 1.5, TensorFlow/Keras 2.14, scikit-learn 1.2.2).
- HypoXPy (v0.3.0) is registered as a git submodule, so `git clone --recurse-submodules` downloads it.
- `install.sh` works on macOS and Linux, no longer edits `~/.bashrc`, and checks for `hyp1.40`, `hypoDD` and `ph2dt`.
- `.gitignore` keeps results, waveforms, caches and compiled objects out of git.

### Removed
- Results saved in git (`data/`), `.dockerignore`, `qq_spec.txt` (old Linux/Python 3.9 environment), `waveform_index.pkl`, old figures in `quickquake/`, a compiled cache, and the one-line `changes.txt` and folder READMEs.
