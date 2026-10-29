# Changelog

Versions stay 0.x until the first official release (1.0).

## Unreleased

- `qq_helpers.plot_mapview_seismicity` and `plot_mapview_seismicity_zoom`: the depth colour bar has 0 km at the top and the deepest value at the bottom (`depth_down=True`, new). Its ticks are now integers chosen by matplotlib; before, they were 9 evenly spaced values written as `int`, so with a 0–13 km bar the ticks at 1.625, 3.25 … km were labelled 1, 3 … The two functions share the new `_depth_colorbar`. In `plot_mapview_seismicity` the legend shows only the earthquakes (`station_legend=False`, new); the networks are in the station map.
- `visualization.ipynb`: the zoom window is set in the zoom cell (`ZOOM_LON`, `ZOOM_LAT` in degrees, passed as `xlim` / `ylim`; `None` keeps `zoom_factor=3`), and the cell prints how many events fall inside it.

- `qq_helpers.plot_mapview_basemap_stations`: the map now covers every station as well as every event. Before, its extent came from the events only, so with a catalog cut to `RADIUS_KM` the farther stations were outside the map: on Aniakchak 2021–2024 (run C, 20 km) it showed 14 of the 40 stations, all of them AV. `visualization.ipynb` draws it with `relief_zoom=9` (a region of about 200 km needs fewer shaded-relief tiles than the 20 km seismicity maps) and calls section 2 "Region and stations".
- Fixed network colours for AK, AT, AM and 6J. AT was not in `NETWORK_COLOURS` and its fallback colour (`#3182bd`) was almost the same blue as AV (`#1f77b4`).
- Station legends count each station once and say "1 station" instead of "1 stations". A station is listed once per instrument (`AV.ANPK..BH` and `AV.ANPK..EH`), so on Aniakchak run C the legend said "AV (33 stations)" for 25 AV stations.
- `visualization.ipynb` draws only the stations that were used: at least one pick in an event of the located catalog (the whole run, before the radius cut), with the new `qq_helpers.stations_with_picks`. The other stations were downloaded but PhaseNet / GaMMA never used them; the notebook prints them. On Aniakchak run C this removes AM.R05D1 (Raspberry Shake, no PhaseNet pick in four years) and keeps 31 stations (39 instruments).

- `visualization.ipynb` and `catalog_analysis.ipynb` now follow the order of a paper, and they bring in what was still only in the old project notebooks (`quickquake_work`). Visualization: context (new regional location map and eruption history), map view, time (the timeline legend is now its own figure), depth, 3-D (new optional GIF, `MAKE_GIF`), location changes, waveforms. Analysis: catalog summary (new), comparison with reference catalogs, Mc, temporal evolution (new), FI (new class statistics), structures (new rose of dip / dip direction of the cluster planes, new two-plane fit). In `catalog_analysis.ipynb`, `START` / `END` now cut the catalog for every section; before, only the DBSCAN section used them. Checked on Aniakchak 2021–2024 run B: the comparison table, Mc (0.40) and the DBSCAN clusters (2028 / 811 / 170 events, surface degrees 2 / 2 / 1) are the same as before the reordering.
- `qq_helpers.plot_regional_location_map`: NOAA moved the GSHHS coastline files and cartopy 0.25 still downloads them from the old address (HTTP 404), so the map failed on a computer without a local copy. It now uses the Natural Earth 50 m land polygons when GSHHS is not available, and prints that it did.
- New `qq_helpers` functions for those sections: `plot_eruption_history` (from timeline.ipynb); `catalog_stage_counts`, `catalog_summary_table`, `plot_depth_magnitude_distributions`, `plot_depth_distance_through_time`, `plot_interevent_times`, `weekly_screening_table` and `plot_fi_class_statistics` (from estadisticas.ipynb); `fit_two_planes_from_seeds` and the functions it uses (`fit_plane_tls`, `points_in_seed_volume`, `build_loose_roi_mask`, `sigma_clip_plane`, `refine_two_planes`, `plane_intersection_line`, `plane_polygon_from_fit`, `two_plane_polys`) (from planos0.0.ipynb). `plot_interevent_times` uses log-spaced bins, because with linear bins (as in estadisticas) almost every event of a multi-year catalog falls into the first bin. Checked on run B: `fit_two_planes_from_seeds` with the planos0.0 seed volumes and settings gives the same seeds (716 / 1582 events), candidates (5132), planes (972 / 1683 events) and orientations (dip 60.01° toward 287.87°, 40.28° toward 273.46°) as planos0.0.

- Download: every channel is now saved at one sampling rate, `download_resample_rate` (default 100 Hz, the rate PhaseNet was trained on; `--resample_rate` in `QQ_dl_data.py`). Before, each channel kept its native rate, and the QC failed when one channel was stored at two rates (a station that changed from 40 to 50 Hz, or one that reports 99.99994659 Hz on some days). Channels faster than the new rate are low-pass filtered at 0.4 × the new rate before resampling (anti-alias); slower channels are interpolated, and the download prints a warning since this adds no signal above their native Nyquist frequency. Interpolation is Lanczos (a = 20): on v05 Aniakchak data (AV.S14K at 40 Hz) the 1–15 Hz spectrum is unchanged (ratio 1.000), while ObsPy's `weighted_average_slopes` (used by seisgo) keeps only 77 % of a 15 Hz wave recorded at 40 Hz and returned a garbage last sample on short segments. Each station is resampled before its segments are merged, so a rate change within a chunk merges into one trace. Integer counts stay integers (compressed MiniSEED). `download_resample_rate = None` keeps the native rates.
- HypoDD keeps the events in the volcanic edifice. HypoInverse gives CRE depths below sea level, so events above sea level (up to the model top, `location_ref_ele`) have negative depths; HypoDD 2.1 moves any depth shallower than 0.01 km down to 0.01 km and drops events that move above its zero (air quakes). `QQ_location.py` now has HypoDD measure depth from the model top: the HypoDD phase file gets every HypoInverse depth + `location_ref_ele`, the HypoDD layer tops are the `.cre` tops unchanged, and HypoXPy's `dep_corr` (now always `location_ref_ele`; the `--dep_corr` option was removed) subtracts it again in `<namebase>_hypodd_catalog.csv`, which stays below sea level like HypoInverse. The HypoInverse catalog is unchanged. On Aniakchak 2021–2024 the AVO model with depths from sea level made HypoDD diverge (0 events); from the model top it relocated 9,097 events in a test, 95 of them above sea level. HypoDD sees every station at the model top (station elevations below its zero are set to 0).
- HypoDD now uses the same velocity model as HypoInverse. Before, `hypox_templates/template_hypodd_par.inp` held the HypoDD example model (6 layers, Vp 3.77–7.98 km/s, Vp/Vs 1.73), so relocation used a different model from location (`location_p_model`, `location_s_model`). `QQ_location.py` now writes the HypoDD forward model from the two `.cre` files (IMOD = 1: one Vp/Vs ratio per layer, from the P and S models), with the layer tops moved up by `location_ref_ele`: HypoInverse gives CRE depths below sea level, and HypoDD 2.1 measures layer tops from sea level and adds station elevations. The template's model section is now only a note. Checked on the Aniakchak example (2023-03-30): HypoInverse catalog unchanged; HypoDD depths change by -0.06 to +0.41 km (median 0), horizontal positions by at most 0.01 km; the same 23 of 34 events pass QC. Catalogs made before this change should be relocated again (`RUN_LOCATION` and `RUN_QC`).
- Download: the radius search is now the only method. The box-search scripts `QQ_dl_stations.py` and `QQ_dl_data.py` (fixed networks and channels) were removed, and so were the driver options `station_search`, `networks` and `channels`. The default data center is `EARTHSCOPE_FEDERATOR`.
- The radius-search scripts `QQ_dl_stations_radius.py` and `QQ_dl_data_radius.py` were renamed to `QQ_dl_stations.py` and `QQ_dl_data.py`, since they are now the only download scripts.
- New driver option `station_excluded_stations`: stations to leave out, as `"NET.STA"`.
- `examples/pavlof_2021/`: radius search within 0.6 degrees of the volcano (the AV network is within 0.51 degrees).
- README: HypoInverse is compiled from HypoXPy (`dependencies/HypoXPy/external/hypoinverse1.40`), whose makefile has the compiler options this old Fortran code needs; added the macOS fix for `ld: library 'System' not found`.
- Removed `dependencies/hyp1.40`: a second copy of the HypoInverse source with a makefile without those options (on the Pavlof example its HypoInverse catalog had 56 instead of 57 good locations), plus 47 compiled `.o` files.
- One-step installation: `environment.yml` now also installs the compilers (`gfortran`, `c-compiler`, `make`), and `install.sh` compiles HypoInverse (`hyp1.40`), HypoDD and `ph2dt` from `dependencies/HypoXPy/external/` into the conda environment (in a temporary folder, from the sources stored in git, so old `.o` files are never reused and nothing is written to `~/bin`). Checked on the Pavlof example: programs built with conda-forge gfortran give the same results as programs built with Homebrew gfortran from the same sources.
- Drivers: `location_binpath = None` (new default) uses the programs compiled into the environment, or `~/bin` if they are not there.
- Removed `environment_mac.yml`: `environment.yml` solves on macOS (Intel and Apple chips).
- README: installation is clone (`--depth 1`, without the history) + `bash install.sh` + `conda activate`; the manual compilation section and the macOS `SDKROOT` fix were removed (install.sh does both).
- `.gitignore`: `compile_error.log`.
- `examples/pavlof_2021/QQ_driver.py`: `RUN_CONFIG`, `RUN_DL` and `RUN_PHASENET` are on again, so the example runs from the download to the QC. Checked from a fresh clone: `install.sh` (3 min) + the Pavlof example (4.5 min) give HypoDD and QC catalogs byte-identical to the previous run.
- 3D views without pygmt: pygmt (GMT) cannot be installed next to TensorFlow 2.14, because GMT needs newer system libraries (icu, grpc, abseil, proj). The new `qq_helpers.load_earth_relief` downloads the same SRTM15+ v2.7 tiles from the GMT data server and reads them with `requests` + `pillow`; it returns the same grid as `pygmt.datasets.load_earth_relief` (checked: identical values and coordinates for 8 test regions at 15s, 30s, 01m and 05m, including regions over 2 and 4 tiles). The five 3D functions use it. Tiles are kept in `~/.quickquake/earth_relief/`.
- `environment.yml`: added `pillow`, `plotly=7.0` and `python-kaleido=1.3` (newest versions for Python 3.10; they install without changing TensorFlow, Keras or scikit-learn).
- `qq_helpers.plot_catalog_3d_topo_seismicity`: `write_image` no longer passes `engine=` (removed in plotly 7); the `image_engine` option is kept but unused.
- `qq_helpers.plot_mapview3d_timecolor` and `plot_mapview3d_months`: fixed `NameError: sizes` (since v4.7); marker area is again `size_k * magnitude`, as before v4.7.
- Notebooks: the plotly renderer is `plotly_mimetype+notebook_connected`, so the 3D figure shows in VS Code as well as in Jupyter.
- The driver is split in two: `QQ_driver.py` now holds only the settings (about 170 lines instead of 460), and the code that ran the steps moved unchanged to the new `quickquake/QQ_run.py`; the driver ends with `run(globals())`. Drivers copied into projects no longer carry their own (aging) copy of the code. `QQ_run.py` checks the settings: region settings missing -> clear stop; other settings missing -> the default (same values as `quickquake/QQ_driver.py`) is used and listed; misspelled settings -> warning with the closest name; old settings (`station_search`, `networks`, `channels`, `None` retries) -> note, ignored. Checked: the Pavlof example (download to QC) gives byte-identical catalogs; the location step of Aniakchak and Jakobshavn gives byte-identical HypoInverse and HypoDD outputs.
- `install.sh`: `env create -y`, so it no longer stops at "Confirm changes [Y/n]". Because `-y` also makes conda delete an existing environment of the same name, `install.sh` first checks whether the environment exists, ignoring upper/lower case (on macOS `quickquake` and `QuickQuake` are the same folder); if it exists it stops without changing anything, unless you run `bash install.sh NAME --update` to reinstall into it.
- README: `gamma_use_dbscan` is `True` by default (it said `False`).

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
