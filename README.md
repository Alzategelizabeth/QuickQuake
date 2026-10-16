# QuickQuake
*Simple and streamlined workflow for earthquake detection and relocation*

![plot1](/figs/QuickQuakeLogo.png)

This repository contains the Python scripts for wrapping up and streamlining the procedures and computer codes for earthquake detection and relocation. Some scripts are modified from QuakeFlow. For detailed description and usage of each individual package, please read the corresponding documentation.

## Major earthquake detection packages used in this workflow.

| Step | Program |
|---|---|
| Station and waveform download | [ObsPy](https://github.com/obspy/obspy) FDSN clients (EarthScope/IRIS and others) |
| Phase picking | [PhaseNet](https://github.com/AI4EPS/PhaseNet) (deep learning) |
| Association | [GaMMA](https://github.com/AI4EPS/GaMMA) (Bayesian Gaussian mixture) |
| Location and relocation | HypoInverse and HypoDD, through [HypoXPy](https://github.com/xtyangpsp/HypoXPy) |
| Quality control | QuickQuake energy-versus-velocity test |

Parts of the download, picking and association scripts are modified from [QuakeFlow](https://github.com/AI4EPS/QuakeFlow ).

## This repository

| Path | What it is |
|---|---|
| `quickquake/QQ_driver.py` | **The file you edit**: all settings at the top  |
| `quickquake/` | The scripts of each step (see section 6) |
| `examples/` | A ready-to-run driver: Pavlof Volcano (Alaska) |
| `hypox_templates/` | Velocity models and HypoInverse/HypoDD parameter templates |
| `dependencies/` | Source code of PhaseNet, GaMMA and HypoXPy (HypoXPy includes HypoInverse and HypoDD) |
| `install.sh` | The installer just run once|
| `environment.yml` | List of Python packages and versions that `install.sh` needs |


## 1. Install

You need [Miniforge](https://github.com/conda-forge/miniforge) (conda) and, for HypoInverse/HypoDD, a Fortran compiler (`gfortran`).

```bash
git clone --recurse-submodules https://github.com/xtyangpsp/QuickQuake.git
cd QuickQuake
bash install.sh            # creates the conda environment "quickquake"
conda activate quickquake
```

`install.sh` creates the environment from `environment.yml`, installs PhaseNet, GaMMA and HypoXPy from `dependencies/`, and checks whether the HypoInverse/HypoDD programs are in `~/bin`.

### Installing HypoInverse and HypoDD

The location step needs three compiled programs: `hyp1.40` (HypoInverse), `hypoDD` and `ph2dt`. Their source code comes with [HypoXPy](https://github.com/xtyangpsp/HypoXPy), in `dependencies/HypoXPy/external/` (downloaded by `git clone --recurse-submodules`). Compile them from there: the HypoXPy makefile of HypoInverse has the compiler options this old Fortran code needs. Install `gfortran` (macOS: `brew install gcc`; Ubuntu: `sudo apt install gfortran`), then:

```bash
mkdir -p ~/bin
cd dependencies/HypoXPy/external/hypoinverse1.40/source && make && cp hyp1.40 ~/bin/ && cd -
cd dependencies/HypoXPy/external/hypodd2.1/src && make && cp hypoDD/hypoDD ph2dt/ph2dt ~/bin/ && cd -
```

If `make` fails, see the README in each folder. The programs can live in any folder; set `location_binpath` in your driver to that folder.

## 2. Run the example

```bash
python examples/pavlof_2021/QQ_driver.py
```

This processes 16 hours of data from Pavlof Volcano, Alaska (about 10 minutes); results go to `examples/pavlof_2021/data/`.

## 3. Your own project

Keep the code and your projects in separate folders, so each project keeps its own settings, data and results:

```
~/src/QuickQuake/                 the code (this repository)
~/research/my_region/
    QQ_driver.py                  your copy of the driver, with your settings
    data/                         created by QuickQuake: chunks, merged results, logs
```

1. Copy `quickquake/QQ_driver.py` (or one of the examples) into your project folder.
2. At the top, set `QUICKQUAKE_DIR` to the folder where QuickQuake is installed, e.g. `"~/src/QuickQuake"`.
3. Edit the settings in the CONFIGURATION section and run it: `python ~/research/my_region/QQ_driver.py`

Results are written to a `data/` folder next to your `QQ_driver.py`.

The settings you will change most often:

| Setting | Meaning |
|---|---|
| `START`, `END`, `HOUR_STEP` | Period to process (UTC) and length of each processing chunk (hours) |
| `RUN_...` | Turn each step on or off, e.g. rerun only `RUN_LOCATION` and `RUN_QC`; the options of each step are listed right below its switch |
| `center`, `deg` | `(longitude, latitude)` of the area, and the width (degrees) of the box around it where GaMMA searches for events |
| `station_maxradius` | Every seismic station of every network within this distance (degrees) of `center` is used; the best channel set of each station is chosen from the data actually available |
| `station_excluded_networks`, `station_excluded_stations` | Networks (e.g. `"SY"`, synthetic data) and stations (as `"NET.STA"`, e.g. `"DK.ILULI"`) to leave out |
| `gamma_use_dbscan` | `False` (default): one job on one CPU (original). `True`: DBSCAN groups on several CPUs (`gamma_ncpu`), much faster; choose `gamma_dbscan_eps` by comparing DBSCAN on and off on one chunk (see the comments in `QQ_driver.py`) |
| `location_p_model`, `location_s_model` | Velocity models: a file in `hypox_templates/` or a path to your own `.cre` file |
| `location_ref_ele` | Reference elevation for HypoInverse (highest station or topography, km) |
| `qc_max_plots` | Number of QC figures (default 20; `None` = one for every event) |

Use one project folder per region and period: the merge step combines **every** chunk folder inside `data/`.

## 4. Results

Inside the `data/` folder next to your driver:

| Path | Content |
|---|---|
| `YYYYMMDDTHHMMSS/` | One folder per time chunk: `config.json`, `stations.*`, `waveforms/`, `picks.csv` (PhaseNet), `gamma_catalog.csv` and `gamma_picks.csv` (GaMMA) |
| `merged/gammacatalog_id.csv`, `merged/gammapicks_id.csv` | All GaMMA events and picks; `event_id` is `<chunk>_<index>`, and `<chunk>_-1` marks picks not linked to any event |
| `merged/output/GAMMA_hypoinv_good.csv` | HypoInverse locations that passed quality checks |
| `merged/output/GAMMA_hypodd_catalog.csv` | HypoDD relocated catalog |
| `merged/output/GAMMA_hypodd_catalog_qcfiltered.csv` | **Final catalog**: relocated events that passed QC |
| `merged/output/..._qcrejected.csv`, `..._qcskipped.csv` | Events that failed QC, or could not be tested  |
| `merged/output/qc_plots_GAMMA/` | QC figures  |



## 5. Troubleshooting

| Message | Fix |
|---|---|
| `QuickQuake was not found` | Set `QUICKQUAKE_DIR` at the top of your driver |
| Location step: `hyp1.40` not found | Compile HypoInverse/HypoDD (section 1) or set `location_binpath` |
| macOS, while compiling: `ld: library 'System' not found` | Run `make` as `SDKROOT=$(xcrun --show-sdk-path) make` |
| `[SKIP] GaMMA: picks.csv does not exist` | No data or no picks in that chunk; check the chunk's download output in the log |
| GaMMA is very slow | `gamma_use_dbscan = True` (see `gamma_use_dbscan` in section 3) |


## 6. Scripts and notebooks

You only run `QQ_driver.py`; it calls the scripts below in order.

| File | Step |
|---|---|
| `QQ_config.py` | Writes `config.json` for one time chunk |
| `QQ_dl_stations_radius.py`, `QQ_dl_data_radius.py` | Stations within `station_maxradius` (all networks) and their waveforms; best channel set of each station chosen from the data; backup data centers when the EarthScope federator has no data |
| `QQ_predict.py` | PhaseNet P/S picking |
| `QQ_gamma.py` | GaMMA association (optional DBSCAN + several CPUs) |
| `QQ_merge_gamma_outputs.py` | Combines all chunks into `merged/` |
| `QQ_pick_arpick.py` | Optional pick refinement with ObsPy `ar_pick` |
| `QQ_location.py` | HypoInverse + HypoDD through HypoXPy |
| `QQ_qc.py` | Quality control and QC figures |
| `qq_helpers.py` | Plotting functions used by the notebooks (3D views also need the optional packages `plotly` and `pygmt`) |

| Notebook | Use |
|---|---|
| `quickquake/visualization.ipynb` | Maps, 3D views, cross-sections and timelines of the catalog |
| `quickquake/catalog_visualization_clean.ipynb` | Publication figures of the final catalog (daily counts, maps, zoom, timeline, cross-section); figures are saved in `data/images/` |
| `quickquake/qc_inspection.ipynb` | Inspect the QC energy-versus-velocity curves |
| `quickquake/frequency_index/FI.ipynb` | Frequency Index classification (VT / LP); first run `QQ_merge_stationxml.py --data_root <your data folder>` |

Each notebook has a few lines to edit at the top: the `data` folder of your project and the file prefix (`GAMMA`).

## Credits

QuickQuake is developed by Elizabeth Alzate Gutierrez and Xiaotao Yang. If you use it, please also cite the programs it runs:

- PhaseNet: Zhu, W., & Beroza, G. C. (2019). PhaseNet: a deep-neural-network-based seismic arrival-time picking method. *Geophysical Journal International*, 216(1), 261–273.
- GaMMA: Zhu, W., McBrearty, I. W., Mousavi, S. M., Ellsworth, W. L., & Beroza, G. C. (2022). Earthquake phase association using a Bayesian Gaussian mixture model. *Journal of Geophysical Research: Solid Earth*, 127, e2021JB023249.
- QuakeFlow: Zhu, W., et al. (2023). QuakeFlow: a scalable machine-learning-based earthquake monitoring workflow with cloud computing. *Geophysical Journal International*, 232(1), 684–693.
- HypoInverse: Klein, F. W. (2002). User's guide to HYPOINVERSE-2000. *U.S. Geological Survey Open-File Report* 02-171.
- HypoDD: Waldhauser, F. (2001). hypoDD: a program to compute double-difference hypocenter locations. *U.S. Geological Survey Open-File Report* 01-113.
- ObsPy: Beyreuther, M., et al. (2010). ObsPy: a Python toolbox for seismology. *Seismological Research Letters*, 81(3), 530–533.
- HypoXPy: https://github.com/xtyangpsp/HypoXPy
