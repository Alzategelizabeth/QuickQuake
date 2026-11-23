#!/usr/bin/env python3
"""
QuickQuake example: Aniakchak volcano (Alaska), 12 March 2023.

One day of the 2021-2024 Aniakchak catalog, with the same settings. Run on 22 November 2026, this
example gave 239 events in the final catalog (17 more rejected by the QC), all within 20 km of the
volcano, with 15 stations. The four-year run first had only 64 events on this day, because its download
could not get the instrument response of station AV.ANVM, close to the activity, and used 14 stations;
after that day was downloaded again (23 November 2026) it has 242 within 20 km.
Other small differences: HypoDD relocates here only the events of this day (the four-year run relocated
all of them together), and data centers can change the data they serve.

Run it from the QuickQuake folder (about 10 minutes, most of it PhaseNet):
    conda activate quickquake
    python examples/aniakchak_2023/QQ_driver.py
Results are written to examples/aniakchak_2023/data/. Then open catalog_visualization.ipynb and
catalog_analysis.ipynb in this folder.

This file holds only the settings: quickquake/QQ_run.py runs the steps with them. Settings that are not
here keep the default values of the template driver, quickquake/QQ_driver.py.
"""

from pathlib import Path
import sys

#
# CONFIGURATION
#

# Paths
QUICKQUAKE_DIR = Path(__file__).resolve().parents[2]   # this example lives in <QuickQuake>/examples/aniakchak_2023/
BASE_DIR  = Path(QUICKQUAKE_DIR).expanduser().resolve()   # QuickQuake code
DATA_ROOT = Path(__file__).resolve().parent / "data"      # results, next to this file
MODEL_DIR = BASE_DIR / "dependencies/PhaseNet/model/190703-214543"

# Period, processed in chunks of HOUR_STEP hours
START, END = ["2023-03-12T00:00:00", "2023-03-13T00:00:00"]
HOUR_STEP = 24

RUN_CONFIG = True   # region and data center
center = (-158.209, 56.9058)   # longitude, latitude of the region of interest
deg    = 1                     # box width (degrees) where GaMMA searches
client = "EARTHSCOPE_FEDERATOR"
region = "aniakchak"

RUN_DL = True   # every network within station_maxradius
station_maxradius = 1.2           # degrees around center
station_excluded_networks = ["SY"]
station_min_sample_rate = 20.0    # Hz
station_min_coverage = 0.1        # minimum fraction of a chunk with data
download_resample_rate = 0        # keep the native sample rate

RUN_PHASENET = True
phasenet_min_p_prob, phasenet_min_s_prob = [0.30, 0.30]
phasenet_mpd = 50                 # minimum distance between picks (samples)

RUN_GAMMA = True
gamma_method            = "BGMM"
gamma_oversample_factor = 30
gamma_min_picks_per_eq  = 6
gamma_max_sigma11, gamma_max_sigma22, gamma_max_sigma12 = [2, 1, 1]
# s, log10(m/s), covariance
gamma_vp, gamma_vp_vs_ratio = [6.0, 1.73]   # uniform velocity model
gamma_use_amplitude = True
gamma_use_dbscan    = True        # group the picks before GaMMA

RUN_MERGE_GAMMA = True   # join the events of all chunks

RUN_LOCATION = True   # HypoInverse and HypoDD through HypoXPy
location_templates_dir = BASE_DIR / "hypox_templates"
location_p_model, location_s_model = ["velo_p_rv_avo.cre", "velo_s_rv_avo.cre"]
location_ref_ele = 1.3            # highest elevation of the topography (km)
location_depth_min, location_depth_max, location_depth_step = [0.0, 20.0, 0.5]
location_min_nsta = 4

RUN_QC = True   # glitch check, then waveform QC of the HypoDD events
qc_glitch_min_stations = 2     # glitch: stations flat before the pick
qc_glitch_min_seconds = 1.0    # flat: nearly constant for this long (s)
qc_min_total_stations = 3      # minimum stations with picks
qc_freqmin, qc_freqmax = [1.0, 15.0]   # bandpass filter (Hz)
qc_station_normalization = "noise"     # divide each station by its noise
qc_noise_window_s = 300.0      # noise: median over 300 s before origin
qc_winlen = 3                  # window around the predicted arrival (s)
qc_vmin_curve, qc_vmax_curve, qc_vsteps_curve = [2.0, 8.0, 150]
# trial velocities (km/s) of the energy curve
qc_noise_percentile, qc_signal_percentile = [50.0, 90.0]   # Q = P90/P50
qc_min_ratio = 4.0             # event accepted if Q >= 4

#
# RUN (no need to edit below: quickquake/QQ_run.py runs the steps with the settings above)
#

if __name__ == "__main__":
    if not (BASE_DIR / "quickquake" / "QQ_run.py").is_file():
        sys.exit(f"QuickQuake was not found in {BASE_DIR}. "
                 "Set QUICKQUAKE_DIR at the top of this file to the folder where QuickQuake is installed.")
    sys.path.insert(0, str(BASE_DIR / "quickquake"))
    from QQ_run import run
    run(globals())
