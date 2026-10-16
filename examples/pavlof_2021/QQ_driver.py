#!/usr/bin/env python3
"""
QuickQuake: run pipeline in time chunks Config > Stations > Download > PhaseNet > GaMMA,
then run a separate merge script, then optionally run HypoXPy relocation, quality control and classification of volcanic signals 

This file is an ORCHESTRATOR only

EXAMPLE: Pavlof Volcano, Alaska, 29 July 2021 (a small, dense volcano network).
It uses the original box station search and GaMMA without DBSCAN, so it reproduces
the results of the previous QuickQuake code (commit b9f0e9c).

How to use it:
    1. Copy this file into a folder for your project (e.g. ~/research/my_region/).
    2. Set QUICKQUAKE_DIR below to the folder where QuickQuake is installed.
    3. Edit the settings in the CONFIGURATION section.
    4. Run it:   python QQ_driver.py
Results are written to a "data" folder next to this file.
"""

import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
import traceback

# 
# CONFIGURATION
# 

QUICKQUAKE_DIR = Path(__file__).resolve().parents[2]   # this example lives in <QuickQuake>/examples/pavlof_2021/

START = "2021-07-29T08:00:00"
END   = "2021-07-30T00:00:00"
HOUR_STEP = 8

BASE_DIR  = Path(QUICKQUAKE_DIR).expanduser().resolve()   # QuickQuake code
DATA_ROOT = Path(__file__).resolve().parent / "data"      # results, next to this file
MODEL_DIR = BASE_DIR / "dependencies/PhaseNet/model/190703-214543"


RUN_CONFIG   =  True

# config options (user edits )
center   = (-161.8903, 55.4133) # central cooordinates for your region of interest 
deg      = 1.0
networks = ["AV"]                     # box search only (radius search uses every network)
channels = "BHZ,BHN,BHE,SHZ,SHN,SHE"  # box search only (radius search picks the channels itself)
client   = "IRIS"                      # FDSN data center; "EARTHSCOPE_FEDERATOR" asks all data centers
                                       # and adds the backup centers below (best for radius search)
region   = "pavlof"

RUN_DL       = True
# download options (user edits )
station_search = "box"               # "box": networks/channels above, inside the region box (original method)
                                     # "radius": every seismic station within station_maxradius; the best
                                     #           channel set of each station is chosen from the real data
station_maxradius = 2.0              # radius search: degrees around center
station_excluded_networks = ["SY"]   # radius search: networks to skip (SY = synthetic data)
station_min_sample_rate = 20.0       # radius search: ignore channels sampled below this rate (Hz)
station_target_rate = 100.0          # radius search: preferred sample rate when ranking channels (Hz)
station_min_coverage = 0.8           # radius search: minimum fraction of the chunk with data
download_max_retries = None          # attempts after a download error (None = box 10, radius 3)
download_retry_sleep = None          # seconds between attempts (None = box 5, radius 3)
download_fallback_clients = ["EARTHSCOPE", "GEOFON", "ETH"]  # radius search: backup data centers tried in
                                     # order when the federator has no data (GEOFON serves DK.GDH, ETH DK.KARAT)

RUN_PHASENET = True
# phasenet options (user edits )
phasenet_min_p_prob = 0.30
phasenet_min_s_prob = 0.30
phasenet_mpd        = 50  # minimum peak distance
phasenet_batch_size = 20  # traces processed together (lower it if memory is short)
phasenet_highpass_filter = 0.0  # high-pass filter in Hz (0 = none)


RUN_GAMMA    = True
# gamma options (user edits )
gamma_method           = "BGMM"   # default: BGMM
gamma_oversample_factor = 30       # default: 30 for BGMM
gamma_min_picks_per_eq = 6        # default: 6
gamma_max_sigma11      = 2      # S
gamma_max_sigma22      = 1     #m/s
gamma_max_sigma12      = 1     #covariance
gamma_min_p_picks_per_eq = 0    # minimum P picks per event
gamma_min_s_picks_per_eq = 0    # minimum S picks per event
gamma_use_amplitude    = True   # use pick amplitudes as well as times
gamma_vp               = 6.0    # P velocity of GaMMA's uniform model (km/s)
gamma_vp_vs_ratio      = 1.73   # Vp/Vs of GaMMA's uniform model
gamma_depth_min_km     = 0      # shallowest event depth searched (km)
gamma_depth_max_km     = 60     # deepest event depth searched (km)
gamma_use_dbscan       = False  # False = all picks of a chunk in one job on 1 CPU (original method)
                                # True  = split picks into groups (DBSCAN) and associate them on several
                                #         CPUs; use it for many stations or chunks longer than ~1 hour
gamma_dbscan_eps       = "auto" # DBSCAN group size in seconds, or "auto" (from station spacing)
                                # "auto" can be too small for dense networks: picks of one earthquake (P and S)
                                # can span 10 s or more. Pavlof: "auto" gave 6.3 s and lost events; 15 s matched
                                # the original method.
gamma_dbscan_min_samples = 3    # minimum picks to start a group
gamma_ncpu             = None   # CPUs used with DBSCAN (None = automatic, at most all CPUs - 1)



RUN_MERGE_GAMMA = True

RUN_ARPICK = False                # Enable ar_pick repicking step

arpick_namebase = "GAMMA"         # Base name used in pipeline files
arpick_pre_p = 1.0                # Seconds before original P pick in local window
arpick_post_p = 2.0               # Seconds after original P pick in local window
arpick_pre_s = 1.5                # Seconds before original S pick in local window
arpick_post_s = 3.0               # Seconds after original S pick in local window
arpick_max_dt_p = 0.13            # Max allowed P repick shift from original pick
arpick_max_dt_s = 0.30            # Max allowed S repick shift from original pick
arpick_freqmin = 1.0              # Bandpass low cutoff before ar_pick
arpick_freqmax = 20.0             # Bandpass high cutoff before ar_pick

arpick_lta_p = 1.0                # Long-term window for P trigger
arpick_sta_p = 0.1                # Short-term window for P trigger
arpick_lta_s = 2.0                # Long-term window for S trigger
arpick_sta_s = 0.2                # Short-term window for S trigger
arpick_m_p = 2                    # AR model order for P
arpick_m_s = 8                    # AR model order for S
arpick_l_p = 0.1                  # P picker smoothing/control parameter
arpick_l_s = 0.2                  # S picker smoothing/control parameter

arpick_keep_debug_cols = True     # Save debug columns in output CSV

RUN_LOCATION      =  True
location_binpath = str(Path.home() / "bin")  # folder with hyp1.40, hypoDD and ph2dt
location_namebase = "GAMMA"
location_extra_args = []   # e.g., ["--cleanup"]
location_p_model = "velo_p_rv_avo.cre"
location_s_model = "velo_s_rv_avo.cre"
location_ref_ele = 3.2 #reference location highest part in the topography 
location_depth_min = 0.0
location_depth_max = 20.0
location_depth_step = 0.5
location_templates_dir = BASE_DIR / "hypox_templates"  # HypoInverse, ph2dt and HypoDD templates (and models
                                                     # given by name); point it to your own copy to change them
location_min_nsta = 4         # minimum stations for a HypoInverse location
location_combine_net_sta = "yes"  # HypoDD station names: "yes" = NET.STA, "no" = STA,
                                  # "auto" = STA when NET.STA is longer than 7 characters (HypoDD limit)



RUN_QC= True
# qc options (user edits)
qc_min_total_stations = 3 #must be a positive number 
qc_min_valid_stations_per_v = None  # None => defaults to qc_min_total_stations
qc_vmin_curve   = 2.0
qc_vmax_curve   = 8.0
qc_vsteps_curve = 150
qc_winlen = 3
qc_noise_percentile  = 50.0
qc_signal_percentile = 90.0
qc_min_ratio = 5.0 
qc_energy_type = "squared_median" # other option is "envelope_median" or "squared_median"
qc_make_plot = True
qc_max_plots = 20 # number of QC figures; set to None to plot every event
qc_pre_s = 10.0       # seconds of waveform before origin time
qc_post_s = 40.0      # seconds of waveform after origin time
qc_freqmin = 1.0      # bandpass low corner (Hz)
qc_freqmax = 15.0     # bandpass high corner (Hz), clipped to Nyquist
qc_vmin_signal = 2.0  # lowest velocity of the signal band (km/s)
qc_vmax_signal = 8.0  # highest velocity of the signal band (km/s)
qc_pad_neighbors = True      # also read neighbor chunks when a window crosses a chunk boundary
qc_geom_correction = False   # multiply energy by distance^2
qc_max_events = 0            # test only the first N events (0 = all)

# SCRIPTS 

SCRIPTS = {
    "config":      BASE_DIR / "quickquake/QQ_config.py",
    "stations":    BASE_DIR / "quickquake/QQ_dl_stations.py",
    "download":    BASE_DIR / "quickquake/QQ_dl_data.py",
    "stations_radius": BASE_DIR / "quickquake/QQ_dl_stations_radius.py",
    "download_radius": BASE_DIR / "quickquake/QQ_dl_data_radius.py",
    "phasenet":    BASE_DIR / "quickquake/QQ_predict.py",
    "gamma":       BASE_DIR / "quickquake/QQ_gamma.py",
    "merge":       BASE_DIR / "quickquake/QQ_merge_gamma_outputs.py",
    "arpick":      BASE_DIR / "quickquake/QQ_pick_arpick.py",
    "location":    BASE_DIR / "quickquake/QQ_location.py",
    "qc_velocity": BASE_DIR / "quickquake/QQ_qc.py",
}

# HELPERS
#

def run_step(cmd, step, cwd=None):
    print("\n" + "=" * 70)
    print(f"Running: {step}")
    print("CMD:", " ".join(str(x) for x in cmd))
    print("=" * 70 + "\n")
    subprocess.run(cmd, check=True, cwd=cwd)


def process_window(start, end, out_dir: Path):
    """
    Runs ONLY until GaMMA inside each chunk folder.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    config   = out_dir / "config.json"
    stations = out_dir / "stations.json"
    picks    = out_dir / "picks.csv"

    # Download commands: box search (original) or radius search
    retries = []
    if download_max_retries is not None:
        retries += ["--max_retries", str(download_max_retries)]
    if download_retry_sleep is not None:
        retries += ["--retry_sleep_s", str(download_retry_sleep)]

    if station_search == "radius":
        stations_cmd = [
            sys.executable, str(SCRIPTS["stations_radius"]),
            "--config", str(config),
            "--output_dir", str(out_dir),
            "--maxradius", str(station_maxradius),
            "--excluded_networks", ",".join(station_excluded_networks),
            "--min_sample_rate", str(station_min_sample_rate),
            "--plot"
        ]
        download_cmd = [
            sys.executable, str(SCRIPTS["download_radius"]),
            "--config", str(config),
            "--output_dir", str(out_dir),
            "--target_rate", str(station_target_rate),
            "--min_sample_rate", str(station_min_sample_rate),
            "--min_coverage", str(station_min_coverage),
            "--fallback_clients", ",".join(download_fallback_clients),
        ] + retries
    else:
        stations_cmd = [
            sys.executable, str(SCRIPTS["stations"]),
            "--config", str(config),
            "--output_dir", str(out_dir),
            "--plot"
        ]
        download_cmd = [
            sys.executable, str(SCRIPTS["download"]),
            "--config", str(config),
            "--output_dir", str(out_dir)
        ] + retries

    # GaMMA options that are added only when used
    gamma_extra = []
    if not gamma_use_amplitude:
        gamma_extra += ["--no_amplitude"]
    if gamma_use_dbscan:
        gamma_extra += ["--use_dbscan",
                        "--dbscan_eps", str(gamma_dbscan_eps),
                        "--dbscan_min_samples", str(gamma_dbscan_min_samples)]
        if gamma_ncpu is not None:
            gamma_extra += ["--ncpu", str(gamma_ncpu)]

    steps = [
        (RUN_CONFIG, [
            sys.executable, str(SCRIPTS["config"]),
            "--start", start.isoformat(),
            "--end", end.isoformat(),
            "--output", str(config),
            "--center=" + f"{center[0]},{center[1]}",
            "--deg", str(deg),
            "--networks", ",".join(networks),
            "--channels", channels,
            "--client", client,
            "--region", region
        ], "Config"),

        (RUN_DL, stations_cmd, "Stations"),

        (RUN_DL, download_cmd, "Data Download"),

        (RUN_PHASENET, [
            sys.executable, str(SCRIPTS["phasenet"]),
            "--model_dir", str(MODEL_DIR),  # predict.py expects --model_dir
            "--data_dir", str(out_dir / "waveforms"),
            "--data_list", str(out_dir / "input_data.csv"),
            "--stations", str(stations),
            "--result_dir", str(out_dir),
            "--format", "mseed_array",
            "--amplitude",
            "--min_p_prob", str(phasenet_min_p_prob),
            "--min_s_prob", str(phasenet_min_s_prob),
            "--mpd", str(phasenet_mpd),
            "--batch_size", str(phasenet_batch_size),
            "--highpass_filter", str(phasenet_highpass_filter),
        ], "PhaseNet"),

        (RUN_GAMMA, [
            sys.executable, str(SCRIPTS["gamma"]),
            "--config", str(config),
            "--picks", str(picks),
            "--stations", str(stations),
            "--output_dir", str(out_dir),

            # user-facing gamma args
            "--method", str(gamma_method),
            "--oversample_factor", str(gamma_oversample_factor),
            "--min_picks_per_eq", str(gamma_min_picks_per_eq),
            "--max_sigma11", str(gamma_max_sigma11),
            "--max_sigma22", str(gamma_max_sigma22),
            "--max_sigma12", str(gamma_max_sigma12),
            "--min_p_picks_per_eq", str(gamma_min_p_picks_per_eq),
            "--min_s_picks_per_eq", str(gamma_min_s_picks_per_eq),
            "--vp", str(gamma_vp),
            "--vp_vs_ratio", str(gamma_vp_vs_ratio),
            "--depth_min_km", str(gamma_depth_min_km),
            "--depth_max_km", str(gamma_depth_max_km),
        ] + gamma_extra, "GaMMA"),
    ]

    for enabled, cmd, name in steps:
        if enabled:
            if name == "GaMMA" and not picks.exists():
                print(f"[SKIP] GaMMA: does not exist {picks} (PhaseNet failed or there was no data).")
                continue
            try:
                run_step(cmd, name, cwd=BASE_DIR)
            except Exception as e:
                print("Error running"+name+": "+str(e))
                traceback.print_exc()
                continue


#
# MAIN
#

def main():
    if not (BASE_DIR / "quickquake").is_dir():
        sys.exit(f"QuickQuake was not found in {BASE_DIR}. "
                 "Set QUICKQUAKE_DIR at the top of this file to the folder where QuickQuake is installed.")

    current  = datetime.fromisoformat(START)
    end_time = datetime.fromisoformat(END)

    # 1) Run chunks
    while current < end_time:
        window_end = min(current + timedelta(hours=HOUR_STEP), end_time)
        date_str   = current.strftime("%Y%m%dT%H%M%S")
        out_dir    = DATA_ROOT / date_str

        print(f"\n{'='*70}\nProcessing chunk: {current} -> {window_end}\nOutput: {out_dir}\n{'='*70}")
        # process_window(current, window_end, out_dir)
        # current = window_end
        try:
            process_window(current, window_end, out_dir)
        except Exception as e:
            print(f"[CHUNK FAILED] {current} -> {window_end}\n  {e}")
            traceback.print_exc()
        finally:
            current = window_end

    # 2) Merge (as subprocess)
    if RUN_MERGE_GAMMA:
        cmd_merge = [
            sys.executable, str(SCRIPTS["merge"]),
            "--data_root", str(DATA_ROOT),
            "--namebase", str(location_namebase),
        ]
        run_step(cmd_merge, "Merge GaMMA outputs", cwd=BASE_DIR)

   # 3) ar_pick refinement (as subprocess)
    if RUN_ARPICK:
        cmd_arpick = [
            sys.executable, str(SCRIPTS["arpick"]),
            "--data_root", str(DATA_ROOT),
            "--namebase", str(arpick_namebase),

            "--pre_p", str(arpick_pre_p),
            "--post_p", str(arpick_post_p),
            "--pre_s", str(arpick_pre_s),
            "--post_s", str(arpick_post_s),

            "--max_dt_p", str(arpick_max_dt_p),
            "--max_dt_s", str(arpick_max_dt_s),

            "--freqmin", str(arpick_freqmin),
            "--freqmax", str(arpick_freqmax),

            "--lta_p", str(arpick_lta_p),
            "--sta_p", str(arpick_sta_p),
            "--lta_s", str(arpick_lta_s),
            "--sta_s", str(arpick_sta_s),

            "--m_p", str(arpick_m_p),
            "--m_s", str(arpick_m_s),
            "--l_p", str(arpick_l_p),
            "--l_s", str(arpick_l_s),
        ]

        if arpick_keep_debug_cols:
            cmd_arpick.append("--keep_debug_cols")

        run_step(cmd_arpick, "ar_pick refinement", cwd=BASE_DIR)   

    # 3) Location (as subprocess)
    if RUN_ARPICK:
        picks_file = DATA_ROOT / "merged" / "gammapicks_id_arpick.csv"
    else:
        picks_file = DATA_ROOT / "merged" / "gammapicks_id.csv"
    if RUN_LOCATION:
        cmd_loc = [
        sys.executable, str(SCRIPTS["location"]),
        "--merged_dir", str(DATA_ROOT / "merged"),
        "--templates_dir", str(location_templates_dir),
        "--binpath", str(location_binpath),
        "--namebase", str(location_namebase),
         "--picks_file", str(picks_file),
        "--p_model", str(location_p_model),
        "--s_model", str(location_s_model),
        "--ref_ele", str(location_ref_ele),

        "--depth_min", str(location_depth_min),
        "--depth_max", str(location_depth_max),
        "--depth_step", str(location_depth_step),
        "--min_nsta", str(location_min_nsta),
        "--combine_net_sta", str(location_combine_net_sta),
    ] + list(location_extra_args)
        run_step(cmd_loc, "HypoXPy relocation (HypoInverse + HypoDD)", cwd=BASE_DIR)

    


    # 4) QC (as subprocess)
    if RUN_QC:
        cmd_qc = [
            sys.executable, str(SCRIPTS["qc_velocity"]),
            "--data_root", str(DATA_ROOT),
            "--namebase", str(location_namebase),

            "--vmin_curve", str(qc_vmin_curve),
            "--vmax_curve", str(qc_vmax_curve),
            "--vsteps_curve", str(qc_vsteps_curve),

            "--winlen", str(qc_winlen),
            "--energy_type", str(qc_energy_type),
            "--min_total_stations", str(qc_min_total_stations),

            "--noise_percentile", str(qc_noise_percentile),
            "--signal_percentile", str(qc_signal_percentile),
            "--min_ratio", str(qc_min_ratio),
            "--max_plots", str(qc_max_plots),   
            "--pre_s", str(qc_pre_s),
            "--post_s", str(qc_post_s),
            "--freqmin", str(qc_freqmin),
            "--freqmax", str(qc_freqmax),
            "--vmin_signal", str(qc_vmin_signal),
            "--vmax_signal", str(qc_vmax_signal),
            "--max_events", str(qc_max_events),
        ]

        if qc_make_plot:
            cmd_qc.append("--make_plot")
        if not qc_pad_neighbors:
            cmd_qc.append("--no_pad_neighbors")
        if qc_geom_correction:
            cmd_qc.append("--geom_correction")
        if qc_min_valid_stations_per_v is not None:
            cmd_qc += ["--min_valid_stations_per_v", str(qc_min_valid_stations_per_v)]    

        run_step(cmd_qc, "QC: velocity percentile ratio filter", cwd=BASE_DIR)







if __name__ == "__main__":
    main()
