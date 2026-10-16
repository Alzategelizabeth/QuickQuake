#!/usr/bin/env python3
"""
This code is based on the "QuakeFlow" repository by Weiqiang Zhu (2021) 
See https://github.com/AI4EPS/QuakeFlow for more details.
This code has been modified 
"""
import os
import json
import pickle
import time
import argparse
import obspy
from obspy.clients.fdsn import Client
from obspy.io.mseed import InternalMSEEDError

def download_waveforms(config_json, output_dir, max_retries=10, retry_sleep_s=5.0):
    # INPUT 1: config.json 
    with open(config_json) as fp:
        config = json.load(fp)

    # Use the FDSN client defined in config.json (written by the driver).
    # If the "client" key is missing, fall back to IRIS.
    client_name = config.get("client", "IRIS")
    client = Client(client_name)
    print(f"[waveforms] FDSN client = {client_name}")

    # OUTPUT DIR: waveforms/ (created here; PhaseNet later reads it as --data_dir)
    waveform_dir = os.path.join(output_dir, "waveforms")
    os.makedirs(waveform_dir, exist_ok=True)

    with open(os.path.join(output_dir, "stations.pkl"), "rb") as fp:
        stations = pickle.load(fp)

    stream = obspy.Stream()
    starttime = obspy.UTCDateTime(config['starttime'])
    endtime = obspy.UTCDateTime(config['endtime'])
    fname = f"{starttime.datetime.strftime('%Y-%m-%dT%H-%M-%S')}.mseed"

    for network in stations:
        for station in network:
            print(f"********{network.code}.{station.code}********")
            retry = 0
            while retry < max_retries:
                try:
                    tmp = client.get_waveforms(
                        network.code, station.code, "*", config["channels"],
                        starttime, endtime
                    )
                    stream += tmp
                    break
                except Exception as err:
                    print(f"Error {network.code}.{station.code}: {err}")
                    retry += 1
                    time.sleep(retry_sleep_s)

    if stream:
        outpath = os.path.join(waveform_dir, fname)
        try:
            stream.write(outpath, format="MSEED")  # default (STEIM2)
        except InternalMSEEDError:
            # Robust fallback: write raw INT32 samples instead of STEIM2 compression
            stream.write(outpath, format="MSEED", encoding="INT32")

        print(f"Waveforms saved at: {outpath}")
        with open(os.path.join(output_dir, "input_data.csv"), "w") as fp:
            fp.write(f"fname\n{fname}\n")
    else:
        print("Download failed - Empty Stream")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--max_retries", type=int, default=10,
                        help="Download attempts per station before giving up (default: 10)")
    parser.add_argument("--retry_sleep_s", type=float, default=5.0,
                        help="Seconds to wait between attempts (default: 5)")
    args = parser.parse_args()

    download_waveforms(args.config, args.output_dir,
                       max_retries=args.max_retries, retry_sleep_s=args.retry_sleep_s)
