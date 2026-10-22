#!/usr/bin/env python3
"""
Discover candidate seismic stations for QuickQuake.

Design:
- radial geographic search around config["center"]
- all networks / all stations / all channels
- EarthScope Federator first, direct EarthScope fallback
- exclude synthetic data
- exclude low-sample-rate channels before waveform download
- keep only seismic/ground-motion channel families and PhaseNet-compatible components
- DO NOT choose the final channel family here
- final waveform verification and family selection happen in QQ_dl_data.py
"""

import os
import json
import pickle
import argparse
import copy
from collections import defaultdict

from obspy import UTCDateTime
from obspy.clients.fdsn import Client, RoutingClient
from obspy.clients.fdsn.header import FDSNNoDataException


# ============================================================
# SETTINGS
# ============================================================

EXCLUDED_NETWORKS = {"SY"}
EXCLUDED_STATIONS = set()   # stations to skip, written as "NET.STA" (e.g. "DK.ILULI")
MIN_SAMPLE_RATE = 20.0

VALID_COMPONENTS = {"E", "N", "Z", "1", "2", "3"}

# FDSN source/instrument code = second channel-code character.
# H: high-gain seismometer
# L: low-gain seismometer
# N: accelerometer
# P: geophone / very-short-period seismometer
# M: mass-position seismometer
SEISMIC_SOURCE_CODES = {"H", "L", "N", "P", "M"}


# ============================================================
# CLIENTS
# ============================================================

def make_clients(client_name):
    """
    Use the EarthScope federator when requested and keep
    direct EarthScope as a fallback.
    """
    if client_name == "EARTHSCOPE_FEDERATOR":
        primary = RoutingClient("earthscope-federator")

        try:
            fallback = Client("EARTHSCOPE")
        except Exception:
            fallback = Client("IRIS")

        return primary, fallback

    return Client(client_name), None


def is_no_data_error(err):
    if isinstance(err, FDSNNoDataException):
        return True

    text = str(err).lower()

    return (
        "204" in text
        or "404" in text
        or "no data available" in text
        or "no data matched request" in text
    )


# ============================================================
# CHANNEL FILTERING
# ============================================================

def sensor_description(channel):
    try:
        value = channel.sensor.description
        return value.strip() if value else ""
    except Exception:
        return ""


def is_candidate_channel(network_code, channel):
    """
    Cheap metadata-only filter applied before any waveform download.
    """
    if network_code in EXCLUDED_NETWORKS:
        return False

    code = channel.code or ""

    if len(code) < 3:
        return False

    component = code[-1]

    if component not in VALID_COMPONENTS:
        return False

    # Second character is the FDSN source/instrument code.
    if code[1] not in SEISMIC_SOURCE_CODES:
        return False

    try:
        rate = float(channel.sample_rate)
    except Exception:
        return False

    if rate < MIN_SAMPLE_RATE:
        return False

    if "synthetic" in sensor_description(channel).lower():
        return False

    return True


def filter_inventory(inventory):
    """
    Keep only physical stations and candidate seismic channels.
    """
    output = copy.deepcopy(inventory)

    for network in output:
        if network.code in EXCLUDED_NETWORKS:
            network.stations = []
            continue

        for station in network:
            station.channels = [
                channel
                for channel in station.channels
                if is_candidate_channel(
                    network.code,
                    channel,
                )
            ]

        network.stations = [
            station
            for station in network.stations
            if station.channels
            and f"{network.code}.{station.code}" not in EXCLUDED_STATIONS
        ]

    output.networks = [
        network
        for network in output.networks
        if network.stations
    ]

    return output


# ============================================================
# METADATA DISCOVERY
# ============================================================

def query_region(
    client,
    config,
    maxradius,
):
    """
    Broad radial discovery.

    We intentionally use:
        network="*"
        station="*"
        channel="*"

    Filtering happens locally after metadata retrieval.
    """
    center_lon, center_lat = config["center"]

    try:
        return client.get_stations(
            network="*",
            station="*",
            location="*",
            channel="*",
            starttime=UTCDateTime(
                config["starttime"]
            ),
            endtime=UTCDateTime(
                config["endtime"]
            ),
            latitude=float(center_lat),
            longitude=float(center_lon),
            minradius=0.0,
            maxradius=float(maxradius),
            level="channel",
        )

    except Exception as err:
        if is_no_data_error(err):
            return None

        raise


def discover_inventory(
    primary,
    fallback,
    config,
    maxradius,
):
    inventory = None

    try:
        inventory = query_region(
            client=primary,
            config=config,
            maxradius=maxradius,
        )
    except Exception as err:
        print(
            "[stations] Federator metadata query failed: "
            f"{err}"
        )

    if (
        inventory is None
        and fallback is not None
    ):
        print(
            "[stations] Trying direct EarthScope fallback..."
        )

        try:
            inventory = query_region(
                client=fallback,
                config=config,
                maxradius=maxradius,
            )
        except Exception as err:
            print(
                "[stations] Direct metadata query failed: "
                f"{err}"
            )

    return inventory


# ============================================================
# OUTPUT HELPERS
# ============================================================

def build_candidate_json(inventory):
    """
    Candidate metadata only.

    Response information is intentionally not required here.
    QQ_dl_data.py fetches response metadata only after a waveform
    family has actually passed the waveform/coverage test.
    """
    output = {}

    for network in inventory:
        for station in network:
            for channel in station:
                code = channel.code
                family = code[:-1]
                component = code[-1]
                location = channel.location_code or ""

                sid = (
                    f"{network.code}."
                    f"{station.code}."
                    f"{location}."
                    f"{family}"
                )

                if sid not in output:
                    output[sid] = {
                        "longitude":
                            float(channel.longitude),
                        "latitude":
                            float(channel.latitude),
                        "elevation(m)":
                            float(channel.elevation),
                        "component": [],
                        "sample_rate": [],
                        "sensor": [],
                    }

                if (
                    component
                    not in output[sid]["component"]
                ):
                    output[sid][
                        "component"
                    ].append(
                        component
                    )

                rate = float(
                    channel.sample_rate
                )

                if (
                    rate
                    not in output[sid]["sample_rate"]
                ):
                    output[sid][
                        "sample_rate"
                    ].append(
                        rate
                    )

                sensor = sensor_description(
                    channel
                )

                if (
                    sensor
                    and sensor
                    not in output[sid]["sensor"]
                ):
                    output[sid][
                        "sensor"
                    ].append(
                        sensor
                    )

    return output


def print_summary(inventory):
    groups = defaultdict(list)

    for network in inventory:
        for station in network:
            for channel in station:
                groups[
                    (
                        network.code,
                        station.code,
                        channel.location_code
                        or "",
                        channel.code[:-1],
                    )
                ].append(channel)

    physical_stations = {
        (network.code, station.code)
        for network in inventory
        for station in network
    }

    print(
        "\n[stations] Candidate seismic metadata:"
    )

    previous_station = None

    for key in sorted(groups):
        (
            network_code,
            station_code,
            location,
            family,
        ) = key

        physical = (
            network_code,
            station_code,
        )

        if physical != previous_station:
            print(
                f"  {network_code}.{station_code}"
            )
            previous_station = physical

        channels = groups[key]

        codes = sorted(
            {
                channel.code
                for channel in channels
            }
        )

        rates = sorted(
            {
                float(channel.sample_rate)
                for channel in channels
            }
        )

        sensors = sorted(
            {
                sensor_description(channel)
                for channel in channels
                if sensor_description(channel)
            }
        )

        location_print = (
            location
            if location
            else "--"
        )

        print(
            f"    loc={location_print} "
            f"family={family} "
            f"channels={','.join(codes)} "
            f"rates={rates} Hz "
            f"sensor={' | '.join(sensors) if sensors else '-'}"
        )

    print(
        "\n[stations] Candidate physical stations: "
        f"{len(physical_stations)}"
    )

    counts = defaultdict(int)

    for network_code, _ in physical_stations:
        counts[network_code] += 1

    print(
        "[stations] Candidate stations by network:"
    )

    for network_code in sorted(
        counts,
        key=lambda key: (
            -counts[key],
            key,
        ),
    ):
        print(
            f"  {network_code}: "
            f"{counts[network_code]}"
        )


# ============================================================
# MAIN
# ============================================================

def download_stations(
    config_json,
    output_dir,
    maxradius=2.0,
    plot=False,
):
    with open(config_json) as fp:
        config = json.load(fp)

    client_name = config.get(
        "client",
        "EARTHSCOPE_FEDERATOR",
    )

    primary, fallback = make_clients(
        client_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    center_lon, center_lat = (
        config["center"]
    )

    print(
        f"[stations] FDSN client = "
        f"{client_name}"
    )

    print(
        "[stations] Discovery mode: "
        "all networks / all stations / all channels"
    )

    print(
        f"[stations] Center = "
        f"({center_lon}, {center_lat})"
    )

    print(
        f"[stations] Search radius = "
        f"{maxradius} degrees"
    )

    print(
        f"[stations] Hard filters: "
        f"network not in {sorted(EXCLUDED_NETWORKS)}, "
        f"station not in {sorted(EXCLUDED_STATIONS)}, "
        f"sample_rate >= {MIN_SAMPLE_RATE} Hz, "
        f"components={sorted(VALID_COMPONENTS)}"
    )

    raw_inventory = discover_inventory(
        primary=primary,
        fallback=fallback,
        config=config,
        maxradius=maxradius,
    )

    if (
        raw_inventory is None
        or not raw_inventory.networks
    ):
        raise RuntimeError(
            "No station metadata found "
            "for the requested region/window."
        )

    stations = filter_inventory(
        raw_inventory
    )

    if not stations.networks:
        raise RuntimeError(
            "Metadata was found, but no channels "
            "survived the seismic/sample-rate filters."
        )

    print_summary(
        stations
    )

    # Candidate inventory used by QQ_dl_data.py.
    stations.write(
        os.path.join(
            output_dir,
            "stations_candidates.xml",
        ),
        format="STATIONXML",
    )

    with open(
        os.path.join(
            output_dir,
            "stations_candidates.pkl",
        ),
        "wb",
    ) as fp:
        pickle.dump(
            stations,
            fp,
        )

    candidates_json = (
        build_candidate_json(
            stations
        )
    )

    with open(
        os.path.join(
            output_dir,
            "stations_candidates.json",
        ),
        "w",
    ) as fp:
        json.dump(
            candidates_json,
            fp,
            indent=2,
        )

    # Keep these two original QuickQuake filenames so the
    # next pipeline step remains compatible.
    stations.write(
        os.path.join(
            output_dir,
            "stations.xml",
        ),
        format="STATIONXML",
    )

    with open(
        os.path.join(
            output_dir,
            "stations.pkl",
        ),
        "wb",
    ) as fp:
        pickle.dump(
            stations,
            fp,
        )

    with open(
        os.path.join(
            output_dir,
            "stations.json",
        ),
        "w",
    ) as fp:
        json.dump(
            candidates_json,
            fp,
            indent=2,
        )

    print(
        "\n[stations] Candidate metadata saved."
    )

    print(
        "[stations] Final family selection will happen "
        "after real waveform verification."
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
    )

    parser.add_argument(
        "--maxradius",
        type=float,
        default=2.0,
        help="Search radius around the region center, in degrees (default: 2.0)",
    )

    parser.add_argument(
        "--excluded_networks",
        default=",".join(sorted(EXCLUDED_NETWORKS)),
        help="Comma-separated network codes to skip (default: SY = synthetic data)",
    )

    parser.add_argument(
        "--excluded_stations",
        default="",
        help='Comma-separated stations to skip, as NET.STA (e.g. "DK.ILULI,3Z.SK12G"; default: none)',
    )

    parser.add_argument(
        "--min_sample_rate",
        type=float,
        default=MIN_SAMPLE_RATE,
        help=f"Drop channels sampled below this rate, in Hz (default: {MIN_SAMPLE_RATE})",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
    )

    args = parser.parse_args()

    # Command-line options override the module defaults
    EXCLUDED_NETWORKS = {code.strip() for code in args.excluded_networks.split(",") if code.strip()}
    EXCLUDED_STATIONS = {code.strip() for code in args.excluded_stations.split(",") if code.strip()}
    MIN_SAMPLE_RATE = float(args.min_sample_rate)

    download_stations(
        config_json=args.config,
        output_dir=args.output_dir,
        maxradius=args.maxradius,
        plot=args.plot,
    )