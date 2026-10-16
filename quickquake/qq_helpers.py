"""
Unified plotting and visualization helpers for QuickQuake/GAMMA seismic catalogs.

This file was consolidated from:
- qq_helpers.py
- qq_helpers_updated.py
- exploratory functions that were embedded in the notebook

Design intent:
- Keep the notebook thin.
- Put reusable logic here.
- Avoid duplicated helper files.
- Return fig/ax/meta objects instead of hiding state inside the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from bisect import bisect_right
from typing import Optional, Tuple, Sequence, Dict, Any, List

import numpy as np
import pandas as pd


# ============================================================
# SCHEMAS
# ============================================================

@dataclass(frozen=True)
class CatalogCols:
    time: str = "time"
    lat: str = "latitude"
    lon: str = "longitude"
    depth_km: str = "depth_km"
    mag: str = "magnitude"
    event_id: str = "event_id"


@dataclass(frozen=True)
class StationCols:
    station_id: str = "station"
    lat: str = "latitude"
    lon: str = "longitude"
    elev_km: str = "elev_km"


@dataclass(frozen=True)
class PicksCols:
    station_id: str = "id"
    time: str = "timestamp"
    phase: str = "type"
    prob: str = "prob"
    event_id: str = "event_id"
    event_id_mapped: str = "event_id_mapped"


# ============================================================
# FIXED COLOURS PER NETWORK
# ============================================================
# The same network always gets the same colour, in every figure and every project.
# Add a network here to fix its colour; any network not listed gets a colour from
# FALLBACK_COLOURS, chosen from its code (so it is also always the same).
NETWORK_COLOURS: Dict[str, str] = {
    "DK": "#000000",   # Danish national network (GEUS), permanent
    "GE": "#7f7f7f",   # GEOFON
    "IU": "#8c564b",   # Global Seismographic Network
    "3Z": "#1f77b4",   # Jakobshavn glacier arrays (ETH)
    "6H": "#d62728",   # GreenlandLakes2020 (WHOI)
    "YF": "#9467bd",   # Helheim (NYU)
    "9F": "#17becf",   # Isunnguata Sermia nodes
    "6R": "#2ca02c",   # SIIOS 2022
    "9C": "#bcbd22",   # SIIOS 2018
    "6D": "#e377c2",   # Bowdoin 2019
    "7Q": "#ff7f0e",   # GreenDrill
    "AV": "#1f77b4",   # Alaska Volcano Observatory (Pavlof example)
}
FALLBACK_COLOURS = ["#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#e6550d", "#31a354"]


def network_of(station_id: str) -> str:
    """Network code of a station id such as '3Z.SK01L..CH' -> '3Z'."""
    return str(station_id).split(".")[0]


def network_colour(network: str) -> str:
    """Fixed colour of a network (see NETWORK_COLOURS)."""
    if network in NETWORK_COLOURS:
        return NETWORK_COLOURS[network]
    return FALLBACK_COLOURS[sum(ord(ch) for ch in str(network)) % len(FALLBACK_COLOURS)]


def scatter_stations_by_network(ax, lons, lats, station_ids, *, size=125, marker="s", transform=None):
    """Draw stations with one fixed colour per network and one legend entry per network
    (with the number of stations of that network)."""
    ids = list(station_ids)
    nets = [network_of(i) for i in ids]
    lons, lats = np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)
    kw = {} if transform is None else {"transform": transform}
    for net in sorted(set(nets)):
        m = np.array([n == net for n in nets])
        ax.scatter(lons[m], lats[m], s=size, marker=marker, facecolor=network_colour(net),
                   edgecolor="w", linewidth=0.5, label=f"{net} ({int(m.sum())} stations)", **kw)


# ============================================================
# BASIC NORMALIZATION HELPERS
# ============================================================

def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_catalog(
    catalog: pd.DataFrame,
    *,
    catc: CatalogCols = CatalogCols(),
) -> pd.DataFrame:
    """
    Rename common column names to the canonical names defined in CatalogCols.
    Columns that already have the right name are left untouched.
    """
    if not isinstance(catalog, pd.DataFrame):
        raise TypeError("catalog must be a pandas.DataFrame")

    df = catalog.copy()

    rename_map: Dict[str, str] = {}

    time_src = _first_existing_column(
        df, [catc.time, "datetime", "origin_time", "date", "event_time"]
    )
    lat_src = _first_existing_column(
        df, [catc.lat, "lat", "latitude_deg", "y"]
    )
    lon_src = _first_existing_column(
        df, [catc.lon, "lon", "longitude_deg", "x"]
    )
    depth_src = _first_existing_column(
        df, [catc.depth_km, "depth", "depth_km", "z", "depth_km_final"]
    )
    mag_src = _first_existing_column(
        df, [catc.mag, "mag", "magnitude", "ml", "mw"]
    )
    event_src = _first_existing_column(
        df, [catc.event_id, "event_idx", "eventid", "eid", "id"]
    )

    if time_src and time_src != catc.time:
        rename_map[time_src] = catc.time
    if lat_src and lat_src != catc.lat:
        rename_map[lat_src] = catc.lat
    if lon_src and lon_src != catc.lon:
        rename_map[lon_src] = catc.lon
    if depth_src and depth_src != catc.depth_km:
        rename_map[depth_src] = catc.depth_km
    if mag_src and mag_src != catc.mag:
        rename_map[mag_src] = catc.mag
    if event_src and event_src != catc.event_id:
        rename_map[event_src] = catc.event_id

    if rename_map:
        df = df.rename(columns=rename_map)

    # conversiones suaves
    if catc.time in df.columns:
        df[catc.time] = pd.to_datetime(df[catc.time], utc=True, errors="coerce", format="mixed")
    if catc.lat in df.columns:
        df[catc.lat] = pd.to_numeric(df[catc.lat], errors="coerce")
    if catc.lon in df.columns:
        df[catc.lon] = pd.to_numeric(df[catc.lon], errors="coerce")
    if catc.depth_km in df.columns:
        df[catc.depth_km] = pd.to_numeric(df[catc.depth_km], errors="coerce")
    if catc.mag in df.columns:
        df[catc.mag] = pd.to_numeric(df[catc.mag], errors="coerce")
    if catc.event_id in df.columns:
        df[catc.event_id] = pd.to_numeric(df[catc.event_id], errors="coerce").astype("Int64")

    return df


# ============================================================
# I/O HELPERS
# ============================================================


def load_catalog_csv(path: Path | str, *, catc: Optional[CatalogCols] = None) -> pd.DataFrame:
    """
    Load a catalog CSV and normalize common QuickQuake/GAMMA/HypoDD columns.

    Parameters
    ----------
    path
        CSV file path.
    catc
        Optional catalog column schema.

    Returns
    -------
    pandas.DataFrame
        Catalog with canonical time/latitude/longitude/depth_km/magnitude columns
        when those columns can be inferred.
    """
    catc = catc or CatalogCols()
    df = pd.read_csv(path)
    return normalize_catalog(df, catc=catc)


def load_stations_json(path: Path | str) -> pd.DataFrame:
    """
    Load a QuickQuake/GAMMA station JSON into a DataFrame.

    Supported JSON shapes:
    - list[dict]
    - {"stations": list[dict]}
    - dict keyed by station id
    """
    import json

    with open(path, "r") as f:
        stations_obj = json.load(f)

    if isinstance(stations_obj, list):
        df = pd.DataFrame(stations_obj)
    elif isinstance(stations_obj, dict) and "stations" in stations_obj:
        df = pd.DataFrame(stations_obj["stations"])
    elif isinstance(stations_obj, dict):
        df = pd.DataFrame(stations_obj).T.reset_index().rename(columns={"index": "station"})
    else:
        raise ValueError("Formato de STATIONS_JSON no reconocido")

    return df


def ensure_dir(path: Path | str) -> Path:
    """Create an output directory if it does not exist and return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


# ============================================================
# TIME AND MAP DATA HELPERS
# ============================================================

def filter_catalog_time(
    catalog: pd.DataFrame,
    start,
    end,
    cols: Optional[CatalogCols] = None,
) -> pd.DataFrame:
    """
    Return the catalog between [start, end] (UTC). start/end can be str or datetime.
    """
    cols = cols or CatalogCols()
    df = normalize_catalog(catalog, catc=cols)

    if cols.time not in df.columns:
        raise KeyError(f"Time column '{cols.time}' not found in catalog.")

    t = pd.to_datetime(df[cols.time], utc=True, errors="coerce")
    start = pd.to_datetime(start, utc=True)
    end = pd.to_datetime(end, utc=True)

    m = (t >= start) & (t <= end)
    return df.loc[m].copy()


def make_quakes_for_map(
    catalog: pd.DataFrame,
    start,
    end,
    catc: Optional[CatalogCols] = None,
) -> pd.DataFrame:
    """
    Exact column layout expected by the map functions:
      quakes_for_map['lon'], ['lat'], ['depth'], ['mag'], ['datetime']
    """
    catc = catc or CatalogCols()
    df = filter_catalog_time(catalog, start, end, cols=catc)

    out = pd.DataFrame({
        "lon": pd.to_numeric(df[catc.lon], errors="coerce"),
        "lat": pd.to_numeric(df[catc.lat], errors="coerce"),
        "depth": pd.to_numeric(df[catc.depth_km], errors="coerce").fillna(0.0)
                 if catc.depth_km in df.columns else np.zeros(len(df)),
        "mag": pd.to_numeric(df[catc.mag], errors="coerce")
               if catc.mag in df.columns else np.full(len(df), np.nan),
        "datetime": pd.to_datetime(df[catc.time], utc=True, errors="coerce"),
    }).dropna(subset=["lon", "lat", "datetime"])

    return out


def make_stations_wide_for_map(
    stations: pd.DataFrame,
    stc: Optional[StationCols] = None,
) -> pd.DataFrame:
    """
    WIDE format, so this access pattern works:
        stations_t = stations_wide.T
        stations_t['longitude'], stations_t['latitude']
    """
    stc = stc or StationCols()
    df = stations.copy()

    rename_map = {}

    if stc.station_id not in df.columns:
        src = _first_existing_column(df, [stc.station_id, "station_id", "id", "sta", "station"])
        if src is not None:
            rename_map[src] = stc.station_id

    if stc.lon not in df.columns:
        src = _first_existing_column(df, [stc.lon, "longitude", "lon", "x"])
        if src is not None:
            rename_map[src] = stc.lon

    if stc.lat not in df.columns:
        src = _first_existing_column(df, [stc.lat, "latitude", "lat", "y"])
        if src is not None:
            rename_map[src] = stc.lat

    if stc.elev_km not in df.columns:
        src = _first_existing_column(df, [stc.elev_km, "elevation_km", "elev_km", "elev", "elevation", "elevation(m)", "elevation_m"])
        if src is not None:
            rename_map[src] = stc.elev_km

    if rename_map:
        df = df.rename(columns=rename_map)

    # If station elevations come in meters, convert them to km for the canonical column.
    if stc.elev_km in df.columns:
        df[stc.elev_km] = pd.to_numeric(df[stc.elev_km], errors="coerce")
        if "elevation(m)" in stations.columns and stc.elev_km != "elevation(m)":
            df[stc.elev_km] = df[stc.elev_km] / 1000.0

    if stc.station_id not in df.columns:
        raise KeyError(f"stations must have a '{stc.station_id}' column. Columns={list(df.columns)}")
    if stc.lon not in df.columns or stc.lat not in df.columns:
        raise KeyError(f"stations must have '{stc.lon}' and '{stc.lat}' columns. Columns={list(df.columns)}")

    cols = [stc.lon, stc.lat]
    if stc.elev_km in df.columns:
        cols.append(stc.elev_km)

    wide = df.set_index(stc.station_id)[cols].T.copy()
    rename_index = {stc.lon: "longitude", stc.lat: "latitude"}
    if stc.elev_km in wide.index:
        rename_index[stc.elev_km] = "elev_km"
    wide = wide.rename(index=rename_index)

    return wide


def ellipse_lonlat_from_km(lon0, lat0, a_km, b_km, theta_deg=0.0, n=200):
    """
    Ellipse centered on (lon0, lat0), semi-axes a_km/b_km (km),
    rotated by theta_deg (degrees, counter-clockwise). Local km->deg approximation.
    """
    theta = np.deg2rad(float(theta_deg))
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.deg2rad(float(lat0)))

    if np.isclose(km_per_deg_lon, 0.0):
        raise ValueError("km_per_deg_lon is ~0. Check lat0 (too close to a pole).")

    t = np.linspace(0, 2 * np.pi, int(n))
    ex = float(a_km) * np.cos(t)
    ey = float(b_km) * np.sin(t)

    x0 = ex * np.cos(theta) - ey * np.sin(theta)
    y0 = ex * np.sin(theta) + ey * np.cos(theta)

    lon_ell = float(lon0) + x0 / km_per_deg_lon
    lat_ell = float(lat0) + y0 / km_per_deg_lat
    return lon_ell, lat_ell


# ============================================================
# 2D MAP VIEWS
# ============================================================

def plot_mapview_seismicity(
    catalog,
    stations,
    start,
    end,
    *,
    depth_vmin=0.0,
    depth_vmax=40.0,
    cmap_name="gist_rainbow",
    size_k=22.0,
    marker_min=6.0,
    plot_ellipse=False,
    ellipse_lon=None,
    ellipse_lat=None,
    a_km=12.0,
    b_km=6.0,
    theta_deg=0.0,
    relief_zoom=12,
    pad_lon=0.25,
    pad_lat=0.15,
    step_lon=0.5,
    step_lat=0.25,
    labelsize=20,
    figsize=(15, 8),
    savefig=False,
    out_png=None,
    dpi=600,
    region_name=None,
):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    import cartopy.mpl.ticker as cticker
    from cartopy.io import img_tiles as cimgt

    class ShadedReliefESRI(cimgt.GoogleTiles):
        def _image_url(self, tile):
            x, y, z = tile
            return (
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                f"World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg"
            )

    quakes_for_map = make_quakes_for_map(catalog, start, end)
    stations_wide = make_stations_wide_for_map(stations)

    if quakes_for_map.empty:
        raise ValueError("No events in the range [start, end].")

    minlon = float(quakes_for_map["lon"].min()) - float(pad_lon)
    maxlon = float(quakes_for_map["lon"].max()) + float(pad_lon)
    minlat = float(quakes_for_map["lat"].min()) - float(pad_lat)
    maxlat = float(quakes_for_map["lat"].max()) + float(pad_lat)
    extent = (minlon, maxlon, minlat, maxlat)

    cmap = plt.get_cmap(str(cmap_name), 256)
    norm = mpl.colors.Normalize(float(depth_vmin), float(depth_vmax))

    center_lon = (minlon + maxlon) / 2
    center_lat = (minlat + maxlat) / 2

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={
            "projection": ccrs.LambertAzimuthalEqualArea(
                central_longitude=center_lon,
                central_latitude=center_lat
            )
        }
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.98)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_image(ShadedReliefESRI(), int(relief_zoom))

    mag = quakes_for_map["mag"].to_numpy(dtype=float)
    sizes = float(size_k) * mag
    sizes = np.where(np.isfinite(sizes), sizes, float(marker_min))
    sizes = np.clip(sizes, float(marker_min), None)

    sc = ax.scatter(
        quakes_for_map["lon"], quakes_for_map["lat"],
        s=sizes,
        c=quakes_for_map["depth"],
        cmap=cmap, norm=norm,
        edgecolor="w", linewidth=0.3,
        transform=ccrs.PlateCarree(),
        alpha=0.9,
        label="Earthquakes"
    )

    stations_t = stations_wide.T
    scatter_stations_by_network(ax, stations_t["longitude"], stations_t["latitude"], stations_t.index,
                                size=125, marker="s", transform=ccrs.PlateCarree())

  

    if plot_ellipse:
        if ellipse_lon is None:
            ellipse_lon = float(np.nanmedian(quakes_for_map["lon"]))
        if ellipse_lat is None:
            ellipse_lat = float(np.nanmedian(quakes_for_map["lat"]))

        lon_ell, lat_ell = ellipse_lonlat_from_km(
            ellipse_lon, ellipse_lat, a_km, b_km, theta_deg=theta_deg, n=200
        )

        ax.plot(
            lon_ell, lat_ell,
            transform=ccrs.PlateCarree(),
            color="black",
            linestyle="-",
            linewidth=1.5,
            label="Reference ellipse"
        )

    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": labelsize, "color": "black"}
    gl.ylabel_style = {"size": labelsize, "color": "black", "rotation": 90}

    gl.xlocator = mticker.FixedLocator(
        np.arange(np.floor(minlon), np.ceil(maxlon) + 1e-6, float(step_lon))
    )
    gl.ylocator = mticker.FixedLocator(
        np.arange(np.floor(minlat), np.ceil(maxlat) + 1e-6, float(step_lat))
    )
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()

    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.015, aspect=40)
    cbar.set_label("Depth (km)", fontsize=20)
    ticks = np.linspace(float(depth_vmin), float(depth_vmax), 9)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(t)}" for t in ticks])
    cbar.ax.tick_params(labelsize=16)

    ax.legend(
        loc="upper left",
        fontsize=18,
        frameon=True,
        framealpha=0.8,
        borderpad=0.3,
        labelspacing=0.3,
        handlelength=1.0,
        handletextpad=0.4
    )

    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)

    if savefig:
        if out_png is None:
            out_png = f"enhanced_seismicity_{start_dt.date()}_{end_dt.date()}.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

    title = f"{region_name} seismicity" if region_name else "Seismicity"
    print(
        f"{title} map from {start_dt.date()} to {end_dt.date()} "
        "(Size = Magnitude | Color = Depth)"
    )

    
    return fig, ax


def plot_mapview_basemap_stations(
    catalog,
    stations,
    start,
    end,
    *,
    relief_zoom=12,
    pad_lon=0.25,
    pad_lat=0.15,
    step_lon=0.5,
    step_lat=0.25,
    labelsize=20,
    figsize=(15, 8),
    savefig=False,
    out_png=None,
    dpi=600,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    import cartopy.mpl.ticker as cticker
    from cartopy.io import img_tiles as cimgt

    class ShadedReliefESRI(cimgt.GoogleTiles):
        def _image_url(self, tile):
            x, y, z = tile
            return (
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                f"World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg"
            )

    # Use catalog ONLY to keep the exact same map bounds
    quakes_for_map = make_quakes_for_map(catalog, start, end)
    stations_wide = make_stations_wide_for_map(stations)

    if quakes_for_map.empty:
        raise ValueError("No events to compute the map extent.")

    minlon = float(quakes_for_map["lon"].min()) - float(pad_lon)
    maxlon = float(quakes_for_map["lon"].max()) + float(pad_lon)
    minlat = float(quakes_for_map["lat"].min()) - float(pad_lat)
    maxlat = float(quakes_for_map["lat"].max()) + float(pad_lat)
    extent = (minlon, maxlon, minlat, maxlat)

    center_lon = (minlon + maxlon) / 2
    center_lat = (minlat + maxlat) / 2

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={
            "projection": ccrs.LambertAzimuthalEqualArea(
                central_longitude=center_lon,
                central_latitude=center_lat,
            )
        },
    )

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.98)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_image(ShadedReliefESRI(), int(relief_zoom))

    stations_t = stations_wide.T
    scatter_stations_by_network(ax, stations_t["longitude"], stations_t["latitude"], stations_t.index,
                                size=125, marker="s", transform=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": labelsize, "color": "black"}
    gl.ylabel_style = {"size": labelsize, "color": "black", "rotation": 90}

    gl.xlocator = mticker.FixedLocator(
        np.arange(np.floor(minlon), np.ceil(maxlon) + 1e-6, float(step_lon))
    )
    gl.ylocator = mticker.FixedLocator(
        np.arange(np.floor(minlat), np.ceil(maxlat) + 1e-6, float(step_lat))
    )

    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()

    ax.legend(
        loc="upper left",
        fontsize=18,
        frameon=True,
        framealpha=0.8,
        borderpad=0.3,
        labelspacing=0.3,
        handlelength=1.0,
        handletextpad=0.4,
    )

    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)

    if savefig:
        if out_png is None:
            out_png = f"basemap_stations_{start_dt.date()}_{end_dt.date()}.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

    print(f"Basemap from {start_dt.date()} to {end_dt.date()} without seismicity")

    return fig, ax


def plot_mapview_seismicity_zoom(
    catalog,
    start,
    end,
    *,
    depth_vmin=0.0,
    depth_vmax=40.0,
    cmap_name="gist_rainbow",
    size_k=22.0,
    marker_min=6.0,
    plot_ellipse=False,
    ellipse_lon=None,
    ellipse_lat=None,
    a_km=12.0,
    b_km=6.0,
    theta_deg=0.0,
    relief_zoom=12,
    pad_lon=0.25,
    pad_lat=0.15,
    step_lon=0.5,
    step_lat=0.25,
    labelsize=20,
    figsize=(15, 8),
    savefig=False,
    out_png=None,
    dpi=600,
    # nuevo
    zoom_factor=None,
    xlim=None,
    ylim=None,
    focal_length=None,
    show=False,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    import cartopy.mpl.ticker as cticker
    from cartopy.io import img_tiles as cimgt

    class ShadedReliefESRI(cimgt.GoogleTiles):
        def _image_url(self, tile):
            x, y, z = tile
            return (
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                f"World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg"
            )

    quakes_for_map = make_quakes_for_map(catalog, start, end)

    if quakes_for_map.empty:
        raise ValueError("No events in the range [start, end].")

    # Extent base
    minlon_data = float(quakes_for_map["lon"].min())
    maxlon_data = float(quakes_for_map["lon"].max())
    minlat_data = float(quakes_for_map["lat"].min())
    maxlat_data = float(quakes_for_map["lat"].max())

    minlon = minlon_data - float(pad_lon)
    maxlon = maxlon_data + float(pad_lon)
    minlat = minlat_data - float(pad_lat)
    maxlat = maxlat_data + float(pad_lat)

    # Automatic zoom around the center
    if zoom_factor is not None:
        if zoom_factor <= 0:
            raise ValueError("zoom_factor must be > 0")

        clon = 0.5 * (minlon + maxlon)
        clat = 0.5 * (minlat + maxlat)

        half_width = 0.5 * (maxlon - minlon) / float(zoom_factor)
        half_height = 0.5 * (maxlat - minlat) / float(zoom_factor)

        minlon = clon - half_width
        maxlon = clon + half_width
        minlat = clat - half_height
        maxlat = clat + half_height

    # Zoom manual domina si se pasa
    if xlim is not None:
        minlon, maxlon = xlim
    if ylim is not None:
        minlat, maxlat = ylim

    extent = (minlon, maxlon, minlat, maxlat)

    cmap = plt.get_cmap(str(cmap_name), 256)
    norm = mpl.colors.Normalize(float(depth_vmin), float(depth_vmax))

    center_lon = (minlon + maxlon) / 2
    center_lat = (minlat + maxlat) / 2

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={
            "projection": ccrs.LambertAzimuthalEqualArea(
                central_longitude=center_lon,
                central_latitude=center_lat
            )
        }
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.98)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_image(ShadedReliefESRI(), int(relief_zoom))

    mag = quakes_for_map["mag"].to_numpy(dtype=float)
    sizes = float(size_k) * mag
    sizes = np.where(np.isfinite(sizes), sizes, float(marker_min))
    sizes = np.clip(sizes, float(marker_min), None)

    sc = ax.scatter(
        quakes_for_map["lon"], quakes_for_map["lat"],
        s=sizes,
        c=quakes_for_map["depth"],
        cmap=cmap, norm=norm,
        edgecolor="w", linewidth=0.3,
        transform=ccrs.PlateCarree(),
        alpha=0.9,
        label="Earthquakes"
    )

    if plot_ellipse:
        if ellipse_lon is None:
            ellipse_lon = float(np.nanmedian(quakes_for_map["lon"]))
        if ellipse_lat is None:
            ellipse_lat = float(np.nanmedian(quakes_for_map["lat"]))

        lon_ell, lat_ell = ellipse_lonlat_from_km(
            ellipse_lon, ellipse_lat, a_km, b_km, theta_deg=theta_deg, n=200
        )

        ax.plot(
            lon_ell, lat_ell,
            transform=ccrs.PlateCarree(),
            color="black",
            linestyle="-",
            linewidth=1.5,
            label="Ellipse"
        )

    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": labelsize, "color": "black"}
    gl.ylabel_style = {"size": labelsize, "color": "black"}

    gl.xlocator = mticker.FixedLocator(
        np.arange(np.floor(minlon), np.ceil(maxlon) + 1e-6, float(step_lon))
    )
    gl.ylocator = mticker.FixedLocator(
        np.arange(np.floor(minlat), np.ceil(maxlat) + 1e-6, float(step_lat))
    )
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()

    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.015, aspect=40)
    cbar.set_label("Depth (km)", fontsize=20)
    ticks = np.linspace(float(depth_vmin), float(depth_vmax), 9)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(t)}" for t in ticks])
    cbar.ax.tick_params(labelsize=16)

    ax.legend(
        loc="upper left",
        fontsize=18,
        frameon=True,
        framealpha=0.8,
        borderpad=0.3,
        labelspacing=0.3,
        handlelength=1.0,
        handletextpad=0.4
    )


    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)

    if savefig:
        if out_png is None:
            out_png = f"enhanced_seismicity_zoom_{start_dt.date()}_{end_dt.date()}.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

    print(
        f"Seismicity map from {start_dt.date()} to {end_dt.date()} "
        "(Size = Magnitude | Color = Depth)"
    )

    if show:
        plt.show()

    meta = {
        "bounds": (minlon, maxlon, minlat, maxlat),
        "center": (center_lon, center_lat),
    }

    return fig, ax, meta


def plot_mapview_seismicity_zoom_2(
    catalog,
    start,
    end,
    *,
    depth_vmin=0.0,
    depth_vmax=40.0,
    cmap_name="gist_rainbow",
    size_k=22.0,
    marker_min=6.0,
    plot_ellipse=False,
    ellipse_lon=None,
    ellipse_lat=None,
    a_km=12.0,
    b_km=6.0,
    theta_deg=0.0,
    relief_zoom=12,
    pad_lon=0.25,
    pad_lat=0.15,
    step_lon=0.5,
    step_lat=0.25,
    labelsize=20,
    figsize=(15, 8),
    savefig=False,
    out_png=None,
    dpi=600,
    zoom_factor=None,
    xlim=None,
    ylim=None,
    focal_length=None,
    show=False,
    # nuevo
    depth_filter_min=None,
    depth_filter_max=None,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    import cartopy.mpl.ticker as cticker
    from cartopy.io import img_tiles as cimgt

    class ShadedReliefESRI(cimgt.GoogleTiles):
        def _image_url(self, tile):
            x, y, z = tile
            return (
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                f"World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg"
            )

    # filter by time first
    quakes_for_map = make_quakes_for_map(catalog, start, end)

    if quakes_for_map.empty:
        raise ValueError("No events in the range [start, end].")

    # -------------------------------
    # Depth filter
    # -------------------------------
    if depth_filter_min is not None:
        quakes_for_map = quakes_for_map[
            quakes_for_map["depth"] >= float(depth_filter_min)
        ].copy()

    if depth_filter_max is not None:
        quakes_for_map = quakes_for_map[
            quakes_for_map["depth"] <= float(depth_filter_max)
        ].copy()

    if quakes_for_map.empty:
        raise ValueError(
            "No events left after the depth filter. "
            "Check depth_filter_min/depth_filter_max."
        )

    # Extent base
    minlon_data = float(quakes_for_map["lon"].min())
    maxlon_data = float(quakes_for_map["lon"].max())
    minlat_data = float(quakes_for_map["lat"].min())
    maxlat_data = float(quakes_for_map["lat"].max())

    minlon = minlon_data - float(pad_lon)
    maxlon = maxlon_data + float(pad_lon)
    minlat = minlat_data - float(pad_lat)
    maxlat = maxlat_data + float(pad_lat)

    # Automatic zoom around the center
    if zoom_factor is not None:
        if zoom_factor <= 0:
            raise ValueError("zoom_factor must be > 0")

        clon = 0.5 * (minlon + maxlon)
        clat = 0.5 * (minlat + maxlat)

        half_width = 0.5 * (maxlon - minlon) / float(zoom_factor)
        half_height = 0.5 * (maxlat - minlat) / float(zoom_factor)

        minlon = clon - half_width
        maxlon = clon + half_width
        minlat = clat - half_height
        maxlat = clat + half_height

    # Zoom manual domina si se pasa
    if xlim is not None:
        minlon, maxlon = xlim
    if ylim is not None:
        minlat, maxlat = ylim

    extent = (minlon, maxlon, minlat, maxlat)

    cmap = plt.get_cmap(str(cmap_name), 256)
    norm = mpl.colors.Normalize(float(depth_vmin), float(depth_vmax))

    center_lon = (minlon + maxlon) / 2
    center_lat = (minlat + maxlat) / 2

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={
            "projection": ccrs.LambertAzimuthalEqualArea(
                central_longitude=center_lon,
                central_latitude=center_lat
            )
        }
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.98)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_image(ShadedReliefESRI(), int(relief_zoom))

    mag = quakes_for_map["mag"].to_numpy(dtype=float)
    sizes = float(size_k) * mag
    sizes = np.where(np.isfinite(sizes), sizes, float(marker_min))
    sizes = np.clip(sizes, float(marker_min), None)

    sc = ax.scatter(
        quakes_for_map["lon"], quakes_for_map["lat"],
        s=sizes,
        c=quakes_for_map["depth"],
        cmap=cmap, norm=norm,
        edgecolor="w", linewidth=0.3,
        transform=ccrs.PlateCarree(),
        alpha=0.9,
        label="Earthquakes"
    )

    if plot_ellipse:
        if ellipse_lon is None:
            ellipse_lon = float(np.nanmedian(quakes_for_map["lon"]))
        if ellipse_lat is None:
            ellipse_lat = float(np.nanmedian(quakes_for_map["lat"]))

        lon_ell, lat_ell = ellipse_lonlat_from_km(
            ellipse_lon, ellipse_lat, a_km, b_km, theta_deg=theta_deg, n=200
        )

        ax.plot(
            lon_ell, lat_ell,
            transform=ccrs.PlateCarree(),
            color="black",
            linestyle="-",
            linewidth=1.5,
            label="Ellipse"
        )

    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": labelsize, "color": "black"}
    gl.ylabel_style = {"size": labelsize, "color": "black"}

    gl.xlocator = mticker.FixedLocator(
        np.arange(np.floor(minlon), np.ceil(maxlon) + 1e-6, float(step_lon))
    )
    gl.ylocator = mticker.FixedLocator(
        np.arange(np.floor(minlat), np.ceil(maxlat) + 1e-6, float(step_lat))
    )
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()

    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.015, aspect=40)
    cbar.set_label("Depth (km)", fontsize=20)
    ticks = np.linspace(float(depth_vmin), float(depth_vmax), 9)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(t)}" for t in ticks])
    cbar.ax.tick_params(labelsize=16)

    ax.legend(
        loc="upper left",
        fontsize=18,
        frameon=True,
        framealpha=0.8,
        borderpad=0.3,
        labelspacing=0.3,
        handlelength=1.0,
        handletextpad=0.4
    )

    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)

    if savefig:
        if out_png is None:
            out_png = f"enhanced_seismicity_zoom_{start_dt.date()}_{end_dt.date()}.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

    print(
        f"Seismicity map from {start_dt.date()} to {end_dt.date()} "
        f"(Size = Magnitude | Color = Depth | "
        f"Depth filter = [{depth_filter_min}, {depth_filter_max}])"
    )

    if show:
        plt.show()

    meta = {
        "bounds": (minlon, maxlon, minlat, maxlat),
        "center": (center_lon, center_lat),
        "n_events": int(len(quakes_for_map)),
        "depth_filter_min": depth_filter_min,
        "depth_filter_max": depth_filter_max,
    }

    return fig, ax, meta


# ============================================================
# TIMELINE PLOTS
# ============================================================

def _to_naive_utc_dt(x):
    """Convierte timestamp (str/datetime) a datetime naive en UTC."""
    t = pd.to_datetime(x, utc=True, errors="coerce")
    if pd.isna(t):
        return None
    return t.tz_convert("UTC").tz_localize(None)


def _series_to_naive_utc(series):
    """Serie a datetime naive UTC."""
    s = pd.to_datetime(series, utc=True, errors="coerce")
    return s.dt.tz_convert("UTC").dt.tz_localize(None)


def plot_timeline_histogram(
    catalog: pd.DataFrame,
    *,
    start,
    end,
    spans: Sequence[Tuple[str, str, str]],
    time_col: Optional[str] = None,
    fi_col: str = "FI",
    fi_categories: Sequence[str] = ("VT", "LP", "NS"),
    fi_colors: Optional[Dict[str, str]] = None,
    original_folder_col: str = "original_folder",
    original_folder_fmt: str = "%Y%m%dT%H%M%S",
    n_bins: int = 90,
    figsize: Tuple[float, float] = (14, 4),
    base_fontsize: int = 15,
    xtick_fontsize: int = 18,
    xlabel_fontsize: int = 20,
    legend_fontsize: int = 15,
    hist_color: str = "black",
    band_cmap: str = "Set2",
    month_interval: int = 1,
    band_y0: float = 0.78,
    band_height: float = 0.06,
    band_gap: float = 0.01,
    show_legend: bool = False,
    savefig: bool = False,
    out_png: Optional[str] = None,
    dpi: int = 600,
):
    """
    Timeline con:
      - Histogram stacked by FI class (VT, LP, NS)
      - Stacked bands (axvspan) per type on top
      - Monthly ticks, rotated 45 degrees
      - Optional legend
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches

    plt.rcParams.update({"font.size": int(base_fontsize)})

    if fi_colors is None:
        fi_colors = {
            "VT": "#1f77b4",
            "LP": "#d62728",
            "NS": "#7f7f7f",
        }

    df = catalog.copy()

    # -------------------------
    # tiempo
    # -------------------------
    if time_col is not None and time_col in df.columns:
        tseries = df[time_col]
    elif "datetime" in df.columns:
        tseries = df["datetime"]
    else:
        catc = CatalogCols()
        if catc.time in df.columns:
            tseries = df[catc.time]
        elif original_folder_col in df.columns:
            tseries = pd.to_datetime(
                df[original_folder_col].astype(str),
                format=original_folder_fmt,
                errors="coerce"
            )
        else:
            raise KeyError(
                "No time column found. Pass time_col=... or provide "
                f"'datetime' o '{original_folder_col}'."
            )

    df["__t__"] = _series_to_naive_utc(tseries)
    df = df.dropna(subset=["__t__"])

    start_dt = _to_naive_utc_dt(start)
    end_dt = _to_naive_utc_dt(end)
    if start_dt is None or end_dt is None:
        raise ValueError("start/end could not be converted to dates.")

    df = df[(df["__t__"] >= start_dt) & (df["__t__"] <= end_dt)].copy()

    # -------------------------
    # columna FI
    # -------------------------
    # If the catalog has no FI/event-type column, fall back to one generic
    # "EVENTS" category instead of failing. This is useful for QuickQuake/GAMMA
    # catalogs that only have time/lat/lon/depth/magnitude.
    if fi_col is None or fi_col not in df.columns:
        fi_col = "__event_type__"
        df[fi_col] = "EVENTS"
        fi_categories = ("EVENTS",)
        fi_colors = {"EVENTS": hist_color}
    else:
        df[fi_col] = df[fi_col].astype(str).str.strip().str.upper()

    # Ensure every requested FI category has a color.
    fi_colors = dict(fi_colors or {})
    for _fi in fi_categories:
        fi_colors.setdefault(_fi, hist_color)

    # -------------------------
    # colores spans superiores
    # -------------------------
    types = list(dict.fromkeys(desc for _, _, desc in spans))
    cmap = plt.get_cmap(band_cmap)
    palette = getattr(cmap, "colors", None)
    if palette is None:
        palette = [cmap(i) for i in np.linspace(0, 1, max(1, len(types)))]

    color_map = {t: palette[i % len(palette)] for i, t in enumerate(types)}

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    # -------------------------
    # histogram stacked by FI class
    # -------------------------
    t_num = mdates.date2num(df["__t__"])
    bin_edges = np.linspace(mdates.date2num(start_dt), mdates.date2num(end_dt), int(n_bins) + 1)
    bin_widths = np.diff(bin_edges)

    bottom = np.zeros(int(n_bins), dtype=float)
    fi_handles = []
    counts_by_fi = {}

    for fi in fi_categories:
        vals = t_num[df[fi_col] == fi]
        counts, _ = np.histogram(vals, bins=bin_edges)
        counts_by_fi[fi] = counts

        ax.bar(
            bin_edges[:-1],
            counts,
            width=bin_widths,
            bottom=bottom,
            align="edge",
            color=fi_colors.get(fi, hist_color),
            edgecolor="black",
            linewidth=0.4,
            alpha=0.95,
            label=fi,
        )

        bottom = bottom + counts
        fi_handles.append(mpatches.Patch(color=fi_colors.get(fi, hist_color), label=fi))

    total_counts = bottom.copy()

    ax.set_ylabel("Earthquakes")
    ax.grid(axis="y", alpha=0.3)

    ymax = float(np.max(total_counts)) if len(total_counts) else 1.0
    ax.set_ylim(0, ymax * 1.3)

    # -------------------------
    # bandas superiores
    # -------------------------
    band_pos = {}
    for i, t in enumerate(types):
        ymin = band_y0 + i * (band_height + band_gap)
        ymax_band = ymin + band_height
        band_pos[t] = (ymin, ymax_band)

    seen = set()
    span_handles_for_meta = []
    for t0, t1, desc in spans:
        t0_dt = _to_naive_utc_dt(t0)
        t1_dt = _to_naive_utc_dt(t1)
        if t0_dt is None or t1_dt is None:
            continue

        ymin_band, ymax_band = band_pos[desc]
        label = desc if desc not in seen else "_nolegend_"

        ax.axvspan(
            t0_dt, t1_dt,
            ymin=ymin_band, ymax=ymax_band,
            transform=ax.get_xaxis_transform(),
            color=color_map[desc],
            alpha=0.9,
            label=label,
        )

        if desc not in seen:
            span_handles_for_meta.append(mpatches.Patch(color=color_map[desc], label=desc))
        seen.add(desc)

    # -------------------------
    # ejes x
    # -------------------------
    ax.set_xlim(start_dt, end_dt)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=int(month_interval)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    for lab in ax.get_xticklabels():
        lab.set_rotation(45)
        lab.set_ha("right")
        lab.set_fontsize(int(xtick_fontsize))

    fig.subplots_adjust(bottom=0.25)
    ax.set_xlabel("Date", fontsize=int(xlabel_fontsize))

    # -------------------------
    # optional legend
    # -------------------------
    legend_handles = fi_handles + span_handles_for_meta
    legend_labels = [h.get_label() for h in legend_handles]

    if show_legend:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            fontsize=int(legend_fontsize),
            frameon=False,
            ncol=1,
        )

    plt.tight_layout()

    if savefig:
        if out_png is None:
            out_png = "timeline_hist_combined.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

    meta = {
        "n_events": int(len(df)),
        "types": types,
        "color_map": color_map,
        "fi_categories": list(fi_categories),
        "fi_colors": fi_colors,
        "counts_by_fi": counts_by_fi,
        "total_counts": total_counts,
        "legend_handles": legend_handles,
        "legend_labels": legend_labels,
    }
    return fig, ax, meta


def plot_timeline_legend_only(
    *,
    fi_categories: Sequence[str],
    fi_colors: Dict[str, str],
    span_types: Sequence[str],
    span_color_map: Dict[str, str],
    figsize: Tuple[float, float] = (4, 3),
    legend_fontsize: int = 16,
    ncol: int = 1,
    title: Optional[str] = None,
    savefig: bool = False,
    out_png: Optional[str] = None,
    dpi: int = 600,
):
    """
    Separate figure that contains only the legend
    for FI classes + spans.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    handles = []

    for fi in fi_categories:
        handles.append(mpatches.Patch(color=fi_colors[fi], label=fi))

    for desc in span_types:
        handles.append(mpatches.Patch(color=span_color_map[desc], label=desc))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        fontsize=int(legend_fontsize),
        ncol=int(ncol),
    )

    if title is not None:
        ax.set_title(title, fontsize=int(legend_fontsize) + 1)

    plt.tight_layout()

    if savefig:
        if out_png is None:
            out_png = "timeline_legend_only.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight", transparent=True)
        print(f"[ok] saved: {out_png}")

    return fig, ax


# ============================================================
# 3D MAP VIEWS
# ============================================================

def plot_catalog_3d_topo_seismicity(
    catalog: pd.DataFrame,
    *,
    catc: CatalogCols = CatalogCols(),
    mode: str = "time",              # "time" | "year_month"
    year: Optional[int] = None,      # required if mode="year_month"
    dem_res: str = "15s",
    topo_exagg: float = 5.0,
    topo_opacity: float = 0.98,
    colorscale: str = "Spectral",
    show: bool = False,
    renderer: Optional[str] = "notebook_connected",
    save_image: Optional[Path] = None,
    image_format: str = "svg",
    image_width: int = 1200,
    image_height: int = 900,
    image_engine: str = "kaleido",
    region_name: Optional[str] = None,
):
    """
    3D catalog plot (region_name is used in the title, e.g. "Pavlof Volcano"):
      - mode="year_month": filter by year and color by month
      - mode="time": color by time (old -> new)
      - DEM (pygmt) + Surface (plotly) + Scatter3d (plotly)
      - interactive if the renderer is set correctly
      - optional: fig.show() and fig.write_image()
    """
    import plotly.graph_objects as go
    import pygmt

    df = normalize_catalog(catalog, catc=catc)
    df = df.dropna(subset=[catc.time, catc.lat, catc.lon]).copy()

    mode = str(mode).lower().strip()
    if mode not in {"time", "year_month"}:
        raise ValueError("mode must be 'time' or 'year_month'.")

    if mode == "year_month":
        if year is None:
            raise ValueError("mode='year_month' needs year=YYYY.")
        df = df[df[catc.time].dt.year == int(year)].copy()
        df["month"] = df[catc.time].dt.month.astype(int)

    if df.empty:
        raise ValueError("No events left after filtering (check year or NaN values).")

    minlon = float(df[catc.lon].min())
    maxlon = float(df[catc.lon].max())
    minlat = float(df[catc.lat].min())
    maxlat = float(df[catc.lat].max())
    region = [minlon, maxlon, minlat, maxlat]

    grid = pygmt.datasets.load_earth_relief(dem_res, region=region)
    lon = grid["lon"].values
    lat = grid["lat"].values
    elev_km = grid.values / 1000.0

    Lon, Lat = np.meshgrid(lon, lat)
    Z = -elev_km * float(topo_exagg)

    surface = go.Surface(
        x=Lon,
        y=Lat,
        z=Z,
        colorscale="earth",
        showscale=False,
        opacity=float(topo_opacity),
        name="Topography",
    )

    if catc.mag in df.columns:
        mags = pd.to_numeric(df[catc.mag], errors="coerce")
        mmin, mmax = float(mags.min()), float(mags.max())
        if np.isfinite(mmin) and np.isfinite(mmax) and (mmax > mmin):
            sizes = 1 + (mags - mmin) / (mmax - mmin) * 15
        else:
            sizes = np.full(len(df), 6.0)
    else:
        sizes = np.full(len(df), 6.0)

    if mode == "year_month":
        color_vals = df["month"].to_numpy()
        cmin, cmax = 1, 12
        cbar = dict(
            title=f"Month ({int(year)})",
            tickvals=list(range(1, 13)),
            ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        )
        title = f"{region_name} – {int(year)}" if region_name else f"Seismicity – {int(year)}"
    else:
        tsec = (df[catc.time].astype("int64") / 1e9).to_numpy()
        color_vals = tsec
        cmin, cmax = float(np.min(tsec)), float(np.max(tsec))

        qs = np.linspace(0, 1, 6)
        tickvals = [float(np.quantile(tsec, q)) for q in qs]
        ticktext = [
            pd.to_datetime(v, unit="s", utc=True).strftime("%Y-%m-%d<br>%H:%M")
            for v in tickvals
        ]
        cbar = dict(title="Time (old → new)", tickvals=tickvals, ticktext=ticktext)
        title = f"{region_name} – time colored" if region_name else "Seismicity – time colored"

    if catc.depth_km in df.columns:
        z_depth = pd.to_numeric(df[catc.depth_km], errors="coerce").fillna(0.0).to_numpy()
    else:
        z_depth = np.zeros(len(df))

    scatter = go.Scatter3d(
        x=df[catc.lon],
        y=df[catc.lat],
        z=z_depth,
        mode="markers",
        marker=dict(
            size=sizes,
            color=color_vals,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            colorbar=cbar,
            line=dict(width=0.2, color="black"),
            opacity=0.9,
        ),
        name="Earthquakes",
    )

    r = np.sqrt(1.5**2 + 1.5**2 + 1.2**2)
    az_rad = np.deg2rad(310)
    el_rad = np.deg2rad(15)
    camera = {
        "center": {"x": 0, "y": 0, "z": 0},
        "eye": {
            "x": float(r * np.cos(el_rad) * np.cos(az_rad)),
            "y": float(r * np.cos(el_rad) * np.sin(az_rad)),
            "z": float(r * np.sin(el_rad)),
        },
        "up": {"x": 0, "y": 0, "z": 1},
    }

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Depth (km)",
            zaxis=dict(autorange="reversed"),
            aspectmode="auto",
            camera=camera,
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )

    fig = go.Figure(data=[surface, scatter], layout=layout)

    if show:
        if renderer is None:
            fig.show()
        else:
            fig.show(renderer=renderer)

    if save_image is not None:
        fig.write_image(
            str(save_image),
            format=image_format,
            width=int(image_width),
            height=int(image_height),
            engine=image_engine,
        )

    return fig


def plot_mapview3d_months(
    catalog: pd.DataFrame,
    *,
    start,
    end,
    depth_max: float = 40.0,
    dem_res: str = "15s",
    exagg: float = 4.0,
    topo_cmap: str = "gist_earth_r",
    quake_cmap: str = "magma",
    size_k: float = 15.0,
    topo_alpha: float = 0.95,
    topo_stride: int = 5,
    view_elev: float = 4.0,
    view_azim: float = 34.0,
    camera_dist: Optional[float] = None,   # <-- nuevo
    n_xticks: int = 5,
    tick_fontsize: int = 13,
    cbar_tick_fontsize: int = 18,
    figsize: Tuple[float, float] = (12, 9),
    savefig: bool = False,
    out_png: Optional[str] = None,
    dpi: int = 800,
    catc: Optional[CatalogCols] = None,
):
    """
    3D figure:
      - PyGMT topography (earth_relief) -> 3D surface
      - Seismicity: x=lon, y=lat, z=depth (inverted)
      - Color: month (1-12)
      - Size: size_k * mag
      - Filters: [start, end) and depth <= depth_max

    Parameter:
      camera_dist : float | None
          Camera distance in Matplotlib 3D.
          Smaller value = more visual zoom.
          Examples:
              6  -> moderate zoom
              3  -> strong zoom

    Note:
      - This does NOT change the data or the region.
      - It only changes the viewing perspective.
      - Depends on Matplotlib and may not behave perfectly.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pygmt
    from matplotlib.colors import Normalize

    if catc is None:
        catc = CatalogCols()

    df = normalize_catalog(catalog, catc=catc)
    df = df.dropna(subset=[catc.time, catc.lat, catc.lon]).copy()

    t = pd.to_datetime(df[catc.time], utc=True, errors="coerce")
    df = df[pd.notna(t)].copy()
    df["_t"] = t

    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)

    df = df[(df["_t"] >= start_dt) & (df["_t"] < end_dt)].copy()

    depth_col = catc.depth_km if catc.depth_km in df.columns else None
    if depth_col is None:
        df["depth_km"] = 0.0
        depth_col = "depth_km"
    else:
        df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce").fillna(0.0)

    df = df[df[depth_col] <= float(depth_max)].copy()

    mag_col = catc.mag if catc.mag in df.columns else None
    if mag_col is None:
        df["magnitude"] = 1.0
        mag_col = "magnitude"
    else:
        df[mag_col] = pd.to_numeric(df[mag_col], errors="coerce").fillna(1.0)

    df["month"] = df["_t"].dt.month.astype(int)

    print(f"Number of events after filters (time, depth<={depth_max} km): {len(df)}")

    if df.empty:
        raise ValueError("No events left after filtering (check start/end/depth_max).")

    minlon = float(df[catc.lon].min())
    maxlon = float(df[catc.lon].max())
    minlat = float(df[catc.lat].min())
    maxlat = float(df[catc.lat].max())
    region = [minlon, maxlon, minlat, maxlat]

    grid = pygmt.datasets.load_earth_relief(dem_res, region=region)
    lon = grid["lon"].values
    lat = grid["lat"].values
    elev_km = grid.values / 1000.0

    Lon, Lat = np.meshgrid(lon, lat)
    Z = -elev_km * float(exagg)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.10, right=0.90)

    ax.plot_surface(
        Lon, Lat, Z,
        cmap=topo_cmap,
        rstride=int(topo_stride),
        cstride=int(topo_stride),
        linewidth=0,
        antialiased=True,
        alpha=float(topo_alpha),
    )

    sc = ax.scatter(
        df[catc.lon].astype(float),
        df[catc.lat].astype(float),
        df[depth_col].astype(float),
        c=df["month"].to_numpy(),
        cmap=quake_cmap,
        norm=Normalize(vmin=1, vmax=12),
        s=sizes,
        edgecolor="k",
        linewidth=0.2,
        alpha=1.0,
        label="Earthquakes",
    )

    ax.tick_params(axis="x", labelsize=int(tick_fontsize), labelrotation=0)
    ax.tick_params(axis="y", labelsize=int(tick_fontsize) + 1, labelrotation=0)
    ax.tick_params(axis="z", labelsize=int(tick_fontsize) + 1, labelrotation=0)

    xticks = np.linspace(minlon, maxlon, int(n_xticks))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.1f}" for x in xticks])

    ax.set_zlim(float(np.nanmin(Z)), float(depth_max))
    ax.invert_zaxis()

    cbar = fig.colorbar(
        sc,
        ax=ax,
        pad=1e-14,
        fraction=0.02,
        shrink=0.4,
        aspect=30,
    )
    cbar.ax.tick_params(labelsize=int(cbar_tick_fontsize))
    months = np.arange(1, 13)
    cbar.set_ticks(months)
    cbar.set_ticklabels([
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ])

    ax.view_init(elev=float(view_elev), azim=float(view_azim))

    # Visual zoom through camera distance
    if camera_dist is not None:
        try:
            ax.dist = float(camera_dist)
        except Exception as e:
            print(f"[warn] No se pudo aplicar camera_dist={camera_dist}: {e}")

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.5, 0.5, 0.5, 0.99)
        axis._axinfo["grid"]["linewidth"] = 0.20
        axis._axinfo["grid"]["linestyle"] = "--"

    if savefig:
        if out_png is None:
            out_png = "mapview3d_months.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

    meta: Dict[str, Any] = {
        "n_events": int(len(df)),
        "region": region,
        "bounds": (minlon, maxlon, minlat, maxlat),
        "dem_res": dem_res,
        "depth_max": float(depth_max),
        "camera_dist": camera_dist,
    }

    return fig, ax, meta


def plot_mapview3d_timecolor(
    catalog: pd.DataFrame,
    *,
    start,
    end,
    depth_max: float = 40.0,
    dem_res: str = "15s",
    exagg: float = 4.0,
    topo_cmap: str = "gist_earth_r",
    quake_cmap: str = "magma",
    size_k: float = 15.0,
    topo_alpha: float = 0.95,
    topo_stride: int = 5,
    view_elev: float = 4.0,
    view_azim: float = 34.0,
    n_xticks: int = 5,
    n_yticks: int = 5,
    tick_fontsize: int = 13,
    cbar_tick_fontsize: int = 18,
    figsize: Tuple[float, float] = (12, 9),
    savefig: bool = False,
    out_png: Optional[str] = None,
    dpi: int = 800,
    catc: Optional["CatalogCols"] = None,
):
    """
    3D figure:
      - PyGMT topography (earth_relief) -> 3D surface
      - Seismicity: x=lon, y=lat, z=depth (inverted)
      - Color: continuous time
      - Size: size_k * mag
      - Filters: [start, end) and depth <= depth_max
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pygmt

    if catc is None:
        catc = CatalogCols()

    df = normalize_catalog(catalog, catc=catc)
    df = df.dropna(subset=[catc.time, catc.lat, catc.lon]).copy()

    t = pd.to_datetime(df[catc.time], utc=True, errors="coerce")
    df = df[pd.notna(t)].copy()
    df["_t"] = t

    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)

    df = df[(df["_t"] >= start_dt) & (df["_t"] < end_dt)].copy()

    depth_col = catc.depth_km if catc.depth_km in df.columns else None
    if depth_col is None:
        df["depth_km"] = 0.0
        depth_col = "depth_km"
    else:
        df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce").fillna(0.0)

    df = df[df[depth_col] <= float(depth_max)].copy()

    mag_col = catc.mag if catc.mag in df.columns else None
    if mag_col is None:
        df["magnitude"] = 1.0
        mag_col = "magnitude"
    else:
        df[mag_col] = pd.to_numeric(df[mag_col], errors="coerce").fillna(1.0)

    print(f"Number of events after filters (time, depth<={depth_max} km): {len(df)}")

    if df.empty:
        raise ValueError("No events left after filtering (check start/end/depth_max).")

    # Continuous time for coloring
    df["_t_num"] = mdates.date2num(df["_t"].dt.to_pydatetime())

    minlon = float(df[catc.lon].min())
    maxlon = float(df[catc.lon].max())
    minlat = float(df[catc.lat].min())
    maxlat = float(df[catc.lat].max())
    region = [minlon, maxlon, minlat, maxlat]

    grid = pygmt.datasets.load_earth_relief(dem_res, region=region)
    lon = grid["lon"].values
    lat = grid["lat"].values
    elev_km = grid.values / 1000.0

    Lon, Lat = np.meshgrid(lon, lat)
    Z = -elev_km * float(exagg)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.10, right=0.90)

    ax.plot_surface(
        Lon, Lat, Z,
        cmap=topo_cmap,
        rstride=int(topo_stride),
        cstride=int(topo_stride),
        linewidth=0,
        antialiased=True,
        alpha=float(topo_alpha),
    )

    sc = ax.scatter(
        df[catc.lon].astype(float),
        df[catc.lat].astype(float),
        df[depth_col].astype(float),
        c=df["_t_num"].to_numpy(),
        cmap=quake_cmap,
        s=sizes,
        edgecolor="k",
        linewidth=0.2,
        alpha=1.0,
        label="Earthquakes",
    )

    ax.tick_params(axis="x", labelsize=int(tick_fontsize), labelrotation=0)
    ax.tick_params(axis="y", labelsize=int(tick_fontsize) + 1, labelrotation=0)
    ax.tick_params(axis="z", labelsize=int(tick_fontsize) + 1, labelrotation=0)

    xticks = np.linspace(minlon, maxlon, int(n_xticks))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.1f}" for x in xticks])

    yticks = np.linspace(minlat, maxlat, int(n_yticks))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks])

    ax.set_zlim(float(np.nanmin(Z)), float(depth_max))
    ax.invert_zaxis()

    cbar = fig.colorbar(
        sc,
        ax=ax,
        pad=1e-14,
        fraction=0.02,
        shrink=0.4,
        aspect=30,
    )
    cbar.ax.tick_params(labelsize=int(cbar_tick_fontsize))
    cbar.set_label("Time", fontsize=int(cbar_tick_fontsize))

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    cbar.ax.yaxis.set_major_locator(locator)
    cbar.ax.yaxis.set_major_formatter(formatter)

    ax.view_init(elev=float(view_elev), azim=float(view_azim))

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.5, 0.5, 0.5, 0.99)
        axis._axinfo["grid"]["linewidth"] = 0.20
        axis._axinfo["grid"]["linestyle"] = "--"

    if savefig:
        if out_png is None:
            out_png = "mapview3d_timecolor.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

    meta: Dict[str, Any] = {
        "n_events": int(len(df)),
        "region": region,
        "bounds": (minlon, maxlon, minlat, maxlat),
        "dem_res": dem_res,
        "depth_max": float(depth_max),
        "time_min": str(df["_t"].min()),
        "time_max": str(df["_t"].max()),
    }

    return fig, ax, meta


def plot_mapview3d_groups(
    catalog: pd.DataFrame,
    *,
    start,
    end,
    depth_max: float = 40.0,
    dem_res: str = "15s",
    exagg: float = 4.0,
    topo_cmap: str = "gist_earth_r",
    size_k: float = 15.0,
    topo_alpha: float = 0.95,
    topo_stride: int = 5,
    view_elev: float = 4.0,
    view_azim: float = 34.0,
    n_xticks: int = 5,
    n_yticks: int = 5,
    tick_fontsize: int = 13,
    figsize: Tuple[float, float] = (12, 9),
    savefig: bool = False,
    out_png: Optional[str] = None,
    dpi: int = 800,
    catc: Optional["CatalogCols"] = None,
    hollow_quakes: bool = True,
    quake_edge_lw: float = 0.7,
    group1_xyz: Optional[np.ndarray] = None,
    group2_xyz: Optional[np.ndarray] = None,
    group1_edgecolor: str = "darkred",
    group2_edgecolor: str = "navy",
    other_edgecolor: str = "black",
    plane_polys: Optional[list] = None,
):
    """
    3D plot of topography + seismicity + planes.

    Earthquake edge colors:
      - group1_xyz -> group1_edgecolor
      - group2_xyz -> group2_edgecolor
      - others     -> black

    Earthquakes can be drawn hollow (no fill).
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pygmt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if catc is None:
        catc = CatalogCols()

    df = normalize_catalog(catalog, catc=catc)
    df = df.dropna(subset=[catc.time, catc.lat, catc.lon]).copy()

    t = pd.to_datetime(df[catc.time], utc=True, errors="coerce")
    df = df[pd.notna(t)].copy()
    df["_t"] = t

    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)

    df = df[(df["_t"] >= start_dt) & (df["_t"] < end_dt)].copy()

    depth_col = catc.depth_km if catc.depth_km in df.columns else None
    if depth_col is None:
        df["depth_km"] = 0.0
        depth_col = "depth_km"
    else:
        df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce").fillna(0.0)

    df = df[df[depth_col] <= float(depth_max)].copy()

    mag_col = catc.mag if catc.mag in df.columns else None
    if mag_col is None:
        df["magnitude"] = 1.0
        mag_col = "magnitude"
    else:
        df[mag_col] = pd.to_numeric(df[mag_col], errors="coerce").fillna(1.0)

    print(f"Number of events after filters: {len(df)}")

    if df.empty:
        raise ValueError("No events left after filtering.")

    minlon = float(df[catc.lon].min())
    maxlon = float(df[catc.lon].max())
    minlat = float(df[catc.lat].min())
    maxlat = float(df[catc.lat].max())
    region = [minlon, maxlon, minlat, maxlat]

    grid = pygmt.datasets.load_earth_relief(dem_res, region=region)
    lon = grid["lon"].values
    lat = grid["lat"].values
    elev_km = grid.values / 1000.0

    Lon, Lat = np.meshgrid(lon, lat)
    Z = -elev_km * float(exagg)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.10, right=0.90)

    ax.plot_surface(
        Lon,
        Lat,
        Z,
        cmap=topo_cmap,
        rstride=int(topo_stride),
        cstride=int(topo_stride),
        linewidth=0,
        antialiased=True,
        alpha=float(topo_alpha),
    )

    # ---------------------------------------------------------
    # preparar grupos
    # ---------------------------------------------------------
    pts_all = df[[catc.lon, catc.lat, depth_col]].to_numpy(dtype=float)
    mags = df[mag_col].to_numpy(dtype=float)
    sizes = float(size_k) * mags
    sizes = np.where(np.isfinite(sizes), sizes, 1.0)
    sizes = np.clip(sizes, 1.0, None)

    mask_g1 = np.zeros(len(df), dtype=bool)
    mask_g2 = np.zeros(len(df), dtype=bool)

    def rows_to_set(arr):
        if arr is None or len(arr) == 0:
            return set()
        arr = np.asarray(arr, dtype=float)
        arr = np.round(arr, 6)
        return set(map(tuple, arr))

    set_g1 = rows_to_set(group1_xyz)
    set_g2 = rows_to_set(group2_xyz)

    pts_round = np.round(pts_all, 6)

    for i, row in enumerate(pts_round):
        tup = tuple(row)
        if tup in set_g1:
            mask_g1[i] = True
        elif tup in set_g2:
            mask_g2[i] = True

    mask_other = ~(mask_g1 | mask_g2)

    scatter_kwargs = dict(
       
        linewidths=float(quake_edge_lw),
        alpha=1.0,
        depthshade=False,
    )

    if hollow_quakes:
        scatter_kwargs["facecolors"] = "none"

    # ---------------------------------------------------------
    # plot grupo 1
    # ---------------------------------------------------------
   # ---------------------------------------------------------
# estilos de scatter
# ---------------------------------------------------------
    filled_kwargs = dict(
        linewidths=float(quake_edge_lw),
        alpha=1.0,
        depthshade=False,
    )

    hollow_kwargs = dict(
        linewidths=float(quake_edge_lw),
        alpha=1.0,
        depthshade=False,
        facecolors="none",
    )

    # ---------------------------------------------------------
    # plot grupo 1 (relleno)
    # ---------------------------------------------------------
    if mask_g1.any():
        ax.scatter(
            df.loc[mask_g1, catc.lon].astype(float),
            df.loc[mask_g1, catc.lat].astype(float),
            df.loc[mask_g1, depth_col].astype(float),
            s=sizes[mask_g1],
            facecolors=group1_edgecolor,
            edgecolors=group1_edgecolor,
            **filled_kwargs,
        )

    # ---------------------------------------------------------
    # plot grupo 2 (relleno)
    # ---------------------------------------------------------
    if mask_g2.any():
        ax.scatter(
            df.loc[mask_g2, catc.lon].astype(float),
            df.loc[mask_g2, catc.lat].astype(float),
            df.loc[mask_g2, depth_col].astype(float),
            s=sizes[mask_g2],
            facecolors=group2_edgecolor,
            edgecolors=group2_edgecolor,
            **filled_kwargs,
        )

    # ---------------------------------------------------------
    # plot resto (huecos)
    # ---------------------------------------------------------
    if mask_other.any():
        ax.scatter(
            df.loc[mask_other, catc.lon].astype(float),
            df.loc[mask_other, catc.lat].astype(float),
            df.loc[mask_other, depth_col].astype(float),
            s=sizes[mask_other],
            edgecolors=other_edgecolor,
            **hollow_kwargs,
        )
    # ---------------------------------------------------------
    # planos
    # plane_polys = [
    #   {"xyz": list(zip(plane1_lon, plane1_lat, plane1_z)),
    #    "facecolor": "red", "edgecolor": "darkred", "alpha": 0.30, "lw": 1.0},
    #   {"xyz": list(zip(plane2_lon, plane2_lat, plane2_z)),
    #    "facecolor": "blue", "edgecolor": "navy", "alpha": 0.30, "lw": 1.0},
    # ]
    # ---------------------------------------------------------
    if plane_polys is not None:
        for poly in plane_polys:
            xyz = np.asarray(poly["xyz"], dtype=float)

            ax.add_collection3d(
                Poly3DCollection(
                    [xyz],
                    facecolors=poly.get("facecolor", "red"),
                    edgecolors=poly.get("edgecolor", "k"),
                    alpha=float(poly.get("alpha", 0.30)),
                    linewidths=float(poly.get("lw", 1.0)),
                )
            )

            ax.plot(
                np.r_[xyz[:, 0], xyz[0, 0]],
                np.r_[xyz[:, 1], xyz[0, 1]],
                np.r_[xyz[:, 2], xyz[0, 2]],
                color=poly.get("edgecolor", "k"),
                lw=float(poly.get("lw", 1.0)) + 0.5,
            )

    # ---------------------------------------------------------
    # ejes y estilo
    # ---------------------------------------------------------
    ax.tick_params(axis="x", labelsize=int(tick_fontsize), labelrotation=0)
    ax.tick_params(axis="y", labelsize=int(tick_fontsize) + 1, labelrotation=0)
    ax.tick_params(axis="z", labelsize=int(tick_fontsize) + 1, labelrotation=0)

    xticks = np.linspace(minlon, maxlon, int(n_xticks))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.1f}" for x in xticks])

    yticks = np.linspace(minlat, maxlat, int(n_yticks))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks])

    ax.set_zlim(float(np.nanmin(Z)), float(depth_max))
    ax.invert_zaxis()
    ax.view_init(elev=float(view_elev), azim=float(view_azim))

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.5, 0.5, 0.5, 0.99)
        axis._axinfo["grid"]["linewidth"] = 0.20
        axis._axinfo["grid"]["linestyle"] = "--"

    if savefig:
        if out_png is None:
            out_png = "mapview3d_groups.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

    meta = {
        "n_events": int(len(df)),
        "n_group1": int(mask_g1.sum()),
        "n_group2": int(mask_g2.sum()),
        "n_other": int(mask_other.sum()),
        "region": region,
        "bounds": (minlon, maxlon, minlat, maxlat),
        "dem_res": dem_res,
        "depth_max": float(depth_max),
        "time_min": str(df["_t"].min()),
        "time_max": str(df["_t"].max()),
    }

    return fig, ax, meta


# ============================================================
# PROFILE AND CROSS-SECTION TOOLS
# ============================================================

def project_profile_distances(
    lon,
    lat,
    *,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
):
    """
    Project (lon, lat) points onto a geodesic profile start -> end.

    Returns
    -------
    along_km : np.ndarray
        Projected distance along the profile [km], 0 at the start point.
    across_km : np.ndarray
        Signed perpendicular distance from the profile [km].
        The sign depends on the azimuth geometry.
    profile_length_km : float
        Total profile length [km].
    az_line_deg : float
        Azimut forward del perfil [deg].
    """
    from pyproj import Geod

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    geod = Geod(ellps="WGS84")
    az_line_deg, _, dist_line_m = geod.inv(
        float(start_lon), float(start_lat), float(end_lon), float(end_lat)
    )
    profile_length_km = float(dist_line_m) / 1000.0

    az_eq_deg, _, dist_m = geod.inv(
        np.full(lon.shape, float(start_lon)),
        np.full(lat.shape, float(start_lat)),
        lon,
        lat,
    )

    delta_rad = np.deg2rad(az_eq_deg - az_line_deg)
    dist_km = dist_m / 1000.0

    along_km = dist_km * np.cos(delta_rad)
    across_km = dist_km * np.sin(delta_rad)

    return along_km, across_km, profile_length_km, float(az_line_deg)


def add_profile_line(
    ax,
    *,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    start_label: str = "A",
    end_label: str = "A'",
    line_color: str = "cyan",
    line_width: float = 2.8,
    text_offset_deg: float = 0.01,
    fontsize: int = 13,
    marker_size: float = 42,
):
    """
    Draw a visible profile line on a Cartopy/Matplotlib map.
    """
    import cartopy.crs as ccrs
    import matplotlib.patheffects as pe

    pe_line = [pe.Stroke(linewidth=float(line_width) + 2.2, foreground="black"), pe.Normal()]
    pe_text = [pe.Stroke(linewidth=2.5, foreground="black"), pe.Normal()]

    # line
    line, = ax.plot(
        [float(start_lon), float(end_lon)],
        [float(start_lat), float(end_lat)],
        color=line_color,
        linewidth=float(line_width),
        transform=ccrs.PlateCarree(),
        zorder=200,
        solid_capstyle="round",
    )
    line.set_path_effects(pe_line)

    # extremos
    sc1 = ax.scatter(
        [float(start_lon)],
        [float(start_lat)],
        s=float(marker_size),
        marker="o",
        facecolor=line_color,
        edgecolor="black",
        linewidth=0.8,
        transform=ccrs.PlateCarree(),
        zorder=201,
    )
    sc2 = ax.scatter(
        [float(end_lon)],
        [float(end_lat)],
        s=float(marker_size),
        marker="o",
        facecolor=line_color,
        edgecolor="black",
        linewidth=0.8,
        transform=ccrs.PlateCarree(),
        zorder=201,
    )

    # etiquetas
    t1 = ax.text(
        float(start_lon) - float(text_offset_deg),
        float(start_lat),
        str(start_label),
        transform=ccrs.PlateCarree(),
        fontsize=int(fontsize),
        weight="bold",
        color="white",
        ha="right",
        va="center",
        zorder=202,
    )
    t2 = ax.text(
        float(end_lon) + float(text_offset_deg),
        float(end_lat),
        str(end_label),
        transform=ccrs.PlateCarree(),
        fontsize=int(fontsize),
        weight="bold",
        color="white",
        ha="left",
        va="center",
        zorder=202,
    )
    t1.set_path_effects(pe_text)
    t2.set_path_effects(pe_text)

    return ax


def plot_cross_section_seismicity(
    catalog: pd.DataFrame,
    *,
    start,
    end,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    catc: Optional[CatalogCols] = None,
    swath_km: Optional[float] = None,
    depth_vmin: float = 0.0,
    depth_vmax: float = 40.0,
    depth_max_plot: Optional[float] = None,
    cmap_name: str = "gist_rainbow",
    size_k: float = 22.0,
    marker_min: float = 6.0,
    figsize: Tuple[float, float] = (12, 6),
    title: Optional[str] = None,
    x_label_start: str = "A",
    x_label_end: str = "A'",
    show_end_line: bool = False,
    grid: bool = True,
    savefig: bool = False,
    out_png: Optional[str] = None,
    dpi: int = 600,
):
    """
    Cross-section of seismicity along a geographic profile.

    depth_max_plot:
        If given, removes events with depth > depth_max_plot and
        limits the Y axis to [0, depth_max_plot].
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    catc = catc or CatalogCols()
    quakes = make_quakes_for_map(catalog, start, end, catc=catc).copy()

    if quakes.empty:
        raise ValueError("No events in the range [start, end].")

    along_km, across_km, profile_length_km, az_line_deg = project_profile_distances(
        quakes["lon"].to_numpy(),
        quakes["lat"].to_numpy(),
        start_lon=float(start_lon),
        start_lat=float(start_lat),
        end_lon=float(end_lon),
        end_lat=float(end_lat),
    )

    quakes["distance_km"] = along_km
    quakes["across_km"] = across_km

    if swath_km is not None:
        swath_km = float(swath_km)
        quakes = quakes[np.abs(quakes["across_km"]) <= swath_km].copy()

    quakes = quakes[np.isfinite(quakes["distance_km"])].copy()
    quakes = quakes[np.isfinite(quakes["depth"])].copy()

    if depth_max_plot is not None:
        depth_max_plot = float(depth_max_plot)
        quakes = quakes[quakes["depth"] <= depth_max_plot].copy()

    # keep only projections that fall within the profile length
    quakes = quakes[
        (quakes["distance_km"] >= 0.0) &
        (quakes["distance_km"] <= float(profile_length_km))
    ].copy()

    if quakes.empty:
        raise ValueError("No events left after the profile/swath/depth filters.")

    sizes = float(size_k) * quakes["mag"].to_numpy(dtype=float)
    sizes = np.where(np.isfinite(sizes), sizes, float(marker_min))
    sizes = np.clip(sizes, float(marker_min), None)

    cmap = plt.get_cmap(str(cmap_name), 256)
    norm = mpl.colors.Normalize(float(depth_vmin), float(depth_vmax))

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    sc = ax.scatter(
        quakes["distance_km"],
        quakes["depth"],
        s=sizes,
        c=quakes["depth"],
        cmap=cmap,
        norm=norm,
        edgecolor="k",
        linewidth=0.25,
        alpha=0.85,
        zorder=10,
    )

    ax.set_xlabel(f"Distance along profile {x_label_start}–{x_label_end} (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_xlim(0.0, float(profile_length_km))

    if depth_max_plot is not None:
        ax.set_ylim(float(depth_max_plot), 0.0)
    else:
        ax.invert_yaxis()

    if grid:
        ax.grid(alpha=0.3, linestyle="--", zorder=0)

    if show_end_line:
        ax.axvline(float(profile_length_km), color="red", linestyle="--", alpha=0.7, zorder=5)

    if title is None:
        start_dt = pd.to_datetime(start, utc=True)
        end_dt = pd.to_datetime(end, utc=True)
        title = (
            f"Seismic profile: {start_dt.date()} to {end_dt.date()}\n"
            f"{x_label_start} ({start_lon:.4f}, {start_lat:.4f}) to "
            f"{x_label_end} ({end_lon:.4f}, {end_lat:.4f}) | "
            f"L = {profile_length_km:.1f} km"
        )
        if swath_km is not None:
            title += f" | swath = ±{swath_km:.1f} km"
        if depth_max_plot is not None:
            title += f" | depth ≤ {depth_max_plot:.1f} km"

    ax.set_title(title, pad=14)

    cbar = plt.colorbar(sc, ax=ax, label="Depth (km)", pad=0.02)
    cbar.set_alpha(1)

    if savefig:
        if out_png is None:
            out_png = "seismic_cross_section.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight", facecolor="white")
        print(f"[ok] saved: {out_png}")

    meta = {
        "profile_length_km": float(profile_length_km),
        "azimuth_deg": float(az_line_deg),
        "n_events": int(len(quakes)),
        "swath_km": None if swath_km is None else float(swath_km),
        "depth_max_plot": depth_max_plot,
        "start": (float(start_lon), float(start_lat)),
        "end": (float(end_lon), float(end_lat)),
        "distance_col": "distance_km",
        "across_col": "across_km",
        "projected_catalog": quakes.copy(),
    }
    return fig, ax, meta


def plot_plane_orientation_rose(
    dips,
    dipdirs,
    labels=None,
    colors=None,
    rmax=90,
    title="Plane orientations: dip direction / dip",
    savepath=None,
    dpi=300,
):
    """
    Polar plot:
      - angle = dip direction (deg clockwise from North)
      - radius = dip (deg)
    """

    import numpy as np
    import matplotlib.pyplot as plt

    if labels is None:
        labels = [f"P{i+1}" for i in range(len(dips))]
    if colors is None:
        colors = ["darkred", "navy", "darkgreen", "purple"]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"projection": "polar"})

    # North at top, clockwise angles
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Radial axis = dip
    ax.set_rlim(0, rmax)
    ax.set_rticks([15, 30, 45, 60, 75, 90])
    ax.set_rlabel_position(225)

    # Angular ticks every 30 deg
    theta_deg = np.arange(0, 360, 30)
    ax.set_thetagrids(theta_deg, labels=[f"{t}°" for t in theta_deg])

    # Grid appearance
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.set_title(title, va="bottom", fontsize=13)

    # Cardinal labels
    for ang_deg, txt in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        ax.text(
            np.deg2rad(ang_deg),
            rmax + 6,
            txt,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

    # Plot each plane
    for i, (dip, dipdir) in enumerate(zip(dips, dipdirs)):
        theta = np.deg2rad(dipdir)
        color = colors[i % len(colors)]
        label = labels[i]

        # radial guide line all the way to outer circle
        ax.plot([theta, theta], [0, rmax], linestyle=":", linewidth=1.8, color=color, alpha=0.8)

        # arrow from center to dip radius
        ax.annotate(
            "",
            xy=(theta, dip),
            xytext=(theta, 0),
            arrowprops=dict(arrowstyle="->", lw=3, color=color),
        )

        # point at the tip
        ax.plot(theta, dip, "o", color=color, markersize=9)

        # label
        ax.text(
            theta,
            min(dip + 7, rmax - 2),
            f"{label}\n{dipdir:.1f}° / {dip:.1f}°",
            color=color,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
        )

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight", transparent=True)

    return fig, ax


# ============================================================
# ANIMATION
# ============================================================

def animate_mapview3d_timecolor(
    catalog: pd.DataFrame,
    *,
    start,
    end,
    out_gif: str = "time_evolution.gif",
    duration_s: float = 15.0,
    fps: int = 20,
    depth_max: float = 40.0,
    dem_res: str = "15s",
    exagg: float = 4.0,
    topo_cmap: str = "gist_earth_r",
    quake_cmap: str = "magma",
    size_k: float = 15.0,
    marker_min: float = 6.0,
    topo_alpha: float = 0.95,
    topo_stride: int = 5,
    view_elev: float = 18.0,
    view_azim: float = 235.0,
    n_xticks: int = 5,
    n_yticks: int = 5,
    tick_fontsize: int = 13,
    cbar_tick_fontsize: int = 18,
    figsize: Tuple[float, float] = (12, 9),
    dpi: int = 200,
    label_fmt: str = "Elapsed: {days:.0f} days",
    box_aspect: Tuple[float, float, float] = (1.35, 1.0, 1.0),
    proj_type: str = "persp",
    catc: Optional["CatalogCols"] = None,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pygmt

    from matplotlib.colors import Normalize
    from matplotlib.animation import FuncAnimation, PillowWriter

    if catc is None:
        catc = CatalogCols()

    df = normalize_catalog(catalog, catc=catc)
    df = df.dropna(subset=[catc.time, catc.lat, catc.lon]).copy()

    df["_t"] = pd.to_datetime(df[catc.time], utc=True, errors="coerce")
    df = df.dropna(subset=["_t"]).copy()

    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)
    df = df[(df["_t"] >= start_dt) & (df["_t"] <= end_dt)].copy()

    depth_col = catc.depth_km if catc.depth_km in df.columns else None
    if depth_col is None:
        df["depth_km"] = 0.0
        depth_col = "depth_km"
    else:
        df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce").fillna(0.0)

    df = df[df[depth_col] <= float(depth_max)].copy()

    mag_col = catc.mag if catc.mag in df.columns else None
    if mag_col is None:
        df["magnitude"] = 1.0
        mag_col = "magnitude"
    else:
        df[mag_col] = pd.to_numeric(df[mag_col], errors="coerce").fillna(1.0)

    df = df.sort_values("_t").reset_index(drop=True)

    if df.empty:
        raise ValueError("No events after filtering.")

    t_num = mdates.date2num(df["_t"].to_numpy(dtype="datetime64[ns]"))

    minlon = float(df[catc.lon].min())
    maxlon = float(df[catc.lon].max())
    minlat = float(df[catc.lat].min())
    maxlat = float(df[catc.lat].max())
    region = [minlon, maxlon, minlat, maxlat]

    grid = pygmt.datasets.load_earth_relief(dem_res, region=region)
    lon = grid["lon"].values
    lat = grid["lat"].values
    elev_km = grid.values / 1000.0

    Lon, Lat = np.meshgrid(lon, lat)
    Z = -elev_km * float(exagg)

    x = df[catc.lon].astype(float).to_numpy()
    y = df[catc.lat].astype(float).to_numpy()
    z = df[depth_col].astype(float).to_numpy()

    mags = df[mag_col].to_numpy(dtype=float)
    sizes = float(size_k) * mags
    sizes = np.where(np.isfinite(sizes), sizes, float(marker_min))
    sizes = np.clip(sizes, float(marker_min), None)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.08, top=0.92)

    ax.plot_surface(
        Lon, Lat, Z,
        cmap=topo_cmap,
        rstride=int(topo_stride),
        cstride=int(topo_stride),
        linewidth=0,
        antialiased=True,
        alpha=float(topo_alpha),
    )

    norm = Normalize(vmin=float(t_num.min()), vmax=float(t_num.max()))

    sc = ax.scatter(
        [], [], [],
        c=[],
        cmap=quake_cmap,
        norm=norm,
        s=[],
        edgecolor="k",
        linewidth=0.2,
        alpha=1.0,
        depthshade=False,
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=quake_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.03, shrink=0.6, aspect=25)
    cbar.ax.tick_params(labelsize=int(cbar_tick_fontsize))
    cbar.set_label("Time", fontsize=int(cbar_tick_fontsize))

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    cbar.ax.yaxis.set_major_locator(locator)
    cbar.ax.yaxis.set_major_formatter(formatter)

    xticks = np.linspace(minlon, maxlon, int(n_xticks))
    yticks = np.linspace(minlat, maxlat, int(n_yticks))
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{v:.1f}" for v in xticks])
    ax.set_yticklabels([f"{v:.1f}" for v in yticks])

    ax.tick_params(axis="x", labelsize=int(tick_fontsize))
    ax.tick_params(axis="y", labelsize=int(tick_fontsize))
    ax.tick_params(axis="z", labelsize=int(tick_fontsize))

    ax.set_xlabel("Longitude", labelpad=12)
    ax.set_ylabel("Latitude", labelpad=12)
    ax.set_zlabel("Depth (km)", labelpad=10)

    ax.set_xlim(minlon, maxlon)
    ax.set_ylim(minlat, maxlat)
    ax.set_zlim(float(np.nanmin(Z)), float(depth_max))
    ax.invert_zaxis()

    # Visual scale for presentation, not the true physical scale.
    # It keeps the figure from looking flattened or distorted.
    ax.set_box_aspect(box_aspect)

    ax.view_init(elev=float(view_elev), azim=float(view_azim))
    ax.set_proj_type(proj_type)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = False
        except Exception:
            pass
        axis._axinfo["grid"]["color"] = (0.45, 0.45, 0.45, 0.8)
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis._axinfo["grid"]["linestyle"] = "--"

    label = ax.text2D(
        0.5, 0.98,
        label_fmt.format(days=0.0),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=15,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.4", alpha=0.9),
    )

    n_frames = max(2, int(round(float(duration_s) * int(fps))))
    frame_times = np.linspace(t_num[0], t_num[-1], n_frames)

    def update(i):
        k = np.searchsorted(t_num, frame_times[i], side="right")
        sc._offsets3d = (x[:k], y[:k], z[:k])
        sc.set_sizes(sizes[:k])
        sc.set_array(t_num[:k])
        label.set_text(label_fmt.format(days=frame_times[i] - t_num[0]))
        return sc, label

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000.0 / fps,
        blit=False,
        repeat=False,
    )

    anim.save(out_gif, writer=PillowWriter(fps=fps), dpi=int(dpi))
    plt.close(fig)

    return out_gif


# ============================================================
# NOTEBOOK UTILITIES MOVED OUT OF THE NOTEBOOK
# ============================================================


def plot_event_count_time(
    catalog: pd.DataFrame,
    *,
    time_col: str = "time",
    start=None,
    end=None,
    freq: str = "D",
    figsize: Tuple[float, float] = (12, 5),
    color: str = "black",
    linewidth: float = 1.5,
    marker=None,
    xlabel: str = "Time",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    grid: bool = True,
    savefig: bool = False,
    out_png: Optional[str] = None,
    dpi: int = 600,
):
    """
    Plot event counts through time.

    freq examples:
    - "D": daily
    - "W": weekly
    - "ME" or "M": monthly, depending on pandas version
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    df = catalog.copy()

    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' does not exist in the catalog.")

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).copy()

    if start is not None:
        start_dt = pd.to_datetime(start, utc=True, errors="coerce")
        df = df[df[time_col] >= start_dt]

    if end is not None:
        end_dt = pd.to_datetime(end, utc=True, errors="coerce")
        df = df[df[time_col] <= end_dt]

    if df.empty:
        raise ValueError("No events within the selected time range.")

    counts = df.set_index(time_col).sort_index().resample(freq).size()

    if ylabel is None:
        label_by_freq = {"D": "Events per day", "W": "Events per week", "M": "Events per month", "ME": "Events per month"}
        ylabel = label_by_freq.get(freq, f"Events per {freq}")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        counts.index,
        counts.values,
        color=color,
        linewidth=float(linewidth),
        marker=marker,
        alpha=1.0,
    )

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    if title is not None:
        ax.set_title(title)

    ax.tick_params(axis="both", labelsize=16)
    if grid:
        ax.grid(True, alpha=0.3)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.tight_layout()

    if savefig:
        if out_png is None:
            out_png = f"event_count_{freq}.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

    return fig, ax, counts


# Backward-compatible name used in the old notebook.
plot_event_count_time_v2 = plot_event_count_time


def plot_regional_location_map(
    *,
    volcano_lon: float,
    volcano_lat: float,
    region_extent: Tuple[float, float, float, float] = (-179, -145, 51, 66),
    figsize: Tuple[float, float] = (10, 10),
    land_facecolor: str = "#6da86d",
    ocean_color: str = "lightblue",
    coast_scale: str = "l",
    marker_color: str = "red",
    marker_size: float = 110,
    marker_style: str = "^",
    label: str = "",
    label_fontsize: int = 15,
    label_dlon: float = 0.0,
    label_dlat: float = -1.2,
    gridline_color: str = "gray",
    gridline_alpha: float = 0.5,
    gridline_style: str = "",
    gridline_width: float = 0.5,
    savefig: bool = False,
    out_png: str = "regional_location_map.png",
    dpi: int = 300,
    show: bool = False,
):
    """
    Regional location map for a volcano/catalog target.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker

    minlon, maxlon, minlat, maxlat = region_extent

    proj = ccrs.LambertAzimuthalEqualArea(
        central_longitude=np.mean([minlon, maxlon]),
        central_latitude=np.mean([minlat, maxlat]),
    )

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw=dict(projection=proj),
        facecolor="w",
    )

    ax.set_extent(region_extent, crs=ccrs.PlateCarree())

    coast_regional = cfeature.GSHHSFeature(
        scale=coast_scale,
        edgecolor="grey",
        facecolor=land_facecolor,
    )

    ax.add_feature(cfeature.OCEAN, color=ocean_color, zorder=0)
    ax.add_feature(coast_regional, zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="black", zorder=2)

    ax.plot(
        volcano_lon,
        volcano_lat,
        marker_style,
        color=marker_color,
        markersize=np.sqrt(marker_size),
        markeredgecolor="black",
        markeredgewidth=0.8,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    if label:
        ax.text(
            volcano_lon + label_dlon,
            volcano_lat + label_dlat,
            label,
            ha="center",
            va="center",
            fontsize=int(label_fontsize),
            transform=ccrs.PlateCarree(),
            zorder=6,
        )

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        x_inline=False,
        y_inline=False,
        color=gridline_color,
        alpha=float(gridline_alpha),
        linestyle=gridline_style,
        linewidth=float(gridline_width),
    )
    gl.xlocator = mticker.MaxNLocator(5)
    gl.ylocator = mticker.MaxNLocator(5)

    if savefig:
        fig.savefig(out_png, format=Path(out_png).suffix.lstrip(".") or "png", dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

    if show:
        plt.show()

    return fig, ax


def plot_aniakchak_alaska_location(**kwargs):
    """
    Backward-compatible wrapper for the old notebook function name.
    """
    kwargs.setdefault("volcano_lon", -158.17)
    kwargs.setdefault("volcano_lat", 56.88)
    kwargs.setdefault("label", "Aniakchak")
    kwargs.setdefault("out_png", "aniakchak_location.png")
    return plot_regional_location_map(**kwargs)


def draw_3d_box_edges(ax, color: str = "k", lw: float = 1.0, alpha: float = 0.95):
    """
    Draw the 12 edges of a Matplotlib 3D axes box using the current axis limits.
    """
    x0, x1 = ax.get_xlim3d()
    y0, y1 = ax.get_ylim3d()
    z0, z1 = ax.get_zlim3d()

    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for i, j in edges:
        xi, yi, zi = corners[i]
        xj, yj, zj = corners[j]
        ax.plot([xi, xj], [yi, yj], [zi, zj], color=color, lw=lw, alpha=alpha, zorder=1000)

    return ax


def style_3d_axes(
    ax,
    *,
    xlabel: str = "Longitude",
    ylabel: str = "Latitude",
    zlabel: str = "Depth (km)",
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    box_edge_color: str = "k",
    box_edge_lw: float = 1.0,
    show_view_text: bool = True,
    view_elev: Optional[float] = None,
    view_azim: Optional[float] = None,
):
    """
    Apply a consistent clean style to Matplotlib 3D axes.
    """
    ax.set_xlabel(xlabel, fontsize=int(label_fontsize), labelpad=14)
    ax.set_ylabel(ylabel, fontsize=int(label_fontsize), labelpad=14)
    ax.set_zlabel(zlabel, fontsize=int(label_fontsize), labelpad=10)

    ax.tick_params(axis="x", labelsize=int(tick_fontsize))
    ax.tick_params(axis="y", labelsize=int(tick_fontsize))
    ax.tick_params(axis="z", labelsize=int(tick_fontsize))

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor((0, 0, 0, 0))
        except Exception:
            pass

        axis._axinfo["grid"]["color"] = (0.45, 0.45, 0.45, 0.8)
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis._axinfo["grid"]["linestyle"] = "--"

    draw_3d_box_edges(ax, color=box_edge_color, lw=float(box_edge_lw), alpha=0.95)

    if show_view_text and (view_elev is not None) and (view_azim is not None):
        ax.text2D(
            0.02, 0.98,
            f"View: elev={float(view_elev):.1f}°, azim={float(view_azim):.1f}°",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=int(label_fontsize) - 1,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.4", alpha=0.85),
        )

    return ax



# ============================================================
# PICKS / WAVEFORM HELPERS
# ============================================================

def _rename_first_match(df: pd.DataFrame, target: str, candidates: List[str]) -> pd.DataFrame:
    """
    If target exists -> do nothing.
    Else: find first candidate that exists and rename it -> target.
    If none found -> leave as is (caller decides if it's fatal).
    """
    if target in df.columns:
        return df
    for c in candidates:
        if c in df.columns:
            return df.rename(columns={c: target})
    return df


def _require_cols(df: pd.DataFrame, required: List[str], name: str = "DataFrame") -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"{name} missing required columns: {missing}\n"
            f"Columns present: {list(df.columns)}"
        )


def _coerce_utc(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def normalize_picks(picks: pd.DataFrame, pkc: PicksCols = PicksCols()) -> pd.DataFrame:
    df = picks.copy()

    df = _rename_first_match(df, pkc.time, ["time", "pick_time", "datetime", "t"])
    df = _rename_first_match(df, pkc.station_id, ["station_id", "station", "sta", "sid"])
    df = _rename_first_match(df, pkc.phase, ["phase", "phase_type", "pick_type"])

    _require_cols(df, [pkc.station_id, pkc.time, pkc.phase], name="picks")

    df = _coerce_utc(df, pkc.time)

    # Force event id columns to nullable integer (event_id / event_id_mapped)
    for c in [pkc.event_id, pkc.event_id_mapped]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    return df


def normalize_stations_for_waveforms(
    stations: pd.DataFrame,
    stc: StationCols = StationCols(),
) -> pd.DataFrame:
    """
    Normalize station metadata for waveform/event-pick plots.

    Returns columns compatible with StationCols:
    station, latitude, longitude, elev_km.

    This accepts both:
    - the cleaned helper format: station/latitude/longitude/elev_km
    - the older waveform helper format: id/lat/lon/elev_km
    - raw GAMMA_station_list.json columns such as elevation(m)
    """
    df = stations.copy()

    rename_map = {}
    if stc.station_id not in df.columns:
        src = _first_existing_column(df, [stc.station_id, "id", "station_id", "station", "sta", "sid"])
        if src is not None:
            rename_map[src] = stc.station_id

    if stc.lat not in df.columns:
        src = _first_existing_column(df, [stc.lat, "lat", "latitude", "Lat", "LAT", "y"])
        if src is not None:
            rename_map[src] = stc.lat

    if stc.lon not in df.columns:
        src = _first_existing_column(df, [stc.lon, "lon", "longitude", "Lon", "LON", "x"])
        if src is not None:
            rename_map[src] = stc.lon

    if stc.elev_km not in df.columns:
        src = _first_existing_column(df, [stc.elev_km, "elevation_km", "elevation(m)", "elevation_m", "elev", "elevation"])
        if src is not None:
            rename_map[src] = stc.elev_km

    if rename_map:
        df = df.rename(columns=rename_map)

    _require_cols(df, [stc.station_id, stc.lat, stc.lon], name="stations")

    df[stc.lat] = pd.to_numeric(df[stc.lat], errors="coerce")
    df[stc.lon] = pd.to_numeric(df[stc.lon], errors="coerce")

    if stc.elev_km in df.columns:
        df[stc.elev_km] = pd.to_numeric(df[stc.elev_km], errors="coerce").fillna(0.0)
        if "elevation(m)" in stations.columns or "elevation_m" in stations.columns:
            df[stc.elev_km] = df[stc.elev_km] / 1000.0
    else:
        df[stc.elev_km] = 0.0

    return df.dropna(subset=[stc.station_id, stc.lat, stc.lon]).copy()


def load_inputs(
    catalog_csv: Path | str,
    picks_csv: Path | str,
    stations_json: Path | str,
    catc: CatalogCols = CatalogCols(),
    pkc: PicksCols = PicksCols(),
    stc: StationCols = StationCols(),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three files needed by waveform visualizations:
    catalog CSV, picks CSV, and station JSON.
    """
    catalog = load_catalog_csv(catalog_csv, catc=catc)

    picks = pd.read_csv(picks_csv)
    picks = normalize_picks(picks, pkc=pkc)

    stations = load_stations_json(stations_json)
    stations = normalize_stations_for_waveforms(stations, stc=stc)

    return catalog, picks, stations


# ============================================================
# WAVEFORMS: INDEX + CACHE
# ============================================================

class WaveformIndex:
    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values("t0").reset_index(drop=True)
        self._t0 = self.df["t0"].to_numpy(dtype=np.float64)
        self._t1 = self.df["t1"].to_numpy(dtype=np.float64)

    @staticmethod
    def build(roots: List[Path], glob_pattern: str = "**/waveforms/*.mseed", quiet: bool = True) -> "WaveformIndex":
        from obspy import read

        rows = []
        for root in roots:
            if not root.exists():
                continue
            for fp in root.glob(glob_pattern):
                if not fp.is_file():
                    continue
                try:
                    st = read(str(fp), headonly=True)
                    if not st:
                        continue
                    t0 = min(tr.stats.starttime for tr in st)
                    t1 = max(tr.stats.endtime   for tr in st)
                    if all(tr.stats.npts == 0 for tr in st):
                        # Header-only reading returns empty traces for files larger than 2 GB;
                        # use the first and last records of the file instead
                        from obspy.io.mseed.util import get_start_and_end_time
                        t0, t1 = get_start_and_end_time(str(fp))
                    rows.append({"path": str(fp), "t0": float(t0.timestamp), "t1": float(t1.timestamp)})
                except Exception as e:
                    if not quiet:
                        print(f"[index warn] {fp}: {e}")

        if not rows:
            raise RuntimeError("No .mseed files found (or their headers could not be read).")

        return WaveformIndex(pd.DataFrame(rows))

    def save(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_pickle(out_path)

    @staticmethod
    def load(in_path: Path) -> "WaveformIndex":
        df = pd.read_pickle(in_path)
        return WaveformIndex(df)

    def files_for_window(self, t_start: UTCDateTime, t_end: UTCDateTime, pad_neighbors: int = 1) -> List[Path]:
        ts = float(t_start.timestamp)
        te = float(t_end.timestamp)

        j = bisect_right(self._t0, te)
        if j <= 0:
            return []

        mask = self._t1[:j] >= ts
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            return []

        i0 = int(idx.min())
        i1 = int(idx.max())

        i0p = max(i0 - pad_neighbors, 0)
        i1p = min(i1 + pad_neighbors, len(self.df) - 1)

        return [Path(p) for p in self.df.loc[i0p:i1p, "path"].tolist()]


class StreamCache:
    def __init__(self):
        self._key = None
        self._st = None

    def get_stream(self, t_start: UTCDateTime, t_end: UTCDateTime, wf_index: WaveformIndex, pad_neighbors: int = 1):
        from obspy import read

        files = wf_index.files_for_window(t_start, t_end, pad_neighbors=pad_neighbors)
        if not files:
            return None, []

        key = tuple(str(f) for f in files)
        if self._key == key and self._st is not None:
            return self._st, list(files)

        st = None
        for fp in files:
            try:
                s = read(str(fp))
                st = s if st is None else (st + s)
            except Exception:
                pass

        if st is None:
            self._key = None
            self._st = None
            return None, list(files)

        st.merge(fill_value="interpolate")
        self._key = key
        self._st = st
        return st, list(files)


def get_trimmed_stream(cache: StreamCache, t_start: UTCDateTime, t_end: UTCDateTime,
                       wf_index: WaveformIndex, pad_neighbors: int = 1):
    st, files = cache.get_stream(t_start, t_end, wf_index, pad_neighbors=pad_neighbors)
    if st is None:
        return None, files

    stw = st.copy()
    stw.trim(t_start, t_end, pad=True, fill_value=0)
    stw.merge(fill_value="interpolate")
    return stw, files


# ============================================================
# PICKS / EVENT FILTERING
# ============================================================

def picks_for_event(row_event: pd.Series, picks: pd.DataFrame,
                    t_start: UTCDateTime, t_end: UTCDateTime,
                    catc: CatalogCols = CatalogCols(),
                    pkc: PicksCols = PicksCols(),
                    prefer_event_id: bool = True,
                    min_prob: Optional[float] = None) -> pd.DataFrame:
    pw = picks.copy()

    used_id_match = False

    if prefer_event_id and (catc.event_id in row_event.index):
        ev_id = row_event[catc.event_id]

        if pd.notna(ev_id):
            # 1) Try event_id_mapped first
            pw_by = pd.DataFrame()
            if pkc.event_id_mapped in pw.columns:
                pw_by = pw[pw[pkc.event_id_mapped] == ev_id]

            # 2) If no match, try event_id
            if pw_by.empty and (pkc.event_id in pw.columns):
                pw_by = pw[pw[pkc.event_id] == ev_id]

            if not pw_by.empty:
                pw = pw_by
                used_id_match = True

    # 3) If no id match, fall back to time window
    if not used_id_match:
        pw = pw[(pw[pkc.time] >= pd.Timestamp(t_start.datetime, tz="UTC")) &
                (pw[pkc.time] <= pd.Timestamp(t_end.datetime,   tz="UTC"))]

    if min_prob is not None and (pkc.prob in pw.columns):
        pw = pw[pw[pkc.prob] >= float(min_prob)]

    return pw


def networks_per_event(
    catalog: pd.DataFrame,
    picks: pd.DataFrame,
    catc: CatalogCols = CatalogCols(),
    pkc: PicksCols = PicksCols(),
) -> pd.DataFrame:
    """
    For every catalog event, which networks have picks of that event.

    The catalog event_id is the same number as event_id_mapped in the picks file
    (GAMMA_picks_cleaned.csv), so the picks of one event are the rows with that number.

    Returns a copy of the catalog with three new columns:
      networks     e.g. "3Z+6H+DK" (sorted, joined with "+"; "" if no picks found)
      n_networks   number of different networks (e.g. 3)
      n_stations   number of different stations with a pick
    """
    pk_col = pkc.event_id_mapped if pkc.event_id_mapped in picks.columns else pkc.event_id
    p = picks[[pk_col, pkc.station_id]].dropna().copy()
    p["_net"] = p[pkc.station_id].map(network_of)
    g = p.groupby(pk_col)
    nets = g["_net"].apply(lambda s: "+".join(sorted(set(s))))
    nsta = g[pkc.station_id].nunique()

    out = catalog.copy()
    ids = pd.to_numeric(out[catc.event_id], errors="coerce")
    out["networks"] = ids.map(nets).fillna("")
    out["n_networks"] = out["networks"].map(lambda s: len(s.split("+")) if s else 0)
    out["n_stations"] = ids.map(nsta).fillna(0).astype(int)
    return out


def select_events_by_networks(
    catalog: pd.DataFrame,
    picks: pd.DataFrame,
    require_networks: Optional[Sequence[str]] = None,
    min_networks: int = 1,
    catc: CatalogCols = CatalogCols(),
    pkc: PicksCols = PicksCols(),
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Keep only the events picked by certain networks.

    require_networks : every network in this list must have at least one pick of the event
                       (e.g. ["3Z", "6H", "DK"] = events seen by the three networks).
                       None or [] = no requirement.
    min_networks     : the event must have picks from at least this many different networks
                       (1 = no requirement, 2 = seen by two networks or more, ...).

    Returns the selected events (with the columns of networks_per_event). With verbose=True
    it prints how many events each combination of networks has, before and after.
    """
    cat = networks_per_event(catalog, picks, catc=catc, pkc=pkc)
    keep = cat["n_networks"] >= int(min_networks)
    for net in (require_networks or []):
        keep &= cat["networks"].str.split("+").map(lambda nets: net in nets)
    sel = cat[keep]
    if verbose:
        print("Events per combination of networks (whole catalog):")
        print(cat["networks"].replace("", "(no picks)").value_counts().to_string())
        print(f"\nSelected: {len(sel)} of {len(cat)} events "
              f"(require_networks={list(require_networks or [])}, min_networks={min_networks})")
    return sel


def add_mpl_xnum(pw: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    import matplotlib.dates as mdates

    pw = pw.copy()
    pw["xnum"] = mdates.date2num(pw[time_col].dt.tz_convert("UTC").dt.tz_localize(None))
    return pw


# ============================================================
# DISTANCES AND TRACE SELECTION
# ============================================================

def stations_by_hypo_distance_km(stations: pd.DataFrame, ev_lat: float, ev_lon: float, ev_depth_km: float,
                                 stc: StationCols = StationCols()) -> Tuple[List[str], Dict[str, float], float]:
    from obspy.geodetics.base import gps2dist_azimuth

    stations = normalize_stations_for_waveforms(stations, stc=stc)
    if stc.elev_km not in stations.columns:
        st = stations.copy()
        st[stc.elev_km] = 0.0
    else:
        st = stations

    hypo = []
    for _, r in st.iterrows():
        d_m, _, _ = gps2dist_azimuth(ev_lat, ev_lon, float(r[stc.lat]), float(r[stc.lon]))
        horiz_km = d_m / 1000.0
        vert_km  = float(ev_depth_km) + float(r[stc.elev_km])
        hypo.append(float(np.hypot(horiz_km, vert_km)))

    tmp = st.copy()
    tmp["hypo_km"] = hypo
    tmp = tmp.sort_values("hypo_km")

    order = tmp[stc.station_id].tolist()
    dist_map = dict(zip(tmp[stc.station_id], tmp["hypo_km"]))
    dmax = float(tmp["hypo_km"].max()) if len(tmp) else 0.0
    return order, dist_map, dmax


def window_l2_energy(trace: obspy.Trace, t_abs: UTCDateTime, win_len: float) -> float:
    half = win_len / 2.0
    tr_win = trace.copy()
    tr_win.trim(t_abs - half, t_abs + half, pad=True, fill_value=0)
    data = tr_win.data.astype(float)
    return float(np.sum(data**2)) if data.size else 0.0


def _select_vertical_trace_for_sid(st, sid: str):
    """
    sid: 'NET.STA.LOC.PREFIX'  e.g. 'AV.DT1..BH'
    Return (trace_copy, chosen_channel) or (None, None)
    """
    parts = sid.split(".")
    if len(parts) < 4:
        return None, None

    net, sta, loc, chprefix = parts[0], parts[1], parts[2], parts[3]

    # 1) Try exact prefix (BH -> BHZ)
    candidates = [f"{chprefix}Z", "BHZ", "EHZ", "SHZ", "HHZ"]

    for ch in candidates:
        tr = st.select(network=net, station=sta, location=loc, channel=ch)
        if len(tr) > 0:
            return tr[0].copy(), tr[0].stats.channel

    # 2) Any vertical channel
    tr = st.select(network=net, station=sta, location=loc, channel="*Z")
    if len(tr) > 0:
        return tr[0].copy(), tr[0].stats.channel

    # 3) Any channel with the prefix (BH*)
    tr = st.select(network=net, station=sta, location=loc, channel=f"{chprefix}*")
    if len(tr) > 0:
        return tr[0].copy(), tr[0].stats.channel

    return None, None


# ============================================================
# EVENT WAVEFORM VISUALIZATIONS
# ============================================================

def plot_event_moveout_picks_only(
    row_event: pd.Series,
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    cache: "StreamCache",
    wf_index: "WaveformIndex",
    # window
    pre_s: float = 10.0,
    post_s: float = 30.0,
    min_prob: Optional[float] = None,
    # filter
    freqmin: Optional[float] = None,
    freqmax: Optional[float] = None,
    corners: int = 4,
    zerophase: bool = True,
    # pick style
    p_style: str = "vline",     # "vline" | "marker"
    s_style: str = "vline",
    p_color: str = "black",
    s_color: str = "red",
    marker_size: float = 4.0,
    line_width: float = 1.2,
    line_height_km: Optional[float] = None,
    # wiggles
    wiggle_frac_of_spacing: float = 0.35,
    y_pad_km: Optional[float] = None,
    # output
    save_png: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 6),
):
    """
    Moveout (wiggles + picks) using only stations that have picks for this event.
    X = UTC time (matplotlib datenums), Y = hypocentral distance (km).
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import timezone
    from obspy import UTCDateTime

    catc = CatalogCols()
    pkc = PicksCols()

    # -------- event and window --------
    t0 = UTCDateTime(row_event[catc.time].to_pydatetime())
    t_start = t0 - float(pre_s)
    t_end   = t0 + float(post_s)

    ev_lat = float(row_event[catc.lat])
    ev_lon = float(row_event[catc.lon])
    ev_depth_km = float(row_event.get(catc.depth_km, 0.0))

    # -------- event picks (id-mapped first) --------
    pw = picks_for_event(
        row_event=row_event,
        picks=picks,
        t_start=t_start,
        t_end=t_end,
        catc=catc,
        pkc=pkc,
        prefer_event_id=True,
        min_prob=min_prob,
    )
    if pw.empty:
        print("[moveout] No picks for this event/window.")
        return None, None, {"files": [], "stations_used": []}

    # -------- only stations with picks --------
    stations_with_picks = sorted(pw[pkc.station_id].dropna().unique().tolist())

    # -------- distances --------
    order_all, dist_map_all, _ = stations_by_hypo_distance_km(
        stations=stations,
        ev_lat=ev_lat,
        ev_lon=ev_lon,
        ev_depth_km=ev_depth_km,
        stc=StationCols(),
    )

    # Keep stations with picks and valid distance
    stations_with_picks = [sid for sid in stations_with_picks if sid in dist_map_all]
    if not stations_with_picks:
        print("[moveout] Picks exist, but none of their station ids are in dist_map.")
        return None, None, {"files": [], "stations_used": []}

    # Sort by distance (near -> far)
    picked_ids = sorted(stations_with_picks, key=lambda sid: dist_map_all.get(sid, np.inf))

    # -------- waveforms (index + cache) --------
    st, files = get_trimmed_stream(cache, t_start, t_end, wf_index, pad_neighbors=1)
    if st is None or len(st) == 0:
        print("[moveout] No waveforms in window.")
        return None, None, {"files": [str(f) for f in files], "stations_used": []}

    # -------- base x-axis --------
    t0_num = mdates.date2num(t0.datetime.replace(tzinfo=None))

    # -------- helper: prep trace -> (d_norm, x_abs) --------
    def prep_trace(tr):
        tf = tr.copy()
        tf.detrend("demean")
        tf.taper(max_percentage=0.05, type="cosine")

        if (freqmin is not None) and (freqmax is not None):
            sr = float(tf.stats.sampling_rate)
            nyq = 0.5 * sr
            fmin = max(0.001, min(float(freqmin), nyq * 0.99))
            fmax = max(fmin + 0.001, min(float(freqmax), nyq * 0.99))
            if fmax > fmin:
                tf.filter(
                    "bandpass",
                    freqmin=fmin,
                    freqmax=fmax,
                    corners=int(corners),
                    zerophase=bool(zerophase),
                )

        d = tf.data.astype(float)
        if d.size == 0:
            return None, None

        m = float(np.max(np.abs(d)))
        if (not np.isfinite(m)) or m == 0.0:
            return None, None

        d = d / m  # normalize for moveout
        x = t0_num + tf.times(reftime=t0) / 86400.0
        return d, x

    # -------- wiggle scale based on station spacing --------
    dvals = np.array([dist_map_all[sid] for sid in picked_ids], dtype=float)
    spac = np.diff(dvals)
    spac = spac[spac > 0]
    med = float(np.median(spac)) if spac.size else 1.0

    amp_km = float(wiggle_frac_of_spacing) * med
    vline_h = float(line_height_km) if line_height_km is not None else 0.8 * med
    ypad = float(y_pad_km) if y_pad_km is not None else 0.5 * med

    # -------- plot --------
    fig, ax = plt.subplots(figsize=figsize)

    # Cache for marker placement on wiggles
    wig_cache = {}   # sid -> (x_arr, d_arr, y0)
    used_y = []

    # Plot only stations with picks
    for sid in picked_ids:
        tr, chosen_ch = _select_vertical_trace_for_sid(st, sid)
        if tr is None:
            continue

        d, x = prep_trace(tr)
        if d is None:
            continue

        y0 = float(dist_map_all[sid])
        ax.plot(x, y0 + amp_km * d, lw=0.6, alpha=0.85, zorder=1)
        wig_cache[sid] = (x, d, y0)
        used_y.append(y0)

    if not used_y:
        print("[moveout] No usable traces for stations-with-picks (nothing plotted).")
        return None, None, {"files": [str(f) for f in files], "stations_used": []}

    # -------- keep picks only for plotted stations --------
    pw = pw[pw[pkc.station_id].isin(wig_cache.keys())].copy()
    if pw.empty:
        print("[moveout] Waveforms plotted, but no picks belong to those plotted stations.")
        # Wiggles can still be shown
    else:
        
        pw = add_mpl_xnum(pw, time_col=pkc.time)

    # -------- helper for marker on wiggle --------
    def samp_on_wiggle(sid, xval):
        if sid not in wig_cache:
            return None
        x_arr, d_arr, y0 = wig_cache[sid]
        if xval < x_arr[0] or xval > x_arr[-1]:
            return None
        return y0 + amp_km * np.interp(xval, x_arr, d_arr)

    # -------- draw picks --------
    def draw_phase(phase_letter: str, style: str, color: str):
        if pw.empty:
            return
        m = pw[pkc.phase].astype(str).str.upper().eq(phase_letter)
        if not m.any():
            return

        sids = pw.loc[m, pkc.station_id].to_numpy()
        xs   = pw.loc[m, "xnum"].to_numpy()

        label_once = True
        if style.lower() == "marker":
            ys = [samp_on_wiggle(s, x) for s, x in zip(sids, xs)]
            xp = [x for x, y in zip(xs, ys) if y is not None]
            yp = [y for y in ys if y is not None]
            ax.plot(
                xp, yp,
                linestyle="none",
                marker=("o" if phase_letter == "P" else "s"),
                ms=marker_size,
                color=color,
                label=(phase_letter if label_once else ""),
                zorder=3,
            )
        else:  # vline
            half = 0.5 * vline_h
            for s, x in zip(sids, xs):
                if s not in wig_cache:
                    continue
                y0 = wig_cache[s][2]
                ax.vlines(
                    x, y0 - half, y0 + half,
                    colors=color,
                    linewidth=line_width,
                    label=(phase_letter if label_once else ""),
                    zorder=3,
                )
                label_once = False

    draw_phase("P", p_style, p_color)
    draw_phase("S", s_style, s_color)

    # -------- formatting --------
    ax.set_ylim(min(used_y) - ypad, max(used_y) + ypad)
    ax.axvline(t0_num, ls="--", lw=0.8)

    ax.set_ylabel("Distance (km)")
    ax.set_xlabel("UTC time")

    # Title
    eid = row_event.get(catc.event_id, "")
    ax.set_title(f"Moveout | event_id={eid} | {row_event[catc.time]}")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.tick_params(axis="x", labelrotation=0)
    ax.legend(loc="best")

    fig.tight_layout()

    if save_png is not None:
        save_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_png, dpi=400, bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] saved: {save_png}")
        return None, None, {"files": [str(f) for f in files], "stations_used": list(wig_cache.keys())}

    return fig, ax, {"files": [str(f) for f in files], "stations_used": list(wig_cache.keys())}


def _select_traces_for_sid(st, sid: str, components: str = "Z"):
    """Traces of one station id 'NET.STA.LOC.PREFIX' (e.g. '3Z.SK01L..CH').
    components = "Z": only the vertical; "all": vertical first, then the horizontals
    (N/E or 1/2 or 2/3, whatever the station has). Returns a list of (trace_copy, channel)."""
    parts = sid.split(".")
    if len(parts) < 4:
        return []
    net, sta, loc, chprefix = parts[0], parts[1], parts[2], parts[3]
    sel = st.select(network=net, station=sta, location=loc, channel=f"{chprefix}*")
    if len(sel) == 0:
        sel = st.select(network=net, station=sta, location=loc)
    vert = [tr for tr in sel if tr.stats.channel.endswith("Z")]
    horiz = sorted([tr for tr in sel if not tr.stats.channel.endswith("Z")], key=lambda tr: tr.stats.channel)
    chosen = vert[:1] if components == "Z" else vert[:1] + horiz[:2]
    return [(tr.copy(), tr.stats.channel) for tr in chosen]


def plot_event_record_section(
    row_event: pd.Series,
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    cache: "StreamCache",
    wf_index: "WaveformIndex",
    *,
    pre_s: float = 2.0,
    post_s: float = 25.0,
    freqmin: Optional[float] = 2.0,
    freqmax: Optional[float] = 20.0,
    corners: int = 4,
    components: str = "Z",
    group_by_network: bool = True,
    min_prob: Optional[float] = None,
    amplitude: float = 0.45,
    show_unpicked_km: Optional[float] = None,
    vp_kms: Optional[float] = None,
    vs_kms: Optional[float] = None,
    p_color: str = "black",
    s_color: str = "red",
    save_png: Optional[Path] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    Record section of one event: one row per station (or per channel), EQUALLY SPACED,
    so stations at very different distances do not leave empty gaps and stations of an
    array do not overlap.

    - Rows are ordered by hypocentral distance (nearest at the top). With
      group_by_network=True they are first grouped by network, then ordered by distance
      inside each network, with a line between networks.
    - Each trace is normalised to its OWN maximum inside the window, so small signals are
      as visible as large ones (amplitudes are NOT comparable between rows).
    - Colour = network (fixed colours, see NETWORK_COLOURS).
    - Label of each row: NET.STA, channel and hypocentral distance (km).
    - P picks: black ticks; S picks: red ticks. Only stations with picks for this event.
    - X axis: seconds after the origin time (the title gives the origin time).

    components: "Z" (vertical only) or "all" (vertical + horizontals, one row each).
    amplitude:  half height of each wiggle as a fraction of the row spacing (0.45 = almost
                touching the next row).
    show_unpicked_km: also draw, in GREY, the stations WITHOUT a pick of this event that are
                closer than this hypocentral distance (km). Useful to check the association:
                a real event should also be visible on nearby stations that have no pick.
                None = only stations with picks.
    vp_kms, vs_kms: if given, draw short dotted marks at the arrival times predicted by a
                straight ray at constant speed (time = hypocentral distance / speed), P in
                black and S in red. Only a rough guide (no layers), to see whether the picks
                are where an event at this location would put them.
    Returns (fig, ax, meta) with meta = {"files", "stations_used", "rows"}.
    """
    import matplotlib.pyplot as plt
    from obspy import UTCDateTime

    catc, pkc = CatalogCols(), PicksCols()
    t0 = UTCDateTime(row_event[catc.time].to_pydatetime())
    t_start, t_end = t0 - float(pre_s), t0 + float(post_s)

    pw = picks_for_event(row_event=row_event, picks=picks, t_start=t_start, t_end=t_end,
                         catc=catc, pkc=pkc, prefer_event_id=True, min_prob=min_prob)
    if pw.empty:
        print("[record section] No picks for this event.")
        return None, None, {"files": [], "stations_used": [], "rows": 0}

    _, dist_map, _ = stations_by_hypo_distance_km(
        stations=stations, ev_lat=float(row_event[catc.lat]), ev_lon=float(row_event[catc.lon]),
        ev_depth_km=float(row_event.get(catc.depth_km, 0.0)), stc=StationCols())
    sids = [sid for sid in pw[pkc.station_id].dropna().unique() if sid in dist_map]
    if not sids:
        print("[record section] Picks exist, but their stations are not in the station list.")
        return None, None, {"files": [], "stations_used": [], "rows": 0}
    picked = set(sids)
    if show_unpicked_km is not None:
        sids += [sid for sid, d in dist_map.items() if sid not in picked and d <= float(show_unpicked_km)]
    if group_by_network:
        # networks ordered by their nearest station, then stations by distance inside each network
        nearest = {}
        for sid in sids:
            nearest[network_of(sid)] = min(nearest.get(network_of(sid), np.inf), dist_map[sid])
        sids.sort(key=lambda sid: (nearest[network_of(sid)], network_of(sid), dist_map[sid]))
    else:
        sids.sort(key=lambda sid: dist_map[sid])

    st, files = get_trimmed_stream(cache, t_start, t_end, wf_index, pad_neighbors=1)
    if st is None or len(st) == 0:
        print("[record section] No waveforms in window.")
        return None, None, {"files": [str(f) for f in files], "stations_used": [], "rows": 0}

    # one row per (station, channel)
    rows = []
    for sid in sids:
        for tr, ch in _select_traces_for_sid(st, sid, components=components):
            tf = tr.copy()
            tf.detrend("demean")
            tf.taper(max_percentage=0.05, type="cosine")
            if freqmin is not None and freqmax is not None:
                nyq = 0.5 * float(tf.stats.sampling_rate)
                fmin = max(0.001, min(float(freqmin), nyq * 0.99))
                fmax = max(fmin + 0.001, min(float(freqmax), nyq * 0.99))
                tf.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=int(corners), zerophase=True)
            d = tf.data.astype(float)
            m = float(np.max(np.abs(d))) if d.size else 0.0
            if not np.isfinite(m) or m == 0.0:
                continue
            rows.append(dict(sid=sid, ch=ch, x=tf.times(reftime=t0), d=d / m))
    if not rows:
        print("[record section] No usable traces for the stations with picks.")
        return None, None, {"files": [str(f) for f in files], "stations_used": [], "rows": 0}

    if figsize is None:
        figsize = (11, max(4.0, 0.32 * len(rows) + 1.5))
    fig, ax = plt.subplots(figsize=figsize)
    pw = pw.copy()
    pw["t_rel"] = (pw[pkc.time] - pd.Timestamp(t0.datetime, tz="UTC")).dt.total_seconds()
    labels, prev_net = [], None
    for k, r in enumerate(rows):
        y = -k                                    # first row at the top
        net = network_of(r["sid"])
        colour = network_colour(net) if r["sid"] in picked else "0.65"
        if group_by_network and prev_net is not None and net != prev_net:
            ax.axhline(y + 0.5, color="0.6", lw=0.8, ls="--")
        prev_net = net
        ax.plot(r["x"], y + amplitude * r["d"], color=colour, lw=0.6)
        sta = r["sid"].split(".")[1]
        labels.append(f"{net}.{sta} {r['ch']}  {dist_map[r['sid']]:.1f} km"
                      + ("" if r["sid"] in picked else "  (no pick)"))
        for v, colour_pred in ((vp_kms, p_color), (vs_kms, s_color)):
            if v:
                ax.vlines(dist_map[r["sid"]] / float(v), y - 0.45, y + 0.45, colors=colour_pred,
                          lw=1.0, ls=":", zorder=2)
        for phase, colour_pick in (("P", p_color), ("S", s_color)):
            m = (pw[pkc.station_id] == r["sid"]) & pw[pkc.phase].astype(str).str.upper().eq(phase)
            for t in pw.loc[m, "t_rel"]:
                ax.vlines(t, y - 0.45, y + 0.45, colors=colour_pick, lw=1.6, zorder=3)
    ax.set_yticks([-k for k in range(len(rows))])
    ax.set_yticklabels(labels, fontsize=7)
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(network_colour(network_of(r["sid"])) if r["sid"] in picked else "0.5")
    ax.set_ylim(-len(rows) + 0.3, 0.7)
    ax.set_xlim(-float(pre_s), float(post_s))
    ax.axvline(0, color="0.4", lw=0.8, ls=":")
    ax.plot([], [], color=p_color, lw=1.6, label="P pick")
    ax.plot([], [], color=s_color, lw=1.6, label="S pick")
    if vp_kms:
        ax.plot([], [], color=p_color, lw=1.0, ls=":", label=f"P predicted ({float(vp_kms):.2f} km/s)")
    if vs_kms:
        ax.plot([], [], color=s_color, lw=1.0, ls=":", label=f"S predicted ({float(vs_kms):.2f} km/s)")
    if show_unpicked_km is not None:
        ax.plot([], [], color="0.65", lw=1.0, label=f"no pick (< {float(show_unpicked_km):g} km)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("seconds after origin time")
    band = f"{freqmin:g}-{freqmax:g} Hz" if freqmin is not None and freqmax is not None else "no filter"
    ax.set_title(f"Record section | event_id={row_event.get(catc.event_id, '')} | {row_event[catc.time]} | "
                 f"depth {float(row_event.get(catc.depth_km, float('nan'))):.1f} km | {band} | "
                 f"each trace normalised to its own maximum", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    meta = {"files": [str(f) for f in files], "stations_used": sorted({r["sid"] for r in rows if r["sid"] in picked}),
            "rows": len(rows)}
    if save_png is not None:
        Path(save_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return None, None, meta
    return fig, ax, meta


def plot_event_spectrograms_vertical_picks_only(
    row_event: pd.Series,
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    cache: "StreamCache",
    wf_index: "WaveformIndex",
    # window around origin time
    pre_s: float = 30.0,
    post_s: float = 90.0,
    min_prob: Optional[float] = None,
    # STFT / spectrogram
    window_s: float = 2.0,
    overlap_frac: float = 0.8,
    freq_min: float = 0.0,
    freq_max: float = 20.0,
    # optional filter (also applied to the trace)
    bp_freqmin: Optional[float] = None,
    bp_freqmax: Optional[float] = None,
    corners: int = 4,
    zerophase: bool = True,
    # color scaling
    db_pmin: float = 5.0,
    db_pmax: float = 98.0,
    global_clim: bool = True,
    # style
    time_color: str = "navy",
    p_color: str = "black",
    s_color: str = "red",
    pick_lw: float = 1.2,
    origin_color: str = "red",
    origin_ls: str = "--",
    # output
    save_png: Optional[Path] = None,
    dpi: int = 300,
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    One figure per event:
      - rows: stations with picks (for this event), sorted by hypocentral distance (near -> far)
      - col 0: Z trace (time)
      - col 1: Z spectrogram (STFT)
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from obspy import UTCDateTime

    # SciPy (preferred). Fallback if not available.
    try:
        from scipy.signal import spectrogram as scipy_spectrogram
        _HAS_SCIPY = True
    except Exception:
        _HAS_SCIPY = False

    catc = CatalogCols()
    pkc = PicksCols()

    # --- event and window ---
    t0 = UTCDateTime(row_event[catc.time].to_pydatetime())
    t_start = t0 - float(pre_s)
    t_end   = t0 + float(post_s)
    t0_num = mdates.date2num(t0.datetime.replace(tzinfo=None))

    ev_lat = float(row_event[catc.lat])
    ev_lon = float(row_event[catc.lon])
    ev_depth_km = float(row_event.get(catc.depth_km, 0.0))
    eid = row_event.get(catc.event_id, "")

    # --- event picks (id-mapped first) ---
    pw = picks_for_event(
        row_event=row_event,
        picks=picks,
        t_start=t_start,
        t_end=t_end,
        catc=catc,
        pkc=pkc,
        prefer_event_id=True,
        min_prob=min_prob,
    )
    if pw.empty:
        print("[spec] No picks for this event/window.")
        return None, None, {"files": [], "stations_used": []}

    # --- stations with picks ---
    stations_with_picks = sorted(pw[pkc.station_id].dropna().unique().tolist())

    # --- distances (to sort near -> far) ---
    _, dist_map_all, _ = stations_by_hypo_distance_km(
        stations=stations,
        ev_lat=ev_lat,
        ev_lon=ev_lon,
        ev_depth_km=ev_depth_km,
        stc=StationCols(),
    )

    stations_with_picks = [sid for sid in stations_with_picks if sid in dist_map_all]
    if not stations_with_picks:
        print("[spec] Picks exist, but none of their station ids are in dist_map.")
        return None, None, {"files": [], "stations_used": []}

    picked_ids = sorted(stations_with_picks, key=lambda sid: dist_map_all.get(sid, np.inf))
    nsta = len(picked_ids)

    # --- waveforms (index + cache) ---
    st, files = get_trimmed_stream(cache, t_start, t_end, wf_index, pad_neighbors=1)
    if st is None or len(st) == 0:
        print("[spec] No waveforms in window.")
        # Still build a "no data" figure
        st = None

    # --- prepare picks for drawing P/S lines ---
    pw = pw.copy()
    pw = pw[pw[pkc.station_id].isin(picked_ids)]
    if not pw.empty:
        pw = add_mpl_xnum(pw, time_col=pkc.time)

    # --- layout ---
    if figsize is None:
        # Height per station: ~2.3"
        figsize = (14.0, max(3.0, 2.3 * nsta))

    fig, axes = plt.subplots(
        nrows=nsta, ncols=2,
        figsize=figsize,
        sharex="col",
        gridspec_kw={"wspace": 0.15, "hspace": 0.25},
    )
    if nsta == 1:
        axes = np.array([axes])  # force shape (1,2)

    # --- helper: prep trace (detrend/taper + optional filter) ---
    def prep_trace_for_plot(tr):
        tf = tr.copy()
        tf.detrend("demean")
        tf.taper(max_percentage=0.05, type="cosine")

        if (bp_freqmin is not None) and (bp_freqmax is not None):
            sr = float(tf.stats.sampling_rate)
            nyq = 0.5 * sr
            fmin = max(0.001, min(float(bp_freqmin), nyq * 0.99))
            fmax = max(fmin + 0.001, min(float(bp_freqmax), nyq * 0.99))
            if fmax > fmin:
                tf.filter(
                    "bandpass",
                    freqmin=fmin,
                    freqmax=fmax,
                    corners=int(corners),
                    zerophase=bool(zerophase),
                )
        return tf

    # --- 1) precompute spectrograms (for global clim) ---
    spec_store = {}  # sid -> dict for plotting
    all_db_vals = []

    for sid in picked_ids:
        if st is None:
            spec_store[sid] = {"status": "no_data"}
            continue

        tr, chosen_ch = _select_vertical_trace_for_sid(st, sid)
        if tr is None or len(tr.data) == 0:
            spec_store[sid] = {"status": "no_data"}
            continue

        tr = tr.copy()
        tr.trim(starttime=t_start, endtime=t_end, pad=True, fill_value=0)
        if len(tr.data) == 0:
            spec_store[sid] = {"status": "no_data"}
            continue

        tf = prep_trace_for_plot(tr)
        fs = float(tf.stats.sampling_rate)

        nperseg = max(8, int(fs * float(window_s)))
        noverlap = int(nperseg * float(overlap_frac))
        noverlap = min(max(0, noverlap), nperseg - 1)

        if _HAS_SCIPY:
            f, t_seg, Sxx = scipy_spectrogram(
                tf.data.astype(float),
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                scaling="density",
                mode="psd",
            )
        else:
            # Simple fallback using matplotlib.specgram
            Pxx, f, t_seg, _ = axes[0, 1].specgram(
                tf.data.astype(float),
                NFFT=nperseg,
                Fs=fs,
                noverlap=noverlap,
                scale="dB",
                mode="psd",
            )
            # specgram returns dB in the plotted image; keep it as "Sxx_dB"
            Sxx = Pxx
            Sxx_dB = Sxx
            seg_times = [tf.stats.starttime + tt for tt in t_seg]
            seg_dates = mdates.date2num([stt.datetime.replace(tzinfo=None) for stt in seg_times])
            spec_store[sid] = {
                "status": "ok",
                "tr": tf,
                "chosen_ch": chosen_ch,
                "f": f,
                "t_seg": t_seg,
                "seg_dates": seg_dates,
                "Sxx_dB": Sxx_dB,
                "fs": fs,
            }
            all_db_vals.append(Sxx_dB[np.isfinite(Sxx_dB)].ravel())
            continue

        Sxx_dB = 10.0 * np.log10(Sxx + 1e-20)
        seg_times = [tf.stats.starttime + tt for tt in t_seg]
        seg_dates = mdates.date2num([stt.datetime.replace(tzinfo=None) for stt in seg_times])

        spec_store[sid] = {
            "status": "ok",
            "tr": tf,
            "chosen_ch": chosen_ch,
            "f": f,
            "t_seg": t_seg,
            "seg_dates": seg_dates,
            "Sxx_dB": Sxx_dB,
            "fs": fs,
        }
        all_db_vals.append(Sxx_dB[np.isfinite(Sxx_dB)].ravel())

    # Global clim (optional)
    vmin = vmax = None
    if global_clim:
        vals = np.concatenate(all_db_vals) if len(all_db_vals) else np.array([])
        if vals.size:
            vmin = float(np.percentile(vals, db_pmin))
            vmax = float(np.percentile(vals, db_pmax))

    # --- 2) plot each station (rows) ---
    stations_used = []
    last_im = None

    for r, sid in enumerate(picked_ids):
        ax_time = axes[r, 0]
        ax_spec = axes[r, 1]

        dist_km = float(dist_map_all.get(sid, np.nan))

        # Picks for this station
        pw_sid = pw[pw[pkc.station_id] == sid] if (pw is not None and not pw.empty) else pd.DataFrame()

        info = spec_store.get(sid, {"status": "no_data"})
        if info["status"] != "ok":
            ax_time.set_title(f"{sid} | {dist_km:.1f} km | NO DATA", color="gray", fontsize=11)
            ax_time.set_yticks([])
            ax_spec.set_yticks([])
            ax_time.axvline(t0_num, color=origin_color, linestyle=origin_ls, linewidth=1.1, alpha=0.85)
            ax_spec.axvline(t0_num, color=origin_color, linestyle=origin_ls, linewidth=1.1, alpha=0.85)
            continue

        tf = info["tr"]
        chosen_ch = info["chosen_ch"]
        f = info["f"]
        seg_dates = info["seg_dates"]
        Sxx_dB = info["Sxx_dB"]
        fs = float(info["fs"])

        stations_used.append(sid)

        # --- TIME TRACE ---
        times = tf.times("matplotlib")
        ax_time.plot(times, tf.data.astype(float), linewidth=0.8, color=time_color)
        ax_time.axvline(t0_num, color=origin_color, linestyle=origin_ls, linewidth=1.1, alpha=0.85)

        # P/S picks as lines
        if not pw_sid.empty:
            for _, rr in pw_sid.iterrows():
                ph = str(rr[pkc.phase]).upper()
                x = float(rr["xnum"])
                if ph == "P":
                    ax_time.axvline(x, color=p_color, linewidth=pick_lw, alpha=0.9)
                elif ph == "S":
                    ax_time.axvline(x, color=s_color, linewidth=pick_lw, alpha=0.9)

        ax_time.set_title(f"{sid} | {dist_km:.1f} km | {chosen_ch}", fontsize=11)
        ax_time.grid(True, linestyle=":", alpha=0.5)
        if r == nsta - 1:
            ax_time.set_xlabel("Time (UTC)")

        # --- SPECTROGRAM ---
        # Use pcolormesh; shading="auto" avoids manual bin issues
        im = ax_spec.pcolormesh(
            seg_dates,
            f,
            Sxx_dB,
            shading="auto",
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
        )
        last_im = im

        ax_spec.axvline(t0_num, color=origin_color, linestyle=origin_ls, linewidth=1.1, alpha=0.85)

        # P/S picks on spectrogram
        if not pw_sid.empty:
            for _, rr in pw_sid.iterrows():
                ph = str(rr[pkc.phase]).upper()
                x = float(rr["xnum"])
                if ph == "P":
                    ax_spec.axvline(x, color=p_color, linewidth=pick_lw, alpha=0.9)
                elif ph == "S":
                    ax_spec.axvline(x, color=s_color, linewidth=pick_lw, alpha=0.9)

        ax_spec.set_ylim(float(freq_min), min(float(freq_max), fs / 2.0))
        if r == nsta - 1:
            ax_spec.set_xlabel("Time (UTC)")
        ax_spec.set_ylabel("Freq [Hz]")

    # --- global x formatting ---
    date_format = mdates.DateFormatter("%H:%M:%S")
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)

    for r in range(nsta):
        for c in range(2):
            axes[r, c].xaxis.set_major_locator(locator)
            axes[r, c].xaxis.set_major_formatter(date_format)
            axes[r, c].tick_params(axis="x", labelrotation=25)

    # --- common labels ---
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 1].set_title("Spectrogram (Z)", fontsize=11)

    fig.suptitle(
        f"Event spectrograms, vertical Z) | event_id={eid} | {row_event[catc.time]} | window=[-{pre_s}s, +{post_s}s]",
        fontsize=12,
        y=0.995,
    )

    # One colorbar for the full figure
    # Reserve space on the right
    fig.tight_layout(rect=[0, 0, 0.90, 0.98])

    # Add a dedicated axis for the colorbar (single colorbar)
    if last_im is not None:
        cax = fig.add_axes([0.92, 0.12, 0.012, 0.76])  # [left, bottom, width, height]
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label("PSD (dB re 1 count$^2$/Hz)")




    meta = {"files": [str(f) for f in files] if files else [], "stations_used": stations_used}

    if save_png is not None:
        save_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_png, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] saved: {save_png}")
        return None, None, meta

    return fig, axes, meta


# Backward-compatible private name from exploratory notebook.
_draw_3d_box_edges = draw_3d_box_edges
_style_3d_axes = style_3d_axes


__all__ = [
    "CatalogCols",
    "StationCols",
    "PicksCols",
    "load_catalog_csv",
    "load_stations_json",
    "load_inputs",
    "ensure_dir",
    "normalize_catalog",
    "normalize_picks",
    "normalize_stations_for_waveforms",
    "filter_catalog_time",
    "make_quakes_for_map",
    "make_stations_wide_for_map",
    "WaveformIndex",
    "StreamCache",
    "get_trimmed_stream",
    "picks_for_event",
    "add_mpl_xnum",
    "stations_by_hypo_distance_km",
    "window_l2_energy",
    "plot_event_moveout_picks_only",
    "plot_event_spectrograms_vertical_picks_only",
    "ellipse_lonlat_from_km",
    "plot_catalog_3d_topo_seismicity",
    "plot_mapview_seismicity",
    "plot_mapview_basemap_stations",
    "plot_mapview_seismicity_zoom",
    "plot_mapview_seismicity_zoom_2",
    "plot_timeline_histogram",
    "plot_timeline_legend_only",
    "plot_event_count_time",
    "plot_event_count_time_v2",
    "plot_mapview3d_months",
    "plot_mapview3d_timecolor",
    "plot_mapview3d_groups",
    "project_profile_distances",
    "add_profile_line",
    "plot_cross_section_seismicity",
    "plot_plane_orientation_rose",
    "animate_mapview3d_timecolor",
    "plot_regional_location_map",
    "plot_aniakchak_alaska_location",
    "draw_3d_box_edges",
    "style_3d_axes",
]