"""
QQ_glitch.py

Find glitches in a catalog: false events caused by telemetry failures. QQ_qc.py calls this before computing the QC.

What a glitch is: when the telemetry of a station fails, the channel stops sending the ground motion and sends a
fixed number instead (seen between about 2000 and 3600 counts, positive or negative, different at each station).
When the signal comes back it can look like an arrival, and PhaseNet picks it at several stations at once. The result
is a false event in the catalog. The QC cannot catch these events reliably, because their energy is real and appears
at all the stations at once.

The test, for each event:
  1. Take the stations that picked the event (their vertical velocity channel).
  2. Look at the raw counts in the 30 s before each station's first pick.
  3. If the count stays still for 1 s or more, that station was stuck before it picked.
  4. If 2 or more stations were stuck, the event is a glitch.

The test does not look for a specific number. "Stuck" means that the count, whatever its value, changes by at most
1 count per sample.

Accelerometers are not tested: their quiet noise can be below one count, so they can look stuck when they are not.

What this test cannot see:
  - a glitch where only one station was stuck
  - a stuck stretch shorter than 1 s
  - a channel that changes by more than 1 count per sample while it is stuck
  - a stuck stretch that ended more than 30 s before the pick
  - glitches recorded only on accelerometers

Gaps in the data are not a blind spot: the download step fills them with 0 (QQ_dl_data.py, merge with fill_value=0),
so a gap looks like a stretch stuck at 0, and this test catches it.
"""
from bisect import bisect_right

import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime

SECONDS_BEFORE_PICK = 30.0   # how far back from each station's first pick we look
MIN_STUCK_SECONDS = 1.0      # shortest stuck stretch that counts (--glitch_min_seconds in QQ_qc.py)
MIN_STUCK_STATIONS = 2       # stuck stations needed to call the event a glitch (--glitch_min_stations in QQ_qc.py)
MAX_STEP_COUNTS = 1          # "still" means the count changes by at most this much per sample
PICK_MARGIN_SECONDS = 0.2    # stop this close to the pick, so the arrival itself is not tested
MIN_DATA_FRACTION = 0.5      # a station with less data than this in the window is not tested


def first_picks_by_event(picks):
    """Return {event_id: {channel: time of its first pick}}. Station ids like AV.ANPK..BH become AV.ANPK..BHZ."""
    first = picks.groupby(["event_id_mapped", "id"])["timestamp"].min()
    return {int(event_id): {station + "Z": UTCDateTime(t.to_pydatetime()) for station, t in group.droplevel(0).items()}
            for event_id, group in first.groupby(level=0)}


class Waveforms:
    """Cut time windows from the chunk folders. Events come in time order, so each chunk is read only once."""

    def __init__(self, chunk_times, chunk_dirs):
        self.chunk_times, self.chunk_dirs = chunk_times, chunk_dirs
        self.loaded_chunk, self.traces = None, obspy.Stream()

    def _chunk_of(self, t):
        return min(max(bisect_right(self.chunk_times, t) - 1, 0), len(self.chunk_times) - 1)

    def _read_chunk(self, index, **options):
        """Raw vertical velocity traces of one chunk (empty if it has no waveform files)."""
        traces = obspy.Stream()
        for path in sorted((self.chunk_dirs[index] / "waveforms").glob("*.mseed")):
            try:
                traces += obspy.read(str(path), sourcename="*.*.*.?HZ", **options)
            except Exception as error:
                print(f"[glitch] could not read {path.name}: {error}")
        return traces

    def window(self, event_time, start, end):
        event_chunk = self._chunk_of(event_time)
        if self.loaded_chunk != event_chunk:
            self.loaded_chunk, self.traces = event_chunk, self._read_chunk(event_chunk)
        cut = self.traces.slice(start, end).copy()
        for other in range(self._chunk_of(start), self._chunk_of(end) + 1):   # the window crosses into another chunk
            if other != event_chunk:
                cut += self._read_chunk(other, starttime=start, endtime=end)
        merged = obspy.Stream()
        for channel in sorted({tr.id for tr in cut}):
            try:
                merged += cut.select(id=channel).merge()
            except Exception:   # e.g. the sampling rate changes inside the window
                print(f"[glitch] {event_time}: {channel} not tested (cannot merge)")
        return merged


def longest_stuck_seconds(trace):
    """Longest stretch, in seconds, where the count stays still. A gap in the data ends the stretch."""
    counts = np.ma.getdata(trace.data).astype(np.int64)
    has_data = ~np.ma.getmaskarray(trace.data)
    still = (np.abs(np.diff(counts)) <= MAX_STEP_COUNTS) & has_data[1:] & has_data[:-1]
    if not still.any():
        return 0.0
    edges = np.diff(np.concatenate(([0], still.astype(np.int8), [0])))
    run_lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return run_lengths.max() / trace.stats.sampling_rate


def check_event(event_time, first_picks, waveforms, min_stuck_seconds):
    """Return how many stations could be tested and the stuck ones, as "STATION:seconds"."""
    if not first_picks:
        return 0, []
    start, end = min(first_picks.values()) - SECONDS_BEFORE_PICK, max(first_picks.values())
    n_tested, stuck = 0, []
    for trace in waveforms.window(event_time, start, end):
        if trace.id not in first_picks:
            continue
        pick = first_picks[trace.id]
        before_pick = trace.slice(pick - SECONDS_BEFORE_PICK, pick - PICK_MARGIN_SECONDS)
        if np.ma.count(before_pick.data) < MIN_DATA_FRACTION * SECONDS_BEFORE_PICK * trace.stats.sampling_rate:
            continue
        n_tested += 1
        seconds = longest_stuck_seconds(before_pick)
        if seconds >= min_stuck_seconds:
            stuck.append(f"{trace.stats.station}:{seconds:.1f}")
    return n_tested, stuck


def find_glitches(catalog, picks, chunk_times, chunk_dirs,
                  min_stuck_stations=MIN_STUCK_STATIONS, min_stuck_seconds=MIN_STUCK_SECONDS):
    """
    Test every event of the catalog. Returns one row per event: event_id, n_tested (stations that could be tested),
    n_stuck, stuck_stations and is_glitch.

    catalog needs event_id and time (as datetimes); picks needs event_id_mapped, id and timestamp (as datetimes).
    """
    first_picks = first_picks_by_event(picks)
    waveforms = Waveforms(chunk_times, chunk_dirs)
    in_time_order = catalog.sort_values("time", kind="stable")

    rows = []
    for i, event in enumerate(in_time_order.itertuples(), start=1):
        event_time = UTCDateTime(event.time.to_pydatetime())
        n_tested, stuck = check_event(event_time, first_picks.get(int(event.event_id), {}), waveforms, min_stuck_seconds)
        rows.append({"event_id": event.event_id, "n_tested": n_tested, "n_stuck": len(stuck),
                     "stuck_stations": ";".join(stuck), "is_glitch": len(stuck) >= min_stuck_stations})
        if i % 500 == 0 or i == len(in_time_order):
            print(f"[glitch] checked {i}/{len(in_time_order)} events", flush=True)
    return pd.DataFrame(rows, columns=["event_id", "n_tested", "n_stuck", "stuck_stations", "is_glitch"])
