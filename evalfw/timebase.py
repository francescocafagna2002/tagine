"""
Put every dataset on true LOCAL wall-clock time before any time-of-day feature.

The two sources use different bases, and neither is local:

  AEW  fixed CET (UTC+1) year-round. Established two ways: fleet PV export peaks
       at 12:15 in June, where solar noon is ~11:28 UTC / 12:28 CET / 13:28 CEST;
       and the exporter writes a fixed 96-slot day with no DST handling at all.
  CKW  explicitly UTC (the column is named start_datetime_utc).

Why it matters. A ripple-controlled load runs on local wall time, so in fixed-CET
data it lands in two different slots depending on the season - 23:00 in summer,
00:00 in winter on the meter that exposed this. Every time-of-day feature is then
smeared across two positions: start-time concentration reads a rigid timer as only
~0.58 regular, tariff windows straddle a boundary, and "overnight" windows clip an
hour differently in summer than in winter.

Converting once at load time fixes all of them, and lets pandas apply the actual
Europe/Zurich rules rather than an approximation of them.
"""

import pandas as pd

SOURCE_TZ = {"CET": "Etc/GMT-1",   # POSIX sign convention: Etc/GMT-1 is UTC+1
             "UTC": "UTC"}


def to_local(ts, source="CET", tz="Europe/Zurich"):
    """Naive timestamps in `source` -> naive local wall-clock time."""
    s = pd.to_datetime(pd.Series(ts))
    if s.dt.tz is not None:
        s = s.dt.tz_convert(SOURCE_TZ[source])
    else:
        s = s.dt.tz_localize(SOURCE_TZ[source])
    return s.dt.tz_convert(tz).dt.tz_localize(None)


def localize(df, source="CET", tz="Europe/Zurich"):
    """Rewrite ts / d / hour to local time, in place-ish. Returns the frame.

    `d` is recomputed from the shifted timestamp, so the last hour of a summer day
    moves to the following date - which is what local time actually means.
    """
    if df.empty or "ts" not in df.columns:
        return df
    df = df.copy()
    df["ts"] = to_local(df.ts, source, tz).to_numpy()
    df["d"] = df.ts.dt.normalize()
    df["hour"] = df.ts.dt.hour
    return df
