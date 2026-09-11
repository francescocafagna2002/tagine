#!/usr/bin/env python3
"""
Visual inspection of metering points and detector output.

    streamlit run app.py

Works against either dataset:
  AEW   the converted raw parquet layer (Data/raw/ym=*/*.parquet)
  CKW   15MinValues.parquet + Homeprofiles.csv + Weather_Data.csv

Three views per meter: a day-by-time-of-day heatmap over the whole history, the
15-minute trace for a chosen window, and every detector's score with the evidence
behind it. The heatmap is the one that earns its place - a battery, a wallbox and
a heat pump all look obvious in it long before any score is computed.
"""

import glob
import importlib
import os
import sys
import numpy as np
import pandas as pd
import duckdb
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from evalfw.explain import load_meter, resolve_gp, CONFIDENCE
from evalfw import ckw as ckwmod
from evalfw.timebase import localize
from evalfw.runner import discover

st.set_page_config(page_title="Energy Fingerprints — meter inspector",
                   layout="wide", initial_sidebar_state="expanded")

# Streamlit's defaults are generous with vertical space; this fits a whole meter
# - identity, verdicts and a chart - on one screen without scrolling.
st.markdown("""<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 0rem;}
  [data-testid="stVerticalBlock"] {gap: 0.45rem;}
  [data-testid="stCaptionContainer"] p {margin-bottom: 0.2rem; line-height: 1.3;}
  /* keep the verdict cards the same height so the footer cannot ride up into them */
  [data-testid="stHorizontalBlock"] [data-testid="stCaptionContainer"] {min-height: 2.6em;}
  h3 {margin-bottom: 0.2rem; padding-top: 0rem;}
  hr {margin: 0.4rem 0;}
  [data-testid="stExpander"] details {border: none;}
</style>""", unsafe_allow_html=True)

AEW_RAW = "Data/raw/*/*.parquet"
AEW_MAP = "Data"
AEW_LABELS = "Data/training_set_v2.parquet"
CKW_TS = "Data/DataCKW/15MinValues.parquet"
CKW_PROFILES = "Data/DataCKW/Homeprofiles.csv"
CKW_WEATHER = "Data/DataCKW/Weather_Data.csv"


# ----------------------------------------------------------------- data access

@st.cache_resource
def _detectors():
    return [d for d in discover()
            if not getattr(d, "trainable", False) and not d.name.startswith("random_")]


@st.cache_data(show_spinner="scanning meters …")
def aew_meters(raw):
    return duckdb.connect().execute(f"""
        SELECT mp_id, count(DISTINCT d) AS days, min(d) AS d0, max(d) AS d1
        FROM read_parquet('{raw}', hive_partitioning=true)
        GROUP BY 1 ORDER BY days DESC
    """).df()


@st.cache_data(show_spinner="loading meter …")
def aew_load(raw, mp_id):
    return load_meter(duckdb.connect(), raw, mp_id)


@st.cache_data(show_spinner=False)
def aew_register(path, mp_id):
    if not os.path.exists(path):
        return None
    df = duckdb.connect().execute(f"""
        SELECT any_value(has_pv) AS pv, any_value(has_hp) AS hp, any_value(has_ev) AS ev,
               any_value(has_battery) AS battery, any_value(has_hp_boiler) AS hp_boiler,
               any_value(gp) AS gp, any_value(plz) AS plz, any_value(ort) AS ort,
               any_value(n_meters_for_gp) AS n_meters
        FROM read_parquet('{path}') WHERE mp_id = {int(mp_id)}
    """).df()
    return None if df.empty or df.gp.isna().all() else df.iloc[0].to_dict()


@st.cache_data(show_spinner=False)
def ckw_profiles(path):
    return ckwmod._profiles(path)


@st.cache_data(show_spinner="loading household …")
def ckw_load(ts, hid, weather):
    df = ckwmod.load_batch(duckdb.connect(), ts, [hid], weather)
    return localize(df, source="UTC") if not df.empty else df


# ----------------------------------------------------------------- plotting


# Primary detector per asset, and the cut-offs that turn its score into a verdict.
# Thresholds come from the measured separations, not from taste:
#   battery  owners 0.924 dark-hour zero-fraction vs 0.000 for non-owners
#   heat pump  heat pumps 3.5 winter/summer night ratio, non-electric 1.5,
#              resistance heating 10.2
#   EV       share of days with a sustained wallbox-band block
#   boiler   weakest of the five (AUC 0.64) - never stronger than "possible"
VERDICTS = [
    # Thresholds are OPERATING POINTS measured on labelled data, quoted as
    # (recall, false-positive rate) at that cutoff. Two tiers each: a stricter one
    # for a confident word, and one at or near Youden-optimal for a hint.
    #
    # PV is the exception and is deliberately NOT fitted. CKW's survey negatives
    # are unreliable - 190 of 563 households declaring "no solar" export more than
    # 1000 kWh - so optimising against them pushes the cutoff to 11,670 kWh/yr for
    # a 10% false-positive rate, which would only ever find large installations.
    # Physical reasoning is sounder here: sustained export IS generation. The
    # values bracket the Youden point (331 kWh/yr, recall 0.92) either side.
    ("PV",        "pv_export_total",     [(1000, "yes"), (150, "possible")], "high"),

    # battery_dark_zero on AEW (142 pos / 66 neg). AEW's battery negatives carry
    # ~18% contamination, so the real false-positive rate is BELOW these figures.
    #   0.75 -> recall 0.69, FPR 0.08      0.35 -> recall 0.88, FPR 0.18
    ("Battery",   "battery_dark_zero",   [(0.75, "yes"), (0.35, "possible")], "high"),

    # hp_winter_ratio on CKW, heat pump vs non-electric heating (463 / 91).
    #   4.0 -> recall 0.51, FPR 0.31       2.2 -> recall 0.73, FPR 0.40 (Youden)
    ("Heat pump", "hp_winter_ratio",     [(4.0, "likely"), (2.2, "possible")], "moderate"),

    # ev_sustained_blocks on CKW (187 / 413). Youden sits at 0.022 but flags half
    # the non-EV households, so the lower tier is pulled up to something usable.
    #   0.50 -> recall 0.34, FPR 0.16      0.10 -> recall 0.69, FPR 0.37
    ("EV",        "ev_sustained_blocks", [(0.50, "likely"), (0.10, "possible")], "low-mod"),

    # boiler_night_band on CKW (177 / 323). The weakest of the five by some way.
    #   0.09 -> recall 0.21, FPR 0.09      0.014 -> recall 0.57, FPR 0.40 (Youden)
    ("Boiler",    "boiler_night_band",   [(0.09, "possible"), (0.014, "weak")], "low"),
]
STYLE = {"yes": ("🟢", "green"), "likely": ("🟢", "green"),
         "possible": ("🟡", "orange"), "weak": ("🟡", "gray"),
         "no": ("⚪", "gray"), "n/a": ("⚫", "gray")}


def declared_assets(ds, labels, prof_row=None):
    """Declared truth per asset, in the same vocabulary as VERDICTS.

    The two datasets declare different things and with different reliability:
    AEW's flags come from a SUBSIDY register, so False means "no subsidy record",
    not "no asset" - on measured data 13 of 19 meters flagged "no PV" were
    exporting. CKW's come from a household survey, so False is a real negative,
    but it carries no battery question at all.
    """
    if ds == "AEW":
        if not labels:
            return {}
        return {"PV": labels.get("pv"), "Battery": labels.get("battery"),
                "Heat pump": labels.get("hp"), "EV": labels.get("ev"),
                "Boiler": labels.get("hp_boiler")}
    if prof_row is None:
        return {}
    return {"PV": prof_row.solar_installed == "true",
            "Battery": None,                       # not asked in the CKW survey
            "Heat pump": prof_row.heating_type_primary in ckwmod.HEATPUMP,
            "EV": prof_row.electric_car == "true",
            "Boiler": prof_row.hotwater_type == "boiler"}


def agreement(verdict, declared):
    """Compare a verdict with a declared label. Returns (symbol, colour, text)."""
    if declared is None:
        return "", "gray", "not declared"
    detected = verdict in ("yes", "likely")
    if verdict in ("possible", "weak"):
        return "•", "orange", f"declared {'yes' if declared else 'no'} — inconclusive"
    if detected == bool(declared):
        return "✓", "green", f"agrees (declared {'yes' if declared else 'no'})"
    return "✗", "red", f"differs (declared {'yes' if declared else 'no'})"


def verdicts(df, dets):
    """One line per asset: the primary detector's score turned into a verdict."""
    by_name = {d.name: d for d in dets}
    mp = df.mp_id.iloc[0] if "mp_id" in df.columns else 0
    # The battery signature - net import flat at zero after dark - was measured
    # AMONG PV OWNERS (AUC 0.917 there). Without generation it is not evidence of
    # storage at all: a low-consumption household draws near-zero at night anyway,
    # and a battery with nothing to charge it makes little sense.
    span_yrs = max(((df.d.max() - df.d.min()).days + 1) / 365.25, 1e-6)
    has_pv = float(df.export_kwh.fillna(0).sum()) / span_yrs > 500
    out = []
    for label, name, cuts, conf in VERDICTS:
        det = by_name.get(name)
        if det is None:
            continue
        try:
            sc, ev = det.score(mp, df)
        except Exception:
            sc, ev = float("nan"), {}
        if sc is None or (isinstance(sc, float) and np.isnan(sc)):
            why = ev.get("reason", "not assessable") if isinstance(ev, dict) else ""
            out.append((label, "n/a", why, conf, None))
            continue
        cmp_val = sc / span_yrs if name == "pv_export_total" else sc
        v = "no"
        for thr, word in cuts:
            if cmp_val >= thr:
                v = word
                break
        if name == "battery_dark_zero" and not has_pv:
            out.append((label, "no", "no PV to charge it", conf, sc))
            continue
        detail = {"pv": f"{sc:,.0f} kWh exported",
                  }.get(name, "")
        if name == "pv_export_total":
            detail = f"{sc/span_yrs:,.0f} kWh/yr exported"
        elif name == "battery_dark_zero":
            detail = f"{sc:.0%} of dark hours at zero"
        elif name == "hp_winter_ratio":
            detail = f"winter/summer night {sc:.1f}×"
        elif name == "ev_sustained_blocks":
            detail = f"irregular-block score {sc:.2f}"
        elif name == "boiler_night_band":
            detail = f"{sc:.0%} of night at boiler power"
        out.append((label, v, detail, conf, sc))
    return out


def heatmap(df, signal):
    """Time of day up the side, days across. The shape of a household at a glance."""
    d = df.copy()
    d["tod"] = d.ts.dt.hour + d.ts.dt.minute / 60
    d["day"] = d.ts.dt.date
    if signal == "net (import − export)":
        d["v"] = d.import_kwh.fillna(0) - d.export_kwh.fillna(0)
        scale, mid = "RdBu_r", 0
    elif signal == "export":
        d["v"], scale, mid = d.export_kwh.fillna(0), "YlOrRd", None
    else:
        d["v"], scale, mid = d.import_kwh.fillna(0), "Viridis", None
    # time of day on the index so it becomes the y axis
    p = d.pivot_table(index="tod", columns="day", values="v", aggfunc="mean")
    z = p.to_numpy()
    lim = np.nanquantile(np.abs(z), 0.995) if np.isfinite(z).any() else 1
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(c) for c in p.columns], y=p.index, colorscale=scale,
        zmid=mid, zmin=-lim if mid == 0 else 0, zmax=lim,
        colorbar=dict(title="kWh / 15min"),
        hovertemplate="%{x} %{y:.2f}h<br>%{z:.3f} kWh<extra></extra>"))
    fig.update_layout(height=368, margin=dict(l=0, r=0, t=26, b=0),
                      xaxis=dict(title="", nticks=14),
                      yaxis=dict(title="hour of day", dtick=2, range=[0, 24]))
    return fig


def trace(df, a, b):
    # st.slider hands back datetime.date; df.d is datetime64 from duckdb
    w = df[(df.d >= pd.Timestamp(a)) & (df.d <= pd.Timestamp(b) + pd.Timedelta(days=1))]
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(go.Scatter(x=w.ts, y=w.import_kwh, name="import",
                             line=dict(width=1, color="#1f77b4")))
    if w.export_kwh.fillna(0).sum() > 0:
        fig.add_trace(go.Scatter(x=w.ts, y=-w.export_kwh, name="export",
                                 line=dict(width=1, color="#ff7f0e")))
    fig.update_layout(height=270, margin=dict(l=0, r=0, t=10, b=0),
                      hovermode="x unified", yaxis_title="kWh / 15 min",
                      legend=dict(orientation="h", y=1.1))
    return fig


def daily_profile(df):
    d = df[~df.dst_day] if "dst_day" in df.columns else df
    g = d.groupby([d.ts.dt.month.rename("m"), d.hour]).import_kwh.mean().reset_index()
    season = {12: "winter", 1: "winter", 2: "winter", 6: "summer", 7: "summer", 8: "summer"}
    g["season"] = g.m.map(season)
    fig = go.Figure()
    for s, col in [("winter", "#3366cc"), ("summer", "#dc3912")]:
        sub = g[g.season == s].groupby("hour").import_kwh.mean()
        if len(sub):
            fig.add_trace(go.Scatter(x=sub.index, y=sub.values * 4, name=s,
                                     line=dict(width=2.5, color=col)))
    fig.update_layout(height=230, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="hour", yaxis_title="mean kW",
                      legend=dict(orientation="h", y=1.15))
    return fig


# ----------------------------------------------------------------- app

if st.sidebar.button("↻ reload detectors", help=(
        "Re-import the detector modules and clear cached instances. Needed after "
        "editing anything under detectors/ — Streamlit reloads app.py on save, but "
        "detector objects are held in cache_resource and keep the old code.")):
    # base and _blocks first: subclasses reloaded afterwards must inherit from the
    # NEW Detector class, or discover()'s issubclass check silently finds nothing
    for name in ["detectors.base", "detectors._blocks"] + \
                [m for m in list(sys.modules) if m.startswith("detectors.")
                 and m not in ("detectors.base", "detectors._blocks")]:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
    st.cache_resource.clear()
    st.rerun()

ds = st.sidebar.radio("dataset", ["AEW", "CKW"], horizontal=True)
df, ident, labels, ckw_row = pd.DataFrame(), "", None, None

if ds == "AEW":
    raw = st.sidebar.text_input("raw parquet glob", AEW_RAW)
    if not glob.glob(raw):
        st.error(f"no parquet matched `{raw}`"); st.stop()
    meters = aew_meters(raw)
    st.sidebar.caption(f"{len(meters):,} meters available")
    mode = st.sidebar.radio("find by", ["MP ID", "Geschäftspartner", "browse"],
                            horizontal=True)
    mp = None
    if mode == "MP ID":
        # No real meter id in source: default to whichever meter has the longest
        # history, or override with EF_DEFAULT_MP for a demo.
        v = st.sidebar.text_input(
            "MP ID", os.environ.get("EF_DEFAULT_MP", str(int(meters.mp_id.iloc[0]))))
        mp = int(v) if v.strip().isdigit() else None
    elif mode == "Geschäftspartner":
        gp = st.sidebar.text_input("GP number", os.environ.get("EF_DEFAULT_GP", ""))
        if gp.strip():
            r = resolve_gp(AEW_MAP, [gp.strip()])
            if r.empty or pd.isna(r.mp_id.iloc[0]):
                st.sidebar.warning("no meter for that partner")
            else:
                mp = int(r.mp_id.iloc[0])
                st.sidebar.success(f"→ MP {mp}")
    else:
        top = meters.head(400)
        mp = int(st.sidebar.selectbox(
            "meter (longest history first)", top.mp_id,
            format_func=lambda m: f"{m} · {int(top[top.mp_id==m].days.iloc[0])} d"))
    if mp:
        df = aew_load(raw, mp)
        ident = f"MP {mp}"
        labels = aew_register(AEW_LABELS, mp)
else:
    if not os.path.exists(CKW_TS):
        st.error(f"missing `{CKW_TS}`"); st.stop()
    prof = ckw_profiles(CKW_PROFILES)
    only = st.sidebar.selectbox("filter", ["all", "electric_car", "solar_installed",
                                           "heat pump heating", "boiler hot water"])
    p = prof
    if only == "electric_car":       p = prof[prof.electric_car == "true"]
    elif only == "solar_installed":  p = prof[prof.solar_installed == "true"]
    elif only == "heat pump heating":p = prof[prof.heating_type_primary.isin(ckwmod.HEATPUMP)]
    elif only == "boiler hot water": p = prof[prof.hotwater_type == "boiler"]
    st.sidebar.caption(f"{len(p):,} of {len(prof):,} households")
    hid = st.sidebar.selectbox("household", p.location_profile_id.tolist(),
                               format_func=lambda x: x[:12] + "…")
    wx = CKW_WEATHER if os.path.exists(CKW_WEATHER) else None
    df = ckw_load(CKW_TS, hid, wx)
    ident = f"household {hid[:12]}…"
    row = prof[prof.location_profile_id == hid].iloc[0]
    labels = {"heating": row.heating_type_primary, "hot water": row.hotwater_type,
              "solar": row.solar_installed, "electric car": row.electric_car,
              "persons": row.persons, "living m²": row.living_space,
              "house": row.house_type}
    ckw_row = row

if df.empty:
    st.info("pick a meter in the sidebar"); st.stop()

# ---- header: identity and the facts on two lines ---------------------------
span = (df.d.max() - df.d.min()).days + 1
yrs = max(span / 365.25, 1e-6)
unit = "kWh/yr" if span >= 180 else "kWh"
div = yrs if span >= 180 else 1
st.markdown(f"### {ident}")
st.markdown(
    f"`{df.d.nunique():,} days` ({100*df.d.nunique()/span:.0f}% of span) · "
    f"import **{df.import_kwh.sum()/div:,.0f}** {unit} · "
    f"export **{df.export_kwh.sum()/div:,.0f}** {unit} · "
    f"peak **{df.import_kwh.max()*4:.1f}** kW · "
    f"{df.d.min():%Y-%m-%d} → {df.d.max():%Y-%m-%d}")

dec = declared_assets(ds, labels, ckw_row)
if labels:
    with st.expander("declared profile" + ("  (subsidy register)" if ds == "AEW"
                                           else "  (household survey)"), expanded=False):
        st.write({k: v for k, v in labels.items()
                  if v is not None and str(v) not in ("nan", "None")})
        if ds == "AEW":
            st.warning("AEW labels come from a **subsidy register**: a missing flag "
                       "means no subsidy record, not no asset. Measured on this data, "
                       "13 of 19 meters flagged 'no PV' were exporting. Treat a "
                       "'differs' below as a question, not a detector error.")
        else:
            st.info("CKW labels are **survey-declared**, so a 'no' is a real negative. "
                    "The survey asks nothing about batteries.")
else:
    st.caption("No declared labels for this meter — "
               + ("not in the AEW register" if ds == "AEW" else "no survey row") + ".")

vs = verdicts(df, _detectors())
cols = st.columns(len(vs))
for col, (label, v, detail, conf, sc) in zip(cols, vs):
    icon, colour = STYLE.get(v, ("⚪", "gray"))
    sym, acol, atxt = agreement(v, dec.get(label))
    mark = f" :{acol}[{sym}]" if sym else ""
    col.markdown(f"{icon} **{label}** · :{colour}[{v}]{mark}")
    col.caption(f"{detail} · {atxt if sym else 'not declared'}")

tab_h, tab_t, tab_d = st.tabs(["heatmap", "15-minute trace", "detectors"])

with tab_h:
    sig = st.radio("signal", ["import", "export", "net (import − export)"],
                   horizontal=True, key="sig", label_visibility="collapsed")
    st.write("")            # breathing room between the controls and the plot
    st.plotly_chart(heatmap(df, sig), width="stretch")
    st.caption("Horizontal stripes are clock-driven loads (boiler on the night "
               "tariff, timed charging). A dark band after sunset in summer only is "
               "a battery. Broad winter-night brightness is electric heating.")

with tab_t:
    # plain dates: Streamlit's slider rejects pandas Timestamps outright
    d0 = pd.Timestamp(df.d.min()).date()
    d1 = pd.Timestamp(df.d.max()).date()
    default_a = max(d0, d1 - pd.Timedelta(days=14).to_pytimedelta())
    a, b = (st.slider("window", min_value=d0, max_value=d1,
                      value=(default_a, d1), format="YYYY-MM-DD")
            if d0 < d1 else (d0, d1))
    st.plotly_chart(trace(df, a, b), width="stretch")
    st.markdown("**Mean day, winter vs summer**")
    st.plotly_chart(daily_profile(df), width="stretch")

with tab_d:
    rows = []
    for det in sorted(_detectors(), key=lambda x: x.asset):
        try:
            sc, ev = det.score(df.mp_id.iloc[0] if "mp_id" in df.columns else 0, df)
        except Exception as e:
            sc, ev = np.nan, {"error": str(e)[:60]}
        conf = CONFIDENCE.get(det.asset, ("unknown", 0))[0]
        if getattr(det, "anti_predictive", False):
            conf = "IGNORE — worse than chance"
        rows.append({"asset": det.asset, "detector": det.name,
                     "score": None if sc is None or (isinstance(sc, float) and np.isnan(sc))
                              else round(float(sc), 4),
                     "confidence": conf,
                     "evidence": (ev.get("reason") if isinstance(ev, dict) and "reason" in ev
                                  else ", ".join(f"{k}={v}" for k, v in list(ev.items())[:3])
                                  if isinstance(ev, dict) else "")})
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={
            "asset": st.column_config.TextColumn("asset", width="small"),
            "detector": st.column_config.TextColumn("detector", width="medium"),
            "score": st.column_config.NumberColumn("score", width="small", format="%.4f"),
            "confidence": st.column_config.TextColumn("confidence", width="medium"),
            "evidence": st.column_config.TextColumn("evidence", width="large"),
        })
    st.caption("Confidence is each detector's measured AUC on labelled data — "
               "battery 0.94, PV ~0.90, heat pump 0.83, EV 0.74, boiler 0.64 — "
               "not a statement about this meter. A blank score means the detector "
               "could not apply (too little history, or no weather joined).")
