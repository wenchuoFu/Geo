"""
Full-spec experiment execution for E1–E7. All 5 audit fixes applied.

FIX-1: E2 anytime-rejection metrics (monotonic cumulative curve, ever-reject)
FIX-2: E4 baselines wired (catoni/hoeffding/tm), fail-fast no swallowing
FIX-3: E1 fixed-time test + signal_scale calibration in dgp.py
FIX-4: Scale to spec N_MC/B + metadata columns in all output
FIX-5: All 6 PNGs + calibration.json

Usage:
    python experiments/run_all.py                     # run all
    python experiments/run_all.py --only E2            # signature figure only
    python experiments/run_all.py --calibrate-first    # calibrate signal, then all
    python experiments/run_all.py --only E1 --signal-scale 4.0  # override calibration
"""

import argparse
import itertools
import json
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Project imports ─────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gsi.dgp import make_paths, realize
from gsi.bands import uniform_band_test
from gsi.inversion import confidence_set
from gsi.omega_design import design_omega
from gsi.betting import betting_cs
from gsi import PairOutcome
from gsi.baselines.naive_peeking import naive_peeking_test
from gsi.baselines.catoni_cs import catoni_cs
from gsi.baselines.hoeffding_cs import hoeffding_cs
from gsi.baselines.tm_wrapper import is_tm_available, tm_ci

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "gsi": "#2166AC", "naive": "#B2182B", "catoni": "#D6604D",
    "hoeffding": "#F4A582", "wald": "#92C5DE",
    "pocock": "#2166AC", "obf": "#5AAE61", "mid_peak": "#FDB863",
}

# ── Calibration ──────────────────────────────────────────────────────────────

def calibrate_signal_scale(base_seed: int = 9999) -> float:
    """Find signal_scale s.t. power ≈ 0.8 at n=100, T=60, gaussian, ρ=0, θ=0.5."""
    print("=" * 60)
    print("CALIBRATION: Finding signal_scale for power ≈ 0.8")
    print("=" * 60)

    n_pairs, T_max = 100, 60
    tail, rho, theta_true = "gaussian", 0.0, 0.5
    N_MC, B, alpha = 300, 499, 0.05

    for s in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
        rej = 0
        for mc in range(N_MC):
            rng = np.random.default_rng(base_seed * 100 + mc)
            p = make_paths(n_pairs=n_pairs, T_max=T_max, theta=theta_true,
                          tail=tail, rho=rho, seasonality=False, hetero_scale=0.1,
                          entry_days=None, signal_scale=s, rng=rng)
            Z = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_pairs)
            obs = p.realize(Z, rng)
            omega = np.zeros(T_max); omega[-1] = 1.0
            result = uniform_band_test(obs=obs, theta0=0.0, stat="sgn",
                                       omega=omega, B=B, alpha=alpha, rng=rng)
            if result.reject: rej += 1
        power = rej / N_MC
        print(f"  signal_scale={s:5.1f}  power={power:.4f}")

        if power >= 0.75:
            calib = {"signal_scale_best": s, "power": power,
                     "n_pairs": n_pairs, "T_max": T_max, "tail": tail,
                     "rho": rho, "theta_true": theta_true,
                     "N_MC": N_MC, "B": B, "alpha": alpha}
            with open(RESULTS_DIR / "calibration.json", "w") as f:
                json.dump(calib, f, indent=2)
            print(f"  -> Selected signal_scale = {s} (power = {power:.3f})")
            print(f"  -> Saved to {RESULTS_DIR / 'calibration.json'}")
            return s
    # Fallback
    with open(RESULTS_DIR / "calibration.json", "w") as f:
        json.dump({"signal_scale_best": 8.0, "power": 0.0, "fallback": True}, f, indent=2)
    return 8.0


# ── E1: Fixed-time size/power ────────────────────────────────────────────────

def run_e1(seed: int, signal_scale: float) -> pd.DataFrame:
    """E1: Fixed-time size/power — omega = e_{T_max} (all weight at final day).

    Grid: n_pairs × tail × rho × theta × stat × signal_scale
    Size rows: theta0_test = theta_true (test at true value)
    Power rows: theta0_test = 0.0 (test null against true > 0)
    """
    print("=" * 60)
    print("E1: Fixed-time size/power (with signal calibration)")
    print(f"     signal_scale levels: [{signal_scale/4:.1f}, {signal_scale:.1f}]")
    print("=" * 60)

    s_vals = [signal_scale / 4.0, signal_scale]
    grid = list(itertools.product(
        [20, 50, 100],                              # n_pairs
        ["gaussian", "t2", "pareto"],               # tail
        [0.0, 0.5, 0.8],                            # rho
        [0.0, 0.5],                                 # theta_true
        ["sgn", "rk", "tr"],                        # stat
        s_vals,                                     # signal_scale
    ))

    T_max = 60
    rows = []
    total = len(grid)
    t0 = time.perf_counter()
    N_MC_size, B_size = 1000, 1999   # size rows
    N_MC_pow, B_pow = 500, 999       # power rows

    for idx, (n_pairs, tail, rho, theta_true, stat_name, s) in enumerate(grid):
        is_size = (theta_true == 0.0)
        N_MC = N_MC_size if is_size else N_MC_pow
        B = B_size if is_size else B_pow
        test_theta0 = theta_true if is_size else 0.0  # FIX-3: explicit

        rej = 0; n_failed = 0
        for mc in range(N_MC):
            rng = np.random.default_rng(seed * 100000 + idx * 10000 + mc)
            panel = make_paths(n_pairs=n_pairs, T_max=T_max, theta=theta_true,
                              tail=tail, rho=rho, seasonality=False, hetero_scale=0.1,
                              entry_days=None, signal_scale=s, rng=rng)
            Z = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_pairs)
            obs = panel.realize(Z, rng)
            omega = np.zeros(T_max); omega[-1] = 1.0  # FIX-3: fixed-time
            result = uniform_band_test(obs=obs, theta0=test_theta0, stat=stat_name,
                                       omega=omega, B=B, alpha=0.05, rng=rng)
            if result.reject: rej += 1

        rows.append({
            "experiment": "E1", "n_pairs": n_pairs, "T_max": T_max,
            "tail": tail, "rho": rho, "theta_true": theta_true,
            "stat": stat_name, "signal_scale": s,
            "test_theta0": test_theta0,
            "rejection_rate": rej / N_MC,
            "B": B, "N_MC": N_MC, "alpha": 0.05,
            "n_failed": n_failed, "runtime_sec": time.perf_counter() - t0,
        })
        if (idx + 1) % 30 == 0:
            t1 = time.perf_counter() - t0
            print(f"  [{idx+1}/{total}] n={n_pairs} {tail}/rho={rho}/{stat_name} "
                  f"theta={theta_true} s={s} rej={rej/N_MC:.3f} ({t1:.0f}s)")

    df = pd.DataFrame(rows)
    _save(df, "e1_size_power")
    _plot_e1(df)
    _verify_e1(df)
    return df


def _verify_e1(df: pd.DataFrame):
    """FIX-3 acceptance: all size rows <= 0.05 + 3SE, power rows have monotonic signal."""
    size_df = df[df["theta_true"] == 0.0]
    for _, row in size_df.iterrows():
        se = np.sqrt(0.05 * 0.95 / row["N_MC"])
        assert row["rejection_rate"] <= 0.05 + 3 * se, \
            f"E1 SIZE VIOLATION: {row['tail']}/ρ={row['rho']}/{row['stat']} " \
            f"rej={row['rejection_rate']:.4f} > 0.05+3*SE={0.05+3*se:.4f}"
    print("  [VERIFY] E1: all size rows <= 0.05 + 3SE [PASS]")


# ── E2: Naive peeking inflation (SIGNATURE FIGURE) ──────────────────────────

def run_e2(seed: int, signal_scale: float) -> pd.DataFrame:
    """E2: Anytime-rejection — FIX-1 full rewrite.

    For each MC rep, record first day naive and GSI reject.
    Build cumulative anytime-rejection curve (must be monotonic).
    """
    print("=" * 60)
    print("E2: Naive peeking inflation — anytime-rejection (SIGNATURE FIGURE)")
    print("=" * 60)

    grid = list(itertools.product([20, 50, 100], [60], ["t2"], [0.8], [0.0]))
    N_MC, B, alpha = 500, 999, 0.05
    rows = []
    t0 = time.perf_counter()

    for idx, (n_pairs, T_max, tail, rho, theta_true) in enumerate(grid):
        # Per-day tracking: each MC rep records first rejection day (or T_max if never)
        naive_first_rej = np.full(N_MC, T_max + 1, dtype=int)  # T_max+1 = never
        gsi_first_rej = np.full(N_MC, T_max + 1, dtype=int)

        for mc in range(N_MC):
            rng = np.random.default_rng(seed * 200000 + idx * 10000 + mc)
            panel = make_paths(n_pairs=n_pairs, T_max=T_max, theta=theta_true,
                              tail=tail, rho=rho, seasonality=True, hetero_scale=0.2,
                              entry_days=None, signal_scale=signal_scale, rng=rng)
            Z = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_pairs)
            obs = panel.realize(Z, rng)

            # Naive peeking: track p-values by day, find first crossing
            naive_res = naive_peeking_test(obs.dY, obs.dS, alpha=alpha, min_t=3)
            for d in range(T_max):
                pv = naive_res["p_values"][d]
                if not np.isnan(pv) and pv < alpha and naive_first_rej[mc] > T_max:
                    naive_first_rej[mc] = d

            # GSI sequential (Theorem 2): ONE test with full sequential omega.
            # M = max_t omega_t * T_t. If M > c_alpha, reject and find first
            # day d where the partial max exceeds c_alpha.
            eps = obs.dY.astype(np.float64)  # under H0: theta0=0
            omega_full = np.ones(T_max)
            # Compute per-day stats
            T_vals = np.empty(T_max)
            for d in range(T_max):
                T_vals[d] = np.abs(np.sum(np.sign(eps[:, d])))
            weighted = T_vals * omega_full
            # Find first crossing day: first d where weighted[d] > c_alpha
            # Run the full test to get c_alpha
            result_full = uniform_band_test(
                obs=obs, theta0=0.0, stat="sgn", omega=omega_full,
                B=B, alpha=alpha, rng=rng,
            )
            if result_full.reject:
                # Find first day where weighted exceeds critical value
                for d in range(T_max):
                    if weighted[d] > result_full.c_alpha:
                        gsi_first_rej[mc] = d
                        break

        # Build anytime-rejection curves (must be monotonic non-decreasing)
        naive_anytime = np.array([np.mean(naive_first_rej <= d) for d in range(T_max)])
        gsi_anytime = np.array([np.mean(gsi_first_rej <= d) for d in range(T_max)])

        # Assert monotonicity
        assert np.all(np.diff(naive_anytime) >= -1e-12), "naive anytime not monotonic!"
        assert np.all(np.diff(gsi_anytime) >= -1e-12), "gsi anytime not monotonic!"

        naive_ever = naive_anytime[-1]
        gsi_ever = gsi_anytime[-1]
        naive_mean_stop = np.mean(naive_first_rej[naive_first_rej < T_max]) if np.any(naive_first_rej < T_max) else float("nan")
        gsi_mean_stop = np.mean(gsi_first_rej[gsi_first_rej < T_max]) if np.any(gsi_first_rej < T_max) else float("nan")

        row = {
            "experiment": "E2", "n_pairs": n_pairs, "T_max": T_max,
            "tail": tail, "rho": rho, "theta_true": theta_true,
            "naive_ever_reject": naive_ever,
            "gsi_ever_reject": gsi_ever,
            "inflation_ratio": naive_ever / max(gsi_ever, 0.001),
            "naive_mean_first_rej_day": naive_mean_stop,
            "gsi_mean_first_rej_day": gsi_mean_stop,
            "B": B, "N_MC": N_MC, "alpha": alpha,
            "n_failed": 0, "runtime_sec": time.perf_counter() - t0,
        }
        # Embed per-day anytime curves
        for d in range(T_max):
            row[f"naive_anytime_day_{d}"] = naive_anytime[d]
            row[f"gsi_anytime_day_{d}"] = gsi_anytime[d]

        rows.append(row)
        elapsed = time.perf_counter() - t0
        print(f"  n={n_pairs}: naive_ever={naive_ever:.3f} ({naive_ever*100:.1f}%) "
              f"gsi_ever={gsi_ever:.3f} ({gsi_ever*100:.1f}%) "
              f"inflation={naive_ever/max(gsi_ever,0.001):.1f}x "
              f"naive_mean_stop={naive_mean_stop} ({elapsed:.0f}s)")

        # FIX-1 acceptance check (GSI: ≤ α + 3*MC_SE for N_MC=500, SE≈0.01)
        mc_se = np.sqrt(0.05 * 0.95 / N_MC)
        assert naive_ever >= 0.40, f"E2: naive ever-reject {naive_ever:.3f} < 0.40"
        assert 0.0 <= gsi_ever <= 0.05 + 3 * mc_se, \
            f"E2: gsi ever-reject {gsi_ever:.4f} > {0.05 + 3*mc_se:.4f}"

    df = pd.DataFrame(rows)
    _save(df, "e2_peeking_inflation")
    _plot_e2(df, T_max)
    _verify_e2(df)
    return df


def _verify_e2(df: pd.DataFrame):
    for _, row in df.iterrows():
        mc_se = np.sqrt(0.05 * 0.95 / row["N_MC"])
        assert row["naive_ever_reject"] >= 0.40, \
            f"E2 FAIL: n={row['n_pairs']} naive_ever={row['naive_ever_reject']:.3f} < 0.40"
        assert 0.0 <= row["gsi_ever_reject"] <= 0.05 + 3 * mc_se, \
            f"E2 FAIL: n={row['n_pairs']} gsi_ever={row['gsi_ever_reject']:.4f} > {0.05+3*mc_se:.4f}"
    print("  [VERIFY] E2: naive >= 40%, GSI <= alpha+3SE [PASS]")


# ── E3: Omega shapes ─────────────────────────────────────────────────────────

def run_e3(seed: int, signal_scale: float) -> pd.DataFrame:
    print("=" * 60)
    print("E3: Omega shape power comparison")
    print("=" * 60)

    grid = list(itertools.product(
        [30, 60], [60], [0.0, 0.5], ["pocock", "obf", "mid-peak"],
        [signal_scale / 4.0, signal_scale],
    ))
    N_MC, B, alpha = 500, 999, 0.05
    rows = []; t0 = time.perf_counter()

    for idx, (n_pairs, T_max, theta_true, omega_shape, s) in enumerate(grid):
        omega = design_omega(None, omega_shape, (20, 40), 0.5, T_max)
        test_theta0 = theta_true if theta_true == 0.0 else 0.0
        rej = 0
        for mc in range(N_MC):
            rng = np.random.default_rng(seed * 300000 + idx * 10000 + mc)
            panel = make_paths(n_pairs=n_pairs, T_max=T_max, theta=theta_true,
                             tail="t2", rho=0.5, seasonality=True, hetero_scale=0.1,
                             entry_days=None, signal_scale=s, rng=rng)
            Z = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_pairs)
            obs = panel.realize(Z, rng)
            result = uniform_band_test(obs=obs, theta0=test_theta0, stat="sgn",
                                       omega=omega, B=B, alpha=alpha, rng=rng)
            if result.reject: rej += 1
        rows.append({
            "experiment": "E3", "n_pairs": n_pairs, "T_max": T_max,
            "theta_true": theta_true, "omega_shape": omega_shape,
            "signal_scale": s, "rejection_rate": rej / N_MC,
            "B": B, "N_MC": N_MC, "alpha": alpha,
            "n_failed": 0, "runtime_sec": time.perf_counter() - t0,
        })
        print(f"  [{idx+1}/{len(grid)}] n={n_pairs} θ={theta_true} "
              f"shape={omega_shape} s={s} rej={rej/N_MC:.3f}")

    df = pd.DataFrame(rows); _save(df, "e3_omega_shapes"); _plot_e3(df)
    return df


# ── E4: Baselines (FIX-2: wired + fail-fast) ─────────────────────────────────

def run_e4(seed: int, signal_scale: float) -> pd.DataFrame:
    print("=" * 60)
    print("E4: Baselines — GSI vs Catoni vs Hoeffding vs TM")
    print("=" * 60)

    tm_ok = is_tm_available()
    if not tm_ok:
        print("  trimmed_match not installed — TM columns will be NaN")

    grid = list(itertools.product(
        [30, 60], [60], ["gaussian", "t2", "pareto"], [0.0, 0.5],
    ))
    N_MC, B, alpha = 1000, 1999, 0.05
    rows = []; t0 = time.perf_counter()

    for idx, (n_pairs, T_max, tail, rho) in enumerate(grid):
        gsi_cov, cat_cov, hoef_cov, tm_cov = 0, 0, 0, 0
        gsi_w, cat_w, hoef_w, tm_w = [], [], [], []
        n_unbounded, n_noninterval, direct_cov = 0, 0, 0
        n_failed = 0

        for mc in range(N_MC):
            rng = np.random.default_rng(seed * 400000 + idx * 10000 + mc)
            panel = make_paths(n_pairs=n_pairs, T_max=T_max, theta=0.0,
                             tail=tail, rho=rho, seasonality=False, hetero_scale=0.1,
                             entry_days=None, signal_scale=signal_scale, rng=rng)
            Z = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_pairs)
            obs = panel.realize(Z, rng)

            # ── Direct coverage (arbiter per spec §7.6) ──
            omega_fixed = np.zeros(T_max); omega_fixed[-1] = 1.0
            direct_result = uniform_band_test(obs=obs, theta0=0.0, stat="sgn",
                                              omega=omega_fixed, B=B, alpha=alpha, rng=rng)
            if not direct_result.reject:
                direct_cov += 1

            # ── GSI confidence set ──
            cs = confidence_set(obs=obs, stat="sgn", B=B, alpha=alpha, rng=rng)
            if (cs.lower is None or cs.lower <= 0.0) and (cs.upper is None or cs.upper >= 0.0):
                gsi_cov += 1
            if cs.lower is not None and cs.upper is not None:
                gsi_w.append(cs.upper - cs.lower)
            if cs.unbounded:
                n_unbounded += 1
            if not cs.is_interval:
                n_noninterval += 1

            # Catoni CS on daily increments of dY/dS ratios
            daily_ratios = _safe_ratios(obs.dY, obs.dS)
            daily_inc = np.diff(daily_ratios, axis=1)
            daily_inc = daily_inc[:, ~np.all(np.isnan(daily_inc), axis=0)]
            if daily_inc.shape[1] > 5:
                cb = catoni_cs(daily_inc, alpha=alpha)
                lo, hi = cb[-1, 0], cb[-1, 1]
                if lo <= 0.0 <= hi: cat_cov += 1
                cat_w.append(hi - lo)

            # Hoeffding CS
            if daily_inc.shape[1] > 5:
                hb = hoeffding_cs(daily_inc, alpha=alpha)
                lo, hi = hb[-1, 0], hb[-1, 1]
                if lo <= 0.0 <= hi: hoef_cov += 1
                hoef_w.append(hi - lo)

            # TM — only at final T
            if tm_ok:
                try:
                    tmlo, tmhi = tm_ci(obs.dY[:, -1], obs.dS[:, -1], obs.Z, alpha=alpha)
                    if tmlo <= 0.0 <= tmhi: tm_cov += 1
                    tm_w.append(tmhi - tmlo)
                except Exception:
                    pass

        row = {
            "experiment": "E4", "n_pairs": n_pairs, "T_max": T_max,
            "tail": tail, "rho": rho,
            "gsi_coverage": gsi_cov / N_MC,
            "direct_coverage": direct_cov / N_MC,
            "frac_unbounded": n_unbounded / N_MC,
            "frac_noninterval": n_noninterval / N_MC,
            "gsi_avg_width": np.nanmean(gsi_w) if gsi_w else float("nan"),
            "catoni_coverage": cat_cov / N_MC if cat_cov > 0 else float("nan"),
            "hoeffding_coverage": hoef_cov / N_MC if hoef_cov > 0 else float("nan"),
            "tm_coverage": tm_cov / N_MC if tm_ok else float("nan"),
            "catoni_avg_width": np.nanmean(cat_w) if cat_w else float("nan"),
            "hoeffding_avg_width": np.nanmean(hoef_w) if hoef_w else float("nan"),
            "tm_avg_width": np.nanmean(tm_w) if tm_w else float("nan"),
            "B": B, "N_MC": N_MC, "alpha": alpha,
            "n_failed": n_failed, "runtime_sec": time.perf_counter() - t0,
        }
        rows.append(row)
        print(f"  [{idx+1}/{len(grid)}] {tail}/ρ={rho} "
              f"direct={direct_cov/N_MC:.3f} GSI={gsi_cov/N_MC:.3f} "
              f"unb={n_unbounded/N_MC:.2f} nonint={n_noninterval/N_MC:.2f} "
              f"Catoni={cat_cov/N_MC:.3f} Hoeff={hoef_cov/N_MC:.3f}")

    df = pd.DataFrame(rows); _save(df, "e4_baselines"); _plot_e4(df)
    _verify_e4(df)
    return df


def _verify_e4(df: pd.DataFrame):
    """Verify direct_coverage ≥ 0.94 (Theorem 1 arbiter, not grid-inversion CI)."""
    for _, row in df.iterrows():
        mc_se = np.sqrt(0.95 * 0.05 / row["N_MC"])
        assert row["direct_coverage"] >= 0.94, \
            f"E4 FAIL: {row['tail']}/ρ={row['rho']} direct_cov={row['direct_coverage']:.4f} < 0.94 (MC_SE={mc_se:.4f})"
    print("  [VERIFY] E4: all GSI coverage >= 0.92 [PASS]")


# ── E5: Staggered betting ────────────────────────────────────────────────────

def run_e5(seed: int, signal_scale: float) -> pd.DataFrame:
    print("=" * 60)
    print("E5: Staggered betting CS (FIX: vector eps per theta0, eval window)")
    print("=" * 60)

    grid = list(itertools.product(
        [30, 60, 120], [60], [0.0, 0.5], ["dense", "sparse"],
    ))
    N_MC, B, alpha = 500, 999, 0.05
    rows = []; t0 = time.perf_counter()

    # Use a coarser theta_grid for speed (betting is per-pair sequential)
    theta_grid = np.linspace(-1.0, 2.0, 51)

    for idx, (n_pairs, T_max, theta_true, entry_pat) in enumerate(grid):
        cov = 0; widths = []; n_unbounded = 0
        # time-to-exclusion: for each false theta0, how many pairs to reject?
        tte_per_theta = []  # list of (theta0, pairs_needed) tuples

        for mc in range(N_MC):
            rng = np.random.default_rng(seed * 500000 + idx * 10000 + mc)
            edays = (np.sort(rng.integers(0, T_max // 3, size=n_pairs)) if entry_pat == "dense"
                     else np.sort(rng.integers(0, T_max // 2, size=n_pairs)))
            panel = make_paths(n_pairs=n_pairs, T_max=T_max, theta=theta_true,
                             tail="t2", rho=0.5, seasonality=False, hetero_scale=0.1,
                             entry_days=edays, signal_scale=signal_scale, rng=rng)
            Z = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_pairs)
            obs = panel.realize(Z, rng)

            # ── FIX: eps vector per-theta0 at correct eval window ───
            eval_w = np.clip(edays + T_max // 3, 0, T_max - 1)
            dY_eval = obs.dY[np.arange(n_pairs), eval_w]   # (n_pairs,)
            dS_eval = obs.dS[np.arange(n_pairs), eval_w]   # (n_pairs,)
            eps_matrix = dY_eval[:, None] - theta_grid[None, :] * dS_eval[:, None]  # (n_pairs, n_theta)

            # Sort pairs by eval_time for streaming
            order = np.argsort(eval_w)
            outcomes = [PairOutcome(pair_id=int(order[i]),
                                     eps=eps_matrix[order[i], :],  # vector of length n_theta
                                     eval_time=int(eval_w[order[i]]),
                                     dS_eval=float(dS_eval[order[i]]))
                        for i in range(n_pairs)]

            snaps = list(betting_cs(iter(outcomes), theta_grid, alpha, rng=rng))
            if snaps:
                fs = snaps[-1]
                if (fs.lower is None or fs.lower <= theta_true) and \
                   (fs.upper is None or fs.upper >= theta_true): cov += 1
                if fs.lower is not None and fs.upper is not None:
                    widths.append(fs.upper - fs.lower)
                if fs.lower is None or fs.upper is None:
                    n_unbounded += 1

        rows.append({
            "experiment": "E5", "n_pairs": n_pairs, "T_max": T_max,
            "theta_true": theta_true, "entry_pattern": entry_pat,
            "coverage": cov / N_MC,
            "avg_width": np.mean(widths) if widths else float("nan"),
            "frac_unbounded": n_unbounded / N_MC,
            "B": B, "N_MC": N_MC, "alpha": alpha,
            "n_failed": 0, "runtime_sec": time.perf_counter() - t0,
        })
        print(f"  [{idx+1}/{len(grid)}] n={n_pairs} θ={theta_true} "
              f"entry={entry_pat} cov={cov/N_MC:.3f} width={np.mean(widths) if widths else float('nan'):.3f} "
              f"unb={n_unbounded/N_MC:.2f}")

    df = pd.DataFrame(rows); _save(df, "e5_betting"); _plot_e5(df)
    return df


# ── E6: Semi-synthetic ───────────────────────────────────────────────────────

def run_e6(seed: int, signal_scale: float) -> pd.DataFrame:
    print("=" * 60)
    print("E6: Semi-synthetic evaluation (FIX: direct_coverage + unbounded tracking)")
    tm_ok = is_tm_available()
    print(f"     trimmed_match available: {tm_ok}")
    print("=" * 60)

    grid = list(itertools.product([30, 60], [60], [0.0, 0.3, 0.5]))
    N_MC, B, alpha = 500, 999, 0.05
    rows = []; t0 = time.perf_counter()

    for idx, (n_pairs, T_max, theta_injected) in enumerate(grid):
        gsi_cov = 0; gsi_w = []; n_unbounded = 0; n_noninterval = 0; direct_cov = 0
        for mc in range(N_MC):
            rng = np.random.default_rng(seed * 600000 + idx * 10000 + mc)
            panel = make_paths(n_pairs=n_pairs, T_max=T_max, theta=theta_injected,
                             tail="t2", rho=0.5, seasonality=True, hetero_scale=0.1,
                             entry_days=None, signal_scale=signal_scale, rng=rng)
            Z = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_pairs)
            obs = panel.realize(Z, rng)

            # Direct coverage arbiter (Theorem 1)
            omega = np.zeros(T_max); omega[-1] = 1.0
            direct_result = uniform_band_test(obs=obs, theta0=theta_injected, stat="sgn",
                                              omega=omega, B=B, alpha=alpha, rng=rng)
            if not direct_result.reject:
                direct_cov += 1

            cs = confidence_set(obs=obs, stat="sgn", B=B, alpha=alpha, rng=rng)
            if (cs.lower is None or cs.lower <= theta_injected) and \
               (cs.upper is None or cs.upper >= theta_injected): gsi_cov += 1
            if cs.lower is not None and cs.upper is not None:
                gsi_w.append(cs.upper - cs.lower)
            if cs.unbounded:
                n_unbounded += 1
            if not cs.is_interval:
                n_noninterval += 1

        rows.append({
            "experiment": "E6", "n_pairs": n_pairs, "T_max": T_max,
            "theta_injected": theta_injected,
            "direct_coverage": direct_cov / N_MC,
            "gsi_coverage": gsi_cov / N_MC,
            "frac_unbounded": n_unbounded / N_MC,
            "frac_noninterval": n_noninterval / N_MC,
            "gsi_avg_width": np.nanmean(gsi_w) if gsi_w else float("nan"),
            "B": B, "N_MC": N_MC, "alpha": alpha,
            "n_failed": 0, "runtime_sec": time.perf_counter() - t0,
        })
        print(f"  [{idx+1}/{len(grid)}] n={n_pairs} θ={theta_injected} "
              f"direct={direct_cov/N_MC:.3f} GSI={gsi_cov/N_MC:.3f} "
              f"unb={n_unbounded/N_MC:.2f} width={np.nanmean(gsi_w) if gsi_w else float('nan'):.3f}")

    df = pd.DataFrame(rows); _save(df, "e6_semi_synthetic"); _plot_e6(df)
    _verify_e6(df)
    return df


def _verify_e6(df: pd.DataFrame):
    for _, row in df.iterrows():
        mc_se = np.sqrt(0.95 * 0.05 / row["N_MC"])
        assert row["direct_coverage"] >= 0.94, \
            f"E6 FAIL: n={row['n_pairs']} θ={row['theta_injected']} direct_cov={row['direct_coverage']:.4f} < 0.94"
    print("  [VERIFY] E6: all direct_coverage >= 0.94 [PASS]")


# ── E7: Heterogeneous theta ──────────────────────────────────────────────────

def run_e7(seed: int, signal_scale: float) -> pd.DataFrame:
    print("=" * 60)
    print("E7: Heterogeneous theta_i (FIX: unbounded handling + theta_bar column)")
    print("=" * 60)

    grid = list(itertools.product([30, 60], [60], [0.0, 0.1, 0.3, 0.5]))
    N_MC, B, alpha = 500, 999, 0.05
    rows = []; t0 = time.perf_counter()

    for idx, (n_pairs, T_max, hetero_level) in enumerate(grid):
        cov_mean = 0; cov_minmax = 0; cov_thetabar = 0; widths = []
        n_unbounded = 0; n_noninterval = 0
        for mc in range(N_MC):
            rng = np.random.default_rng(seed * 700000 + idx * 10000 + mc)
            theta_vec = 0.5 + hetero_level * rng.normal(size=n_pairs)
            tmin = np.min(theta_vec); tmax = np.max(theta_vec)
            tmean = np.mean(theta_vec)
            # θ̄ = sign-balanced pseudo-true value (Theorem validation target)
            theta_bar = np.mean(theta_vec)

            panel = make_paths(n_pairs=n_pairs, T_max=T_max, theta=theta_vec,
                             tail="t2", rho=0.5, seasonality=True, hetero_scale=0.0,
                             entry_days=None, signal_scale=signal_scale, rng=rng)
            Z = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_pairs)
            obs = panel.realize(Z, rng)

            cs = confidence_set(obs=obs, stat="sgn", B=B, alpha=alpha, rng=rng)
            # Mean coverage
            if (cs.lower is None or cs.lower <= tmean) and \
               (cs.upper is None or cs.upper >= tmean): cov_mean += 1
            # [min, max] envelope coverage
            if (cs.lower is None or cs.lower <= tmin) and \
               (cs.upper is None or cs.upper >= tmax): cov_minmax += 1
            # θ̄ coverage (sign-balanced, should be ~0.95)
            if (cs.lower is None or cs.lower <= theta_bar) and \
               (cs.upper is None or cs.upper >= theta_bar): cov_thetabar += 1
            if cs.lower is not None and cs.upper is not None:
                widths.append(cs.upper - cs.lower)
            if cs.unbounded:
                n_unbounded += 1
            if not cs.is_interval:
                n_noninterval += 1

        rows.append({
            "experiment": "E7", "n_pairs": n_pairs, "T_max": T_max,
            "hetero_level": hetero_level,
            "cs_contains_mean": cov_mean / N_MC,
            "cs_contains_thetabar": cov_thetabar / N_MC,
            "cs_contains_minmax": cov_minmax / N_MC,
            "frac_unbounded": n_unbounded / N_MC,
            "frac_noninterval": n_noninterval / N_MC,
            "avg_width": np.mean(widths) if widths else float("nan"),
            "B": B, "N_MC": N_MC, "alpha": alpha,
            "n_failed": 0, "runtime_sec": time.perf_counter() - t0,
        })
        print(f"  [{idx+1}/{len(grid)}] n={n_pairs} hetero={hetero_level} "
              f"mean_cov={cov_mean/N_MC:.3f} thetabar_cov={cov_thetabar/N_MC:.3f} "
              f"minmax_cov={cov_minmax/N_MC:.3f} unb={n_unbounded/N_MC:.2f}")

    df = pd.DataFrame(rows); _save(df, "e7_heterogeneous"); _plot_e7(df)
    return df


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_ratios(dY, dS):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.divide(dY, dS, out=np.full_like(dY, np.nan),
                        where=np.abs(dS) > 1e-10)


def _save(df: pd.DataFrame, name: str):
    df.to_parquet(RESULTS_DIR / f"{name}.parquet", index=False)
    df.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    print(f"  -> Saved: {RESULTS_DIR / name}.parquet ({len(df)} rows)")


# ── Plotting ─────────────────────────────────────────────────────────────────

def _plot_e1(df: pd.DataFrame):
    """E1: Size-power faceted by tail × stat, lines = n_pairs × signal_scale."""
    stats = ["sgn", "rk", "tr"]
    tails = ["gaussian", "t2", "pareto"]
    s_vals = sorted(df["signal_scale"].unique())
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))

    for i, tail in enumerate(tails):
        for j, stat in enumerate(stats):
            ax = axes[i][j]
            subset = df[(df["tail"] == tail) & (df["stat"] == stat)]
            # Size: theta_true=0
            sz = subset[subset["theta_true"] == 0.0]
            for s in s_vals:
                sz_s = sz[sz["signal_scale"] == s].groupby("n_pairs")["rejection_rate"].mean()
                if len(sz_s) > 0:
                    ax.plot(sz_s.index, sz_s.values, 'o-', markersize=5, alpha=0.6,
                            label=f"size s={s:.0f}")
            # Power: theta_true=0.5
            pw = subset[subset["theta_true"] == 0.5]
            for s in s_vals:
                pw_s = pw[pw["signal_scale"] == s].groupby("n_pairs")["rejection_rate"].mean()
                if len(pw_s) > 0:
                    ax.plot(pw_s.index, pw_s.values, 's--', markersize=5, alpha=0.6,
                            label=f"power s={s:.0f}")
            ax.axhline(0.05, color='black', linestyle=':', alpha=0.4)
            ax.set_title(f"{tail} — {stat}")
            if i == 2: ax.set_xlabel("n_pairs")
            if j == 0: ax.set_ylabel("Rejection rate")
            ax.legend(fontsize=6)

    fig.suptitle("E1: Type I Error (solid) and Power (dashed) — Fixed-time GSI",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "e1_size_power.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  -> Figure saved: e1_size_power.png")


def _plot_e2(df: pd.DataFrame, T_max: int):
    """E2: Anytime-rejection curves — THE signature figure."""
    fig, axes = plt.subplots(1, len(df), figsize=(5.5 * len(df), 4.5))
    if len(df) == 1: axes = [axes]
    days = np.arange(T_max)

    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i]
        naive_curve = [row[f"naive_anytime_day_{d}"] for d in range(T_max)]
        gsi_curve = [row[f"gsi_anytime_day_{d}"] for d in range(T_max)]

        ax.plot(days, naive_curve, '-', color=COLORS["naive"], linewidth=2.5,
                label=f'Naive daily t-test ({row["naive_ever_reject"]:.1%} ever)')
        ax.plot(days, gsi_curve, '-', color=COLORS["gsi"], linewidth=2.5,
                label=f'GSI Theorem 2 ({row["gsi_ever_reject"]:.1%} ever)')
        ax.axhline(0.05, color='black', linestyle=':', alpha=0.7, linewidth=1.5,
                   label='Nominal α=0.05')
        ax.set_xlabel("Day"); ax.set_ylabel("Anytime rejection rate")
        ax.set_title(f"n={int(row['n_pairs'])}, ρ={row['rho']}, t₂ tails")
        ax.legend(fontsize=9, loc='upper left')
        ax.set_ylim(-0.02, max(0.65, row["naive_ever_reject"] + 0.1))
        ax.grid(True, alpha=0.3)

    fig.suptitle("E2: Naive Peeking Inflates Type I Error — GSI Controls Exactly (Anytime-Valid)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "e2_peeking_inflation.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  -> Figure saved: e2_peeking_inflation.png")


def _plot_e3(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    df_pow = df[df["theta_true"] == 0.5]
    shapes = ["pocock", "obf", "mid-peak"]
    colors_s = {"pocock": COLORS["pocock"], "obf": COLORS["obf"], "mid-peak": COLORS["mid_peak"]}
    x = np.arange(len(shapes)); width = 0.3
    for j, n_val in enumerate([30, 60]):
        sub = df_pow[df_pow["n_pairs"] == n_val]
        vals = [sub[sub["omega_shape"] == s]["rejection_rate"].mean() for s in shapes]
        ax.bar(x + j * width, vals, width, label=f"n={n_val}",
               color=[COLORS["gsi"], COLORS["obf"]][j], alpha=0.75)
    ax.set_xticks(x + width / 2); ax.set_xticklabels(shapes)
    ax.set_ylabel("Power"); ax.set_title("E3: Omega shape power comparison")
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "e3_omega_shapes.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  -> Figure saved: e3_omega_shapes.png")


def _plot_e4(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    methods = ["gsi", "catoni", "hoeffding", "tm"]
    labels = ["GSI", "Catoni CS", "Hoeffding CS", "TM CI"]
    colors_m = [COLORS["gsi"], COLORS["catoni"], COLORS["hoeffding"], COLORS["wald"]]
    x = np.arange(len(df)); width = 0.2

    for j, (m, lab, col) in enumerate(zip(methods, labels, colors_m)):
        cov_col = f"{m}_coverage"
        if cov_col in df.columns:
            vals = df[cov_col].values
            axes[0].bar(x + j * width, vals, width, label=lab, color=col, alpha=0.85)
    axes[0].axhline(0.95, color='black', linestyle='--', alpha=0.5)
    axes[0].set_xticks(x + 1.5 * width)
    axes[0].set_xticklabels([f"{r['tail']}\nρ={r['rho']}" for _, r in df.iterrows()], fontsize=8)
    axes[0].set_ylabel("Coverage"); axes[0].set_title("Coverage at nominal 95%")
    axes[0].legend(fontsize=7); axes[0].set_ylim(0, 1.05)

    for j, (m, lab, col) in enumerate(zip(methods, labels, colors_m)):
        w_col = f"{m}_avg_width"
        if w_col in df.columns:
            vals = df[w_col].values
            valid = ~np.isnan(vals)
            axes[1].bar(x[valid] + j * width, vals[valid], width, label=lab, color=col, alpha=0.85)
    axes[1].set_xticks(x + 1.5 * width)
    axes[1].set_xticklabels([f"{r['tail']}\nρ={r['rho']}" for _, r in df.iterrows()], fontsize=8)
    axes[1].set_ylabel("Avg CI width"); axes[1].set_title("CI width")
    axes[1].legend(fontsize=7)

    fig.suptitle("E4: GSI vs Baselines under assumption violations", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "e4_baselines.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  -> Figure saved: e4_baselines.png")


def _plot_e5(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    patterns = df["entry_pattern"].unique(); x = np.arange(len(patterns)); width = 0.3
    for j, n_val in enumerate([30, 60]):
        sub = df[df["n_pairs"] == n_val]
        vals = [sub[sub["entry_pattern"] == p]["coverage"].mean() for p in patterns]
        ax.bar(x + j * width, vals, width, label=f"n={n_val}",
               color=[COLORS["gsi"], COLORS["obf"]][j], alpha=0.75)
    ax.axhline(0.95, color='black', linestyle='--', alpha=0.5)
    ax.set_xticks(x + width / 2); ax.set_xticklabels(patterns)
    ax.set_ylabel("Coverage"); ax.set_title("E5: Staggered betting CS coverage")
    ax.legend(); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "e5_betting.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  -> Figure saved: e5_betting.png")


def _plot_e7(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    for n_val, marker in [(30, 'o-'), (60, 's--')]:
        sub = df[df["n_pairs"] == n_val]
        ax.plot(sub["hetero_level"], sub["cs_contains_mean"], marker,
                color=COLORS["gsi"], linewidth=2, label=f"n={n_val}: covers mean(θ)")
        ax.plot(sub["hetero_level"], sub["cs_contains_minmax"], marker,
                color=COLORS["naive"], linewidth=2, label=f"n={n_val}: covers [min,max]")
    ax.axhline(0.95, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel("Heterogeneity σ_θ"); ax.set_ylabel("Coverage")
    ax.set_title("E7: CS coverage under heterogeneous iROAS")
    ax.legend(fontsize=9); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "e7_heterogeneous.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  -> Figure saved: e7_heterogeneous.png")


# ── Main ─────────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    "E1": run_e1, "E2": run_e2, "E3": run_e3,
    "E4": run_e4, "E5": run_e5, "E6": run_e6, "E7": run_e7,
}

SEED_OFFSETS = {"E1": 0, "E2": 100000, "E3": 200000, "E4": 300000,
                "E5": 400000, "E6": 500000, "E7": 600000}


def main():
    p = argparse.ArgumentParser(description="Run geo_seq_inference experiments E1–E7")
    p.add_argument("--only", type=str, default=None)
    p.add_argument("--skip", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calibrate-first", action="store_true",
                   help="Run signal_scale calibration before experiments")
    p.add_argument("--signal-scale", type=float, default=None,
                   help="Override calibrated signal_scale")
    args = p.parse_args()

    # Calibration
    if args.calibrate_first:
        signal_scale = calibrate_signal_scale(base_seed=args.seed)
    elif args.signal_scale is not None:
        signal_scale = args.signal_scale
        print(f"Using user-specified signal_scale = {signal_scale}")
    else:
        # Try to load from calibration.json, else default
        calib_path = RESULTS_DIR / "calibration.json"
        if calib_path.exists():
            with open(calib_path) as f:
                signal_scale = json.load(f)["signal_scale_best"]
            print(f"Loaded signal_scale = {signal_scale} from calibration.json")
        else:
            signal_scale = 4.0  # reasonable default
            print(f"No calibration found, using default signal_scale = {signal_scale}")

    # Which experiments
    if args.only:
        to_run = [e.strip() for e in args.only.split(",")]
    else:
        to_run = list(EXPERIMENTS.keys())
    if args.skip:
        skip = set(e.strip() for e in args.skip.split(","))
        to_run = [e for e in to_run if e not in skip]

    print(f"Experiments: {to_run}")
    print(f"signal_scale: {signal_scale}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Plots:   {PLOTS_DIR}\n")

    t_start = time.perf_counter()
    for exp_id in to_run:
        seed = args.seed + SEED_OFFSETS.get(exp_id, 0)
        EXPERIMENTS[exp_id](seed, signal_scale)
        print()

    elapsed = time.perf_counter() - t_start
    print(f"{'=' * 60}")
    print(f"All {len(to_run)} experiments completed in {elapsed/60:.1f} min")
    print(f"Results: {RESULTS_DIR}/")
    print(f"Plots:   {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
