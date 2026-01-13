# counterfactual_flows/evaluate.py
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from .data import load_dataframe, load_schema

# --- Optional SciPy (preferred); fall back to NumPy-only if missing ----
try:
    from scipy.stats import ks_2samp, wasserstein_distance
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Fallback Wasserstein-1 (a.k.a. Earth Mover) for 1D if SciPy is unavailable."""
    if _HAS_SCIPY:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(a, b))
    # NumPy-only: sort and average absolute CDF diff (quantile coupling)
    a = np.sort(a.astype(np.float64))
    b = np.sort(b.astype(np.float64))
    # Pad to same length via interpolation on quantiles
    n = max(len(a), len(b))
    qa = np.interp(np.linspace(0, 1, n, endpoint=True),
                   np.linspace(0, 1, len(a), endpoint=True), a)
    qb = np.interp(np.linspace(0, 1, n, endpoint=True),
                   np.linspace(0, 1, len(b), endpoint=True), b)
    return float(np.mean(np.abs(qa - qb)))

def _ks_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Fallback Kolmogorov–Smirnov statistic for 1D if SciPy is unavailable."""
    if _HAS_SCIPY:
        from scipy.stats import ks_2samp
        return float(ks_2samp(a, b, alternative="two-sided", mode="auto").statistic)
    # NumPy-only KS: empirical CDF sup norm
    a = np.sort(a.astype(np.float64))
    b = np.sort(b.astype(np.float64))
    all_vals = np.sort(np.unique(np.concatenate([a, b])))
    # CDFs at all_vals
    Fa = np.searchsorted(a, all_vals, side="right") / len(a)
    Fb = np.searchsorted(b, all_vals, side="right") / len(b)
    return float(np.max(np.abs(Fa - Fb)))

# ---------------------- Notebook-style metrics -----------------------------

def _success_rate(cf_list: List[List[Tuple[float, float, Dict[str, Any]]]]) -> float:
    """Fraction of instances with at least one accepted CF."""
    return float(np.mean([len(L) > 0 for L in cf_list])) if len(cf_list) else float("nan")

def _avg_success_prob(cf_list) -> float:
    """Average top-1 success probability among successful CFs (notebook-style)."""
    probs = [L[0][0] for L in cf_list if L]
    return float(np.mean(probs)) if probs else float("nan")

def _avg_cost(cf_list) -> float:
    """Average cost (with notebook's simple fixed cost this just returns that value)."""
    costs = [L[0][1] for L in cf_list if L]
    return float(np.mean(costs)) if costs else float("nan")

def _collect_w_before_after(
    need_df: pd.DataFrame,
    w_cols: List[str],
    cf_list: List[List[Tuple[float, float, Dict[str, Any]]]]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Returns dicts mapping each W-feature to arrays:
      - 'before': original W from the 'need' set
      - 'after' : top-1 successful CF W' (only for successes)
    """
    before = {w: need_df[w].to_numpy(dtype=float) for w in w_cols}
    after = {w: [] for w in w_cols}

    # Align top-1 successful CF with row order
    for i, L in enumerate(cf_list):
        if not L:
            continue
        _, _, xprime = L[0]
        for w in w_cols:
            after[w].append(float(xprime[w]))

    after = {w: np.array(v, dtype=float) if len(v) > 0 else np.array([], dtype=float) for w, v in after.items()}
    return before, after

def _distribution_shift_metrics(
    before: Dict[str, np.ndarray],
    after: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    Per-feature Wasserstein and KS comparing:
      - BEFORE: original W values among all 'need' rows
      - AFTER : top-1 successful CF W' among successful rows
    """
    metrics = {}
    for w, bvals in before.items():
        avals = after[w]
        if avals.size == 0 or bvals.size == 0:
            metrics[w] = {"wasserstein": float("nan"), "ks": float("nan")}
        else:
            # Match support; no standardization here—report in natural units
            metrics[w] = {
                "wasserstein": _wasserstein_1d(bvals, avals),
                "ks": _ks_1d(bvals, avals),
            }
    return metrics

def _groupwise_shift(
    need_df: pd.DataFrame,
    sensitive_col: str,
    w_cols: List[str],
    cf_list: List[List[Tuple[float, float, Dict[str, Any]]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Groupwise (by sensitive attribute) per-feature Wasserstein and KS.
    Only rows with that group label are used for 'before';
    'after' uses successful CFs within that group.
    """
    results = {}
    # Build an index of successes to their group label
    success_idx = [i for i, L in enumerate(cf_list) if L]
    groups = need_df[sensitive_col].astype(str).values if sensitive_col else np.array(["all"] * len(need_df))

    for g in np.unique(np.asarray(groups)):
        mask = (groups == g)
        if not np.any(mask):
            continue

        # BEFORE: original W values for this group
        before_g = {w: np.asarray(need_df.loc[mask, w], dtype=float) for w in w_cols}

        # AFTER: top-1 W' only for successful instances within this group
        after_g = {w: [] for w in w_cols}
        # successful indices that also belong to group g
        for i in success_idx:
            if groups[i] != g:
                continue
            xprime = cf_list[i][0][2]
            for w in w_cols:
                after_g[w].append(float(xprime[w]))
        after_g = {w: np.array(v, dtype=float) if len(v) > 0 else np.array([], dtype=float) for w, v in after_g.items()}

        # compute
        metrics_g = {}
        for w in w_cols:
            bvals, avals = before_g[w], after_g[w]
            if avals.size == 0 or bvals.size == 0:
                metrics_g[w] = {"wasserstein": float("nan"), "ks": float("nan")}
            else:
                metrics_g[w] = {
                    "wasserstein": _wasserstein_1d(bvals, avals),
                    "ks": _ks_1d(bvals, avals),
                }
        results[str(g)] = metrics_g

    return results

# --------------------------- Entry point -----------------------------------

def run_eval(cfg_path: str):
    """
    Evaluate notebook metrics and distribution-shift metrics (Wasserstein & KS)
    for mediators W. Saves:
      - JSON summary: outputs/{cfg_stem}_metrics.json
      - CSV with per-feature shift: outputs/{cfg_stem}_shift.csv
      - (optional) CSV with groupwise shift if sensitive column exists
    """
    cfg = yaml.safe_load(open(cfg_path))
    schema = load_schema(cfg)

    data_path = cfg["dataset"]["path"]
    data_fmt = cfg["dataset"]["format"]
    target = cfg["dataset"]["target"]
    sensitive = cfg["dataset"].get("sensitive")

    w_cols = schema.partitions.W
    cond_cols = schema.partitions.X + schema.partitions.Z

    # Load data and 'need' set (Y == 0)
    df = load_dataframe(data_path, data_fmt)
    need = df[df[target] == 0].copy().reset_index(drop=True)

    # Load counterfactuals generated by generate_cf.py
    cf_path = Path("outputs") / (Path(cfg_path).stem + "_counterfactuals.npy")
    if not cf_path.exists():
        raise FileNotFoundError(f"Counterfactual file not found: {cf_path}")
    cf_list = np.load(cf_path, allow_pickle=True).tolist()

    # Notebook-style scalar metrics
    metrics_overall = {
        "success_rate": _success_rate(cf_list),
        "avg_success_prob_top1": _avg_success_prob(cf_list),
        "avg_cost_top1": _avg_cost(cf_list),
        "num_instances": int(len(cf_list)),
        "num_successes": int(sum(1 for L in cf_list if L)),
    }

    # Distribution shift metrics: BEFORE=original W (need set), AFTER=top-1 successful W'
    before, after = _collect_w_before_after(need, w_cols, cf_list)
    shift = _distribution_shift_metrics(before, after)

    # Groupwise (if sensitive is available)
    group_shift = _groupwise_shift(need, sensitive, w_cols, cf_list) if sensitive else {}

    # --- Persist outputs ---
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON summary
    summary = {
        "overall": metrics_overall,
        "shift_per_feature": shift,
        "group_shift_per_feature": group_shift,
        "notes": "Wasserstein and KS compare original W in the negative cohort vs top-1 successful CF W'.",
    }
    (out_dir / f"{Path(cfg_path).stem}_metrics.json").write_text(json.dumps(summary, indent=2))

    # CSV (per-feature)
    rows = []
    for w, m in shift.items():
        rows.append({"feature": w, "wasserstein": m["wasserstein"], "ks": m["ks"]})
    pd.DataFrame(rows).to_csv(out_dir / f"{Path(cfg_path).stem}_shift.csv", index=False)

    # CSV (groupwise) if present
    if group_shift:
        grows = []
        for g, per_w in group_shift.items():
            for w, m in per_w.items():
                grows.append({"group": g, "feature": w, "wasserstein": m["wasserstein"], "ks": m["ks"]})
        pd.DataFrame(grows).to_csv(out_dir / f"{Path(cfg_path).stem}_group_shift.csv", index=False)

    print(f"Wrote metrics to {out_dir / f'{Path(cfg_path).stem}_metrics.json'}")
    print(f"Wrote per-feature shift to {out_dir / f'{Path(cfg_path).stem}_shift.csv'}")
    if group_shift:
        print(f"Wrote groupwise shift to {out_dir / f'{Path(cfg_path).stem}_group_shift.csv'}")
