# counterfactual_flows/generate_cf.py
import yaml
import torch
import joblib
import numpy as np
from pathlib import Path

from .schemas import load_schema
from .datasets import load_dataframe
from .flows import CondFlow
from .constraints import project_to_actionable
from .utils import device_of

def _get_cost(x_orig, x_prime, schema, recourse_cfg):
    """Notebook-style fixed cost by default. Optionally supports L1 if requested."""
    ctype = recourse_cfg.get("cost_type", "fixed")  # "fixed" | "l1"
    if ctype == "fixed":
        return float(recourse_cfg.get("fixed_cost", 1.0))
    elif ctype == "l1":
        # lightweight inline L1, no external costs.py needed
        cost = 0.0
        for f, meta in schema.actionable.items():
            cost += abs(float(x_prime[f]) - float(x_orig[f])) * float(meta.cost_per_unit)
        return float(cost)
    else:
        raise ValueError(f"Unknown cost_type: {ctype}")

def _build_feature_vector(x_prime, cond_cols, w_cols):
    """Order features exactly as in training: [cond..., W...]"""
    return [float(x_prime[c]) for c in cond_cols + w_cols]

def run_generate(cfg_path: str):
    # --- Load checkpoint & config ---
    ckpt_path = Path("models") / (Path(cfg_path).stem + "_flow.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    schema = load_schema(cfg)

    device = device_of()

    # --- Data & classifier ---
    df = load_dataframe(cfg["dataset"]["path"], cfg["dataset"]["format"])
    clf = joblib.load(cfg["recourse"]["classifier_path"])

    # Instances needing recourse: Y == 0
    need = df[df[cfg["dataset"]["target"]] == 0].copy().reset_index(drop=True)

    cond_cols = schema.partitions.X + schema.partitions.Z
    w_cols = schema.partitions.W

    # --- Build conditional flow model & load weights (Zuko) ---
    dim_w = len(w_cols)
    dim_c = len(cond_cols)
    model = CondFlow(dim_w=dim_w, dim_cond=dim_c, cfg=cfg["model"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # --- Prepare conditioning tensor ---
    C = torch.tensor(need[cond_cols].values, dtype=torch.float32, device=device)

    # --- Sample mediators with Zuko ---
    # Zuko API: get conditional distribution and sample k draws per row
    k = int(cfg["recourse"].get("k_samples_per_x", 128))
    thresh = float(cfg["recourse"].get("accept_if_prob_above", 0.5))
    topk = int(cfg["recourse"].get("topk_return", 5))

    with torch.no_grad():
        # Obtain conditional distribution pθ(W | C)
        dist = model.flow(context=C)  # returns a Distribution conditioned on C
        # Sample shape: (k, B, |W|)
        Wsamp = dist.sample((k,)).cpu().numpy()

    # Reorder to [B, k, |W|] for easier iteration
    Wsamp = np.transpose(Wsamp, (1, 0, 2))

    # --- Build candidates and score with classifier ---
    results = []
    for i, row in need.iterrows():
        x_orig = row.to_dict()
        cands = []
        for s in range(k):
            x_prime = dict(x_orig)
            # plug new W
            for j, wcol in enumerate(w_cols):
                x_prime[wcol] = float(Wsamp[i, s, j])

            # project to actionability/immutability
            x_prime = project_to_actionable(x_orig, x_prime, schema)

            # score with your classifier f(x') using the same feature order as training: [cond, W]
            fv = _build_feature_vector(x_prime, cond_cols, w_cols)
            p1 = float(clf.predict_proba([fv])[0][1])

            if p1 >= thresh:
                cost = _get_cost(x_orig, x_prime, schema, cfg["recourse"])
                cands.append((p1, cost, x_prime))

        # Rank: higher success prob first, then lower cost
        cands.sort(key=lambda t: (-t[0], t[1]))
        results.append(cands[:topk])

    out = Path("outputs") / (Path(cfg_path).stem + "_counterfactuals.npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, results)
    print(f"Saved counterfactuals to {out}")
