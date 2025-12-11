# counterfactual_flows/train_flow.py
import math
import yaml
import torch
import joblib
import numpy as np
from pathlib import Path
from dataclasses import asdict
from torch.utils.data import TensorDataset, DataLoader

from .schemas import load_schema
from .datasets import load_dataframe, fit_encoders, transform, split
from .utils import set_seed, device_of
from .flows import CondFlow  # uses Zuko under the hood

# ---- Helpers ---------------------------------------------------------------

def _pack_W_and_cond(df, schema):
    """
    Build raw (W, cond) tensors directly from the dataframe columns.
    This mirrors a notebook-style approach where we don't re-encode W/cond
    beyond what's specified in the config.
    """
    W = torch.tensor(df[schema.partitions.W].values, dtype=torch.float32)
    cond_cols = schema.partitions.X + schema.partitions.Z
    C = torch.tensor(df[cond_cols].values, dtype=torch.float32)
    return W, C

def _make_loader(W, C, bs, shuffle):
    return DataLoader(TensorDataset(W, C), batch_size=bs, shuffle=shuffle, drop_last=False)

def _epoch_loop(model, loader, optimizer=None, scaler=None, clip_grad=None, device=None):
    """
    One pass (train if optimizer is provided, else eval).
    Loss is -log p_theta(w|c).mean() (maximize likelihood).
    """
    is_train = optimizer is not None
    total_ll = 0.0
    count = 0

    if is_train:
        model.train()
    else:
        model.eval()

    for W, C in loader:
        W = W.to(device)
        C = C.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                    dist = model.flow(W, context=C)  # Zuko returns a distribution object
                    logp = dist.log_prob(W)          # [B,]
                    loss = -logp.mean()
                scaler.scale(loss).backward()
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                dist = model.flow(W, context=C)
                logp = dist.log_prob(W)
                loss = -logp.mean()
                loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                optimizer.step()

            total_ll += float(logp.sum().detach().cpu())
            count += W.size(0)
        else:
            with torch.no_grad():
                dist = model.flow(W, context=C)
                logp = dist.log_prob(W)
                total_ll += float(logp.sum().cpu())
                count += W.size(0)

    # average negative log-likelihood per sample for reporting
    avg_nll = -total_ll / max(1, count)
    avg_ll = total_ll / max(1, count)
    return avg_nll, avg_ll

# ---- Main entry ------------------------------------------------------------

def run_train(cfg_path: str):
    """
    Train a conditional flow Pθ(W|X,Z) with Zuko (NSF by default).
    - Maximizes average log-likelihood on train.
    - Early-stops on best validation log-likelihood.
    - Saves a checkpoint with: state_dict, cfg, and an encoder snapshot (for reproducibility).
    """
    cfg = yaml.safe_load(open(cfg_path))
    schema = load_schema(cfg)
    set_seed(cfg["model"]["seed"])

    device = device_of()
    data_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    # Load DF and fit encoders (kept for parity; not strictly required by the flow step)
    df = load_dataframe(data_cfg["path"], data_cfg["format"])
    enc = fit_encoders(df, schema, cfg)

    # Split
    df_tr, df_va, df_te = split(df, data_cfg["train_valid_test"])

    # Prepare raw tensors (no imposed transformation beyond df columns)
    Wtr, Ctr = _pack_W_and_cond(df_tr, schema)
    Wva, Cva = _pack_W_and_cond(df_va, schema)

    # DataLoaders
    bs = model_cfg.get("batch_size", 512)
    tr_dl = _make_loader(Wtr, Ctr, bs, shuffle=True)
    va_dl = _make_loader(Wva, Cva, bs, shuffle=False)

    # Build conditional flow with Zuko backend
    dim_w = Wtr.size(1)
    dim_c = Ctr.size(1)
    model = CondFlow(dim_w=dim_w, dim_cond=dim_c, cfg=model_cfg).to(device)

    # Optimizer + (optional) scheduler
    lr = float(model_cfg.get("lr", 1e-3))
    wd = float(model_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    use_cosine = bool(model_cfg.get("cosine_anneal", False))
    T_max = int(model_cfg.get("cosine_Tmax", max(10, model_cfg.get("epochs", 50))))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max) if use_cosine else None

    # AMP and grad clip knobs (match common notebook toggles)
    use_amp = bool(model_cfg.get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    clip_grad = model_cfg.get("grad_clip", None)

    # Training loop with early stopping on best val log-likelihood
    best_val_ll = -math.inf
    patience = 0
    max_patience = int(model_cfg.get("early_stop_patience", 5))

    ckpt_path = Path("models") / (Path(cfg_path).stem + "_flow.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = int(model_cfg.get("epochs", 50))
    for epoch in range(1, epochs + 1):
        tr_nll, tr_ll = _epoch_loop(
            model, tr_dl, optimizer=optimizer, scaler=scaler, clip_grad=clip_grad, device=device
        )
        va_nll, va_ll = _epoch_loop(model, va_dl, optimizer=None, device=device)

        if scheduler is not None:
            scheduler.step()

        # Early stopping on validation log-likelihood
        improved = va_ll > best_val_ll
        if improved:
            best_val_ll = va_ll
            patience = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "cfg": cfg,
                    "enc": enc,  # snapshot to keep preprocessing consistent
                    "epoch": epoch,
                    "val_ll": best_val_ll,
                },
                ckpt_path,
            )
        else:
            patience += 1

        # Simple console log (notebook-like)
        print(
            f"[Epoch {epoch:03d}] "
            f"train NLL: {tr_nll:.4f} | val NLL: {va_nll:.4f} | "
            f"val LL: {va_ll:.4f} | lr: {optimizer.param_groups[0]['lr']:.2e} "
            f"{'(saved)' if improved else ''}"
        )

        if patience >= max_patience:
            print(f"Early stopping (no val LL improvement in {max_patience} epochs).")
            break

    print(f"Checkpoint: {ckpt_path}")
