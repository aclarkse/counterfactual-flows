# counterfactual_flows/flows.py
import torch
import torch.nn as nn
from zuko.flows import NSF  # Neural Spline Flow (conditionally)
from zuko.distributions import DiagNormal

class CondFlow(nn.Module):
    """
    Conditional flow Pθ(W | X,Z) implemented with Zuko.
    """
    def __init__(self, dim_w: int, dim_cond: int, cfg: dict):
        super().__init__()
        self.dim_w = dim_w
        self.dim_cond = dim_cond

        # Conditional NSF — matches nflows.MaskedUMNNAutoregressiveTransform
        self.flow = NSF(
            features=dim_w,
            context=dim_cond,
            hidden_features=cfg.get("hidden_features", 128),
            num_transforms=cfg.get("num_transforms", 8),
            bins=cfg.get("num_bins", 8)
        )

        # Base distribution (Zuko uses flexible base objects)
        self.base = DiagNormal(dim_w)

    def forward_logprob(self, w: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Compute log pθ(w | cond)
        return self.flow(w, context=cond).log_prob(w)

    def sample(self, cond: torch.Tensor, num_samples: int) -> torch.Tensor:
        # Sample W ~ pθ(. | cond)
        return self.flow.sample(num_samples, context=cond)
