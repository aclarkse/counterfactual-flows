import numpy as np
from .schemas import Schema

def project_to_actionable(x_orig: dict, x_prime: dict, schema: Schema):
    # clamp actionable features; forbid changes to immutable
    for f in schema.immutable:
        x_prime[f] = x_orig[f]
    for f, acfg in schema.actionable.items():
        if f in x_prime:
            x_prime[f] = float(np.clip(x_prime[f], acfg.min, acfg.max))
    return x_prime

def violates_graph_edges(x_prime: dict, parents: dict=None):
    # Hook: optionally check simple parent-child monotonic constraints or tolerances
    return False
