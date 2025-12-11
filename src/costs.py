from .schemas import Schema

def l1_cost(x_orig: dict, x_prime: dict, schema: Schema) -> float:
    c = 0.0
    for f, acfg in schema.actionable.items():
        c += abs(x_prime[f] - x_orig[f]) * acfg.cost_per_unit
    return c
