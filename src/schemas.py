from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Partition:
    X: List[str]
    Z: List[str]
    W: List[str]

@dataclass
class FeatureCfg:
    type: str               # continuous|integer|categorical|binary
    min: Optional[float]=None
    max: Optional[float]=None
    step: Optional[float]=None
    cost_per_unit: float=1.0

@dataclass
class Schema:
    target: str
    sensitive: Optional[str]
    partitions: Partition
    immutable: List[str]
    actionable: Dict[str, FeatureCfg]
    encodings: Dict[str, List[str]]
    standardize: List[str]

def load_schema(cfg: dict) -> Schema:
    p = cfg["dataset"]["partitions"]
    actionable = {k: FeatureCfg(**v) for k,v in cfg["dataset"]["actionable"].items()}
    return Schema(
        target=cfg["dataset"]["target"],
        sensitive=cfg["dataset"].get("sensitive"),
        partitions=Partition(X=p["X"], Z=p["Z"], W=p["W"]),
        immutable=cfg["dataset"]["immutable"],
        actionable=actionable,
        encodings=cfg["dataset"]["encodings"],
        standardize=cfg["dataset"]["standardize"],
    )
