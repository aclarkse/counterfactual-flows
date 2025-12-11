import pandas as pd
import numpy as np
from pathlib import Path
from .schemas import Schema
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class Encoders:
    def __init__(self, ohe: OneHotEncoder, scaler: StandardScaler, cat_cols, cont_cols):
        self.ohe, self.scaler = ohe, scaler
        self.cat_cols, self.cont_cols = cat_cols, cont_cols

def load_dataframe(path: str, fmt: str) -> pd.DataFrame:
    p = Path(path)
    if fmt == "csv":     return pd.read_csv(p)
    if fmt == "parquet": return pd.read_parquet(p)
    raise ValueError(f"Unknown format: {fmt}")

def fit_encoders(df: pd.DataFrame, schema: Schema, cfg: dict) -> Encoders:
    cat = schema.encodings["categorical"]
    cont = schema.encodings["continuous"]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(df[cat])
    scaler = StandardScaler()
    scaler.fit(df[cont])
    return Encoders(ohe, scaler, cat, cont)

def transform(df: pd.DataFrame, enc: Encoders, schema: Schema):
    Xcat = enc.ohe.transform(df[enc.cat_cols]) if enc.cat_cols else np.empty((len(df),0))
    Xcont = enc.scaler.transform(df[enc.cont_cols]) if enc.cont_cols else np.empty((len(df),0))
    X_all = np.hstack([Xcat, Xcont])

    # recover column slices for (X,Z,W) within encoded space via index maps if needed
    # For simplicity here, we keep W only from continuous/int features (recommended).
    # If W has categoricals, model via Gumbel-Softmax or separate heads.
    return X_all, Xcat, Xcont

def split(df: pd.DataFrame, ratios):
    n = len(df); ntr=int(n*ratios[0]); nva=int(n*ratios[1])
    return np.split(df.sample(frac=1.0, random_state=42), [ntr, ntr+nva])
