import torch, random, numpy as np
from pathlib import Path
import joblib

def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def device_of():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_classifier(path: str):
    # expects a scikit-learn classifier with predict_proba
    # (Or adapt to your torch model wrapper.)
    return joblib.load(Path(path))
