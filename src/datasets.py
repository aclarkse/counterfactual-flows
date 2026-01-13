import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataHandler:
    def __init__(self, config):
        self.cfg = config
        self.scalers = {}
        
    def load_data(self):
        # Dispatcher for different datasets
        if self.cfg['dataset']['name'] == 'synthetic_sfm':
            df = self._generate_synthetic()
        else:
            # Load from CSV for real datasets
            path = self.cfg['dataset'].get('path')
            df = pd.read_csv(path)
            
        self.df = df
        return self._process_data(df)

    def _generate_synthetic(self, n_samples=5000):
        """
        Generates synthetic data with clear bias against minority group.
        
        The minority group (X=1) has:
        - Lower mediator values (W1, W2) due to structural disadvantage
        - Lower probability of positive outcome (Y=1)
        - This can be mitigated by generating counterfactuals for the mediators (W1, W2).
        """
        np.random.seed(42)
        X = np.random.binomial(1, 0.3, n_samples)
        Z = np.random.normal(0, 1, n_samples)
        W1 = -1.0 * X + 0.6 * Z + 0.4 * np.random.normal(0, 1, n_samples)
        W2 = -0.6 * X + 0.4 * Z + 0.3 * W1 + 0.4 * np.random.normal(0, 1, n_samples)
        logits = 1.2 * W1 + 1.0 * W2 + 0.3 * Z - 0.5
        prob_y = 1 / (1 + np.exp(-logits))
        Y = np.random.binomial(1, prob_y)
        return pd.DataFrame({'X': X, 'Z': Z, 'W1': W1, 'W2': W2, 'Y': Y})

    def _process_data(self, df):
        d_cfg = self.cfg['dataset']
        
        # Split features
        self.X_sens = df[d_cfg['sensitive_attribute']].values
        self.Z = df[d_cfg['confounders']].values
        self.W = df[d_cfg['mediators']].values
        self.Y = df[d_cfg['target']].values
        
        # Scale continuous variables (Mediators and Confounders usually)
        self.scalers['W'] = StandardScaler().fit(self.W)
        self.W_scaled = self.scalers['W'].transform(self.W)
        
        # Split train/test
        data = train_test_split(
            self.W_scaled, self.Z, self.X_sens, self.Y,
            test_size=d_cfg['test_size'], random_state=42
        )
        
        # Pack into a standardized dictionary
        keys = ['W_train', 'W_test', 'Z_train', 'Z_test', 'X_train', 'X_test', 'Y_train', 'Y_test']
        self.data_dict = dict(zip(keys, data))
        
        return self.data_dict

    def get_tensors(self, device):
        """Convert stored numpy arrays to torch tensors"""
        tensors = {}
        for k, v in self.data_dict.items():
            tensors[k] = torch.FloatTensor(v).to(device)
        return tensors