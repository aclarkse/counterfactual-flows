import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from zuko.flows import NSF
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

class CausalFlow:
    def __init__(self, w_dim, context_dim, config, device):
        self.device = device
        self.flow = NSF(
            features=w_dim,
            context=context_dim,
            transforms=config['transforms'],
            hidden_features=config['hidden_features']
        ).to(device)
        self.lr = config['lr']
        self.epochs = config['epochs']

    def fit(self, W, Context):
        optimizer = optim.Adam(self.flow.parameters(), lr=self.lr)
        dataset = TensorDataset(W, Context)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        print("Training Causal Flow...")
        self.flow.train()
        for epoch in range(self.epochs):
            for w_batch, ctx_batch in loader:
                optimizer.zero_grad()
                loss = -self.flow(ctx_batch).log_prob(w_batch).mean()
                loss.backward()
                optimizer.step()
        self.flow.eval()

class DifferentiableSurrogate(nn.Module):
    """
    A PyTorch neural network trained to mimic a sklearn model's predictions.
    Includes Dropout for regularization as per your snippet.
    """
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features).squeeze(-1)

class OutcomeModel:
    def __init__(self, w_dim, z_dim, config, device):
        self.device = device
        self.config = config
        self.input_dim = w_dim + z_dim
        self.train_metrics: dict[str, float] = {}
        self.test_metrics: dict[str, float] = {}
        
        # 1. Initialize specific Sklearn Model based on config
        t = config['type']
        if t == 'LogisticRegression':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif t == 'NeuralNet' or t == 'MLP':
            self.model = MLPClassifier(
                hidden_layer_sizes=(32, 16), 
                max_iter=500, 
                random_state=42,
                early_stopping=True
            )
        elif t == 'RandomForest':
            self.model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42
            )
        elif t == 'GradientBoosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=5, 
                random_state=42
            )
        else:
            raise ValueError(f"Model {t} not implemented")
            
        # 2. Initialize Surrogate Architecture
        self.surrogate = DifferentiableSurrogate(
            input_dim=self.input_dim, 
            hidden_dims=config.get('surrogate_hidden', [64, 32])
        ).to(device)

    def fit(self, W_train, Z_train, Y_train, W_test=None, Z_test=None, Y_test=None):
        """
        Fits the sklearn model and then trains the surrogate to mimic it.
        """
        # --- A. Fit Sklearn Model ---
        print(f"\nTraining Sklearn Model: {self.config['type']}")
        features_train = np.column_stack([W_train, Z_train])
        self.model.fit(features_train, Y_train)
        
        # Evaluation Metrics
        train_pred = self.model.predict(features_train)
        train_acc = accuracy_score(Y_train, train_pred)
        print(f"  Train Acc: {train_acc:.4f}")

        # --- FIX: Added check for Z_test is not None ---
        if W_test is not None and Z_test is not None and Y_test is not None:
            features_test = np.column_stack([W_test, Z_test])
            test_pred = self.model.predict(features_test)
            test_acc = accuracy_score(Y_test, test_pred)
            
            # AUC requires probabilities
            try:
                test_probs = self.model.predict_proba(features_test)[:, 1]
                test_auc = roc_auc_score(Y_test, test_probs)
                print(f"  Test AUC:  {test_auc:.4f}")
            except AttributeError:
                # Handle models that might not have predict_proba (though most classifiers do)
                print("  Test AUC:  N/A (No predict_proba)")
                
            print(f"  Test Acc:  {test_acc:.4f}")

        # --- B. Train Surrogate ---
        print(f"\nTraining Surrogate for {self.config['type']}...")
        
        # Generate soft labels from sklearn model (Target for surrogate)
        sklearn_probs = self.model.predict_proba(features_train)[:, 1]
        
        # Convert to tensors
        features_t = torch.FloatTensor(features_train).to(self.device)
        targets_t = torch.FloatTensor(sklearn_probs).to(self.device)
        
        optimizer = optim.Adam(self.surrogate.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        dataset = TensorDataset(features_t, targets_t)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        self.surrogate.train()
        for epoch in range(self.config.get('surrogate_epochs', 200)):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.surrogate.net(x_batch).squeeze(-1)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
        
        self.surrogate.eval()
        
        # Verify Fidelity
        with torch.no_grad():
            surrogate_probs = self.surrogate.net(features_t).squeeze(-1).cpu().numpy()
            
        corr = np.corrcoef(sklearn_probs, surrogate_probs)[0, 1]
        mae = np.abs(sklearn_probs - surrogate_probs).mean()
        print(f"  Surrogate Fidelity -> Correlation: {corr:.4f} | MAE: {mae:.4f}")

    def predict_proba_sklearn(self, W, Z):
        """Use the underlying sklearn model for inference (requires numpy inputs)."""
        features = np.column_stack([W, Z])
        return self.model.predict_proba(features)[:, 1]

    def predict_proba_tensor(self, W, Z):
        """Use the differentiable surrogate for optimization (requires tensor inputs)."""
        x = torch.cat([W, Z], dim=1)
        return self.surrogate(x)