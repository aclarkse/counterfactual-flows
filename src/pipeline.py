import torch
import numpy as np
import pandas as pd
import os
from datetime import datetime
from .data import DataHandler
from .models import CausalFlow, OutcomeModel
from .recourse import RecourseOptimizer
from .utils import compute_metrics

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Running without experiment tracking.")


class Pipeline:
    def __init__(self, config_path, use_wandb=True,
                  wandb_project="fairness-recourse",
                  wandb_entity="andrea-c-sev-columbia-university"):
        import yaml
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup Logging
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"experiment_results_{timestamp}.csv")
        
        # Wandb setup
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        
    def _init_wandb(self, run_name=None, tags=None):
        """Initialize wandb run with config."""
        if not self.use_wandb:
            return
        
        wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,  # <-- Added entity
            name=run_name,
            tags=tags or [],
            config=self.cfg,
            reinit=True
        )
        
    def _log_wandb(self, metrics_dict, step=None):
        """Log metrics to wandb."""
        if not self.use_wandb:
            return
        if step is not None:
            wandb.log(metrics_dict, step=step)
        else:
            wandb.log(metrics_dict)
    
    def _finish_wandb(self):
        """Finish wandb run."""
        if self.use_wandb:
            wandb.finish()
        
    def run(self):
        # Initialize main wandb run for full experiment
        self._init_wandb(
            run_name=f"full_experiment_{datetime.now().strftime('%H%M%S')}",
            tags=["full_run", self.cfg.get('data', {}).get('dataset', 'unknown')]
        )
        
        # 1. Load Data
        print(">>> Loading data...")
        data_handler = DataHandler(self.cfg)
        data_dict = data_handler.load_data()
        tensors = data_handler.get_tensors(self.device)
        
        w_dim = data_dict['W_train'].shape[1]
        z_dim = data_dict['Z_train'].shape[1]
        
        # Log data stats
        self._log_wandb({
            "data/n_train": len(data_dict['W_train']),
            "data/n_test": len(data_dict['W_test']),
            "data/w_dim": w_dim,
            "data/z_dim": z_dim,
            "data/minority_pct_train": (data_dict['X_train'] == 1).mean() * 100,
            "data/positive_pct_train": data_dict['Y_train'].mean() * 100,
        })

        # 2. Train Causal Flow (Shared across all outcome models)
        print("\n" + "="*50)
        print(">>> Training Causal Flow P(W | Z, X)")
        print("="*50)
        flow = CausalFlow(w_dim, z_dim + 1, self.cfg['models']['flow'], self.device)
        
        context_train = torch.cat([tensors['X_train'].unsqueeze(1), tensors['Z_train']], dim=1)
        flow_history = flow.fit(tensors['W_train'], context_train)
        
        # Log flow training (if history available)
        if flow_history is not None and isinstance(flow_history, list):
            for step, loss in enumerate(flow_history):
                self._log_wandb({"flow/train_loss": loss}, step=step)
        
        # 3. Iterate through Outcome Models
        model_list = self.cfg['models']['model_types_to_run']
        all_metrics = []
        
        for model_idx, model_name in enumerate(model_list):
            print("\n" + "#"*60)
            print(f" PROCESSING MODEL: {model_name}")
            print("#"*60)
            
            # Construct specific config for this model
            outcome_cfg = self.cfg['models']['outcome_params'].copy()
            outcome_cfg['type'] = model_name
            
            # A. Train Outcome Model & Surrogate
            outcome = OutcomeModel(w_dim, z_dim, outcome_cfg, self.device)
            outcome.fit(
                data_dict['W_train'], data_dict['Z_train'], data_dict['Y_train'],
                W_test=data_dict['W_test'], Z_test=data_dict['Z_test'], Y_test=data_dict['Y_test']
            )
            
            # Log sklearn model performance
            if hasattr(outcome, 'train_metrics'):
                self._log_wandb({
                    f"sklearn/{model_name}/train_acc": outcome.train_metrics.get('acc', 0),
                    f"sklearn/{model_name}/test_acc": outcome.test_metrics.get('acc', 0),
                    f"sklearn/{model_name}/train_auc": outcome.train_metrics.get('auc', 0),
                    f"sklearn/{model_name}/test_auc": outcome.test_metrics.get('auc', 0),
                })
            
            if hasattr(outcome, 'surrogate_fidelity'):
                self._log_wandb({
                    f"surrogate/{model_name}/correlation": outcome.surrogate_fidelity.get('corr', 0),
                    f"surrogate/{model_name}/mae": outcome.surrogate_fidelity.get('mae', 0),
                })
            
            # B. Select Candidates (Specific to this model's predictions)
            with torch.no_grad():
                preds = outcome.predict_proba_tensor(tensors['W_test'], tensors['Z_test'])
            
            candidates_mask = (preds < 0.5)
            if candidates_mask.sum() == 0:
                print(f"Skipping {model_name}: No candidates found with P(Y=1) < 0.5")
                continue

            W_cand = tensors['W_test'][candidates_mask]
            Z_cand = tensors['Z_test'][candidates_mask]
            X_cand = tensors['X_test'][candidates_mask]
            
            # Log candidate stats
            X_cand_np = X_cand.cpu().numpy()
            self._log_wandb({
                f"candidates/{model_name}/total": len(W_cand),
                f"candidates/{model_name}/minority": (X_cand_np == 1).sum(),
                f"candidates/{model_name}/majority": (X_cand_np == 0).sum(),
                f"candidates/{model_name}/minority_pct": (X_cand_np == 1).mean() * 100,
            })
            
            print(f"Selected {len(W_cand)} candidates for recourse.")
            
            # C. Run Recourse Optimization
            optimizer = RecourseOptimizer(flow, outcome, self.cfg['recourse'], self.device)
            
            # Run Baseline (No Fairness)
            print(f"--- Running Baseline Recourse ({model_name}) ---")
            res_base = optimizer.generate(W_cand, Z_cand, X_cand, lambda_fair_override=0.0)
            metrics_base = compute_metrics(res_base, X_cand, f"{model_name}_Baseline")
            metrics_base['model_type'] = model_name
            metrics_base['constraint'] = 'None'
            all_metrics.append(metrics_base)
            
            # Log baseline results
            self._log_wandb({
                f"results/{model_name}/baseline/success_rate": metrics_base['success_rate'],
                f"results/{model_name}/baseline/success_minority": metrics_base['success_minority'],
                f"results/{model_name}/baseline/success_majority": metrics_base['success_majority'],
                f"results/{model_name}/baseline/init_disparity": metrics_base['init_disparity'],
                f"results/{model_name}/baseline/final_disparity": metrics_base['final_disparity'],
                f"results/{model_name}/baseline/disparity_reduction_pct": metrics_base.get('disparity_reduction_pct', 0),
                f"results/{model_name}/baseline/mean_cost": metrics_base['mean_cost'],
            })
            
            # Run Fairness
            print(f"--- Running Fair Recourse ({model_name}) ---")
            res_fair = optimizer.generate(W_cand, Z_cand, X_cand)
            metrics_fair = compute_metrics(res_fair, X_cand, f"{model_name}_Fair")
            metrics_fair['model_type'] = model_name
            metrics_fair['constraint'] = 'Fairness'
            all_metrics.append(metrics_fair)
            
            # Log fair results
            self._log_wandb({
                f"results/{model_name}/fair/success_rate": metrics_fair['success_rate'],
                f"results/{model_name}/fair/success_minority": metrics_fair['success_minority'],
                f"results/{model_name}/fair/success_majority": metrics_fair['success_majority'],
                f"results/{model_name}/fair/init_disparity": metrics_fair['init_disparity'],
                f"results/{model_name}/fair/final_disparity": metrics_fair['final_disparity'],
                f"results/{model_name}/fair/disparity_reduction_pct": metrics_fair.get('disparity_reduction_pct', 0),
                f"results/{model_name}/fair/mean_cost": metrics_fair['mean_cost'],
            })
            
            # Log comparison (fairness benefit)
            self._log_wandb({
                f"comparison/{model_name}/disparity_improvement": metrics_base['final_disparity'] - metrics_fair['final_disparity'],
                f"comparison/{model_name}/minority_success_improvement": metrics_fair['success_minority'] - metrics_base['success_minority'],
                f"comparison/{model_name}/cost_difference": metrics_fair['mean_cost'] - metrics_base['mean_cost'],
            })
            
            # Save intermediate results
            self._save_logs(all_metrics)

        # Log summary table to wandb
        if self.use_wandb and all_metrics:
            summary_table = wandb.Table(dataframe=pd.DataFrame(all_metrics))
            wandb.log({"summary_table": summary_table})
        
        # Finish wandb run
        self._finish_wandb()
        
        print(f"\n>>> Experiment Complete. Results saved to {self.log_file}")
        return all_metrics

    def run_ablation(self, lambda_fair_values=None):
        """
        Run ablation study over lambda_fair values.
        Creates separate wandb runs for each configuration.
        """
        if lambda_fair_values is None:
            lambda_fair_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
        
        # 1. Load Data (once)
        print(">>> Loading data...")
        data_handler = DataHandler(self.cfg)
        data_dict = data_handler.load_data()
        tensors = data_handler.get_tensors(self.device)
        
        w_dim = data_dict['W_train'].shape[1]
        z_dim = data_dict['Z_train'].shape[1]

        # 2. Train Flow (once)
        print("\n>>> Training Causal Flow P(W | Z, X)")
        flow = CausalFlow(w_dim, z_dim + 1, self.cfg['models']['flow'], self.device)
        context_train = torch.cat([tensors['X_train'].unsqueeze(1), tensors['Z_train']], dim=1)
        flow.fit(tensors['W_train'], context_train)
        
        # 3. Train Outcome Models (once)
        model_list = self.cfg['models']['model_types_to_run']
        outcome_models = {}
        
        for model_name in model_list:
            outcome_cfg = self.cfg['models']['outcome_params'].copy()
            outcome_cfg['type'] = model_name
            outcome = OutcomeModel(w_dim, z_dim, outcome_cfg, self.device)
            outcome.fit(
                data_dict['W_train'], data_dict['Z_train'], data_dict['Y_train'],
                W_test=data_dict['W_test'], Z_test=data_dict['Z_test'], Y_test=data_dict['Y_test']
            )
            outcome_models[model_name] = outcome
        
        # 4. Run ablation over lambda_fair
        all_ablation_results = []
        
        for lambda_fair in lambda_fair_values:
            print(f"\n{'='*60}")
            print(f" ABLATION: lambda_fair = {lambda_fair}")
            print(f"{'='*60}")
            
            for model_name, outcome in outcome_models.items():
                # Initialize wandb run for this config
                self._init_wandb(
                    run_name=f"{model_name}_lambda{lambda_fair}",
                    tags=["ablation", model_name, f"lambda_{lambda_fair}"]
                )
                
                # Log config
                self._log_wandb({
                    "config/model_name": model_name,
                    "config/lambda_fair": lambda_fair,
                    "config/lambda_cost": self.cfg['recourse'].get('lambda_cost', 0.1),
                })
                
                # Get candidates
                with torch.no_grad():
                    preds = outcome.predict_proba_tensor(tensors['W_test'], tensors['Z_test'])
                
                candidates_mask = (preds < 0.5)
                if candidates_mask.sum() == 0:
                    self._finish_wandb()
                    continue
                
                W_cand = tensors['W_test'][candidates_mask]
                Z_cand = tensors['Z_test'][candidates_mask]
                X_cand = tensors['X_test'][candidates_mask]
                
                # Override lambda_fair in config
                recourse_cfg = self.cfg['recourse'].copy()
                recourse_cfg['lambda_fair'] = lambda_fair
                
                optimizer = RecourseOptimizer(flow, outcome, recourse_cfg, self.device)
                results = optimizer.generate(W_cand, Z_cand, X_cand)
                metrics = compute_metrics(results, X_cand, f"{model_name}_lambda{lambda_fair}")
                
                # Add config info
                metrics['model_type'] = model_name
                metrics['lambda_fair'] = lambda_fair
                all_ablation_results.append(metrics)
                
                # Log to wandb
                self._log_wandb({
                    "results/success_rate": metrics['success_rate'],
                    "results/success_minority": metrics['success_minority'],
                    "results/success_majority": metrics['success_majority'],
                    "results/init_disparity": metrics['init_disparity'],
                    "results/final_disparity": metrics['final_disparity'],
                    "results/disparity_reduction_pct": metrics.get('disparity_reduction_pct', 0),
                    "results/mean_cost": metrics['mean_cost'],
                })
                
                self._finish_wandb()
        
        # Save all ablation results
        ablation_df = pd.DataFrame(all_ablation_results)
        ablation_file = os.path.join(self.log_dir, f"ablation_results_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv")
        ablation_df.to_csv(ablation_file, index=False)
        print(f"\n>>> Ablation complete. Results saved to {ablation_file}")
        
        return all_ablation_results

    def _save_logs(self, metrics_list):
        """Helper to save list of dicts to CSV"""
        df = pd.DataFrame(metrics_list)
        # Reorder columns for readability
        cols = ['model_type', 'constraint', 'success_rate', 'success_majority', 'success_minority', 
                'init_disparity', 'final_disparity', 'disparity_reduction_pct', 'mean_cost']
        
        # Ensure only existing columns are selected
        cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
        
        df = df[cols]
        df.to_csv(self.log_file, index=False)
        print(f"Logs updated: {self.log_file}")