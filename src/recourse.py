import torch
import torch.optim as optim

class RecourseOptimizer:
    def __init__(self, flow, outcome_model, config, device):
        self.flow = flow
        self.outcome_model = outcome_model
        self.cfg = config
        self.device = device

    def generate(self, W, Z, X, lambda_fair_override=None):
        # Latent initialization
        context = torch.cat([X.unsqueeze(1), Z], dim=1)
        dist = self.flow.flow(context)
        with torch.no_grad():
            z_init = dist.transform(W)
            initial_probs = self.outcome_model.predict_proba_tensor(W, Z)

        # Optimization Setup
        delta = torch.zeros_like(z_init, requires_grad=True, device=self.device)
        optimizer = optim.Adam([delta], lr=self.cfg['lr'])
        
        target_prob = self.cfg['target_prob']
        lambda_cost = self.cfg['lambda_cost']
        lambda_fair = lambda_fair_override if lambda_fair_override is not None else self.cfg['lambda_fair']
        
        # Group Masks & Counts
        mask_0 = (X == 0).float()
        mask_1 = (X == 1).float()
        n_0 = mask_0.sum().clamp(min=1)
        n_1 = mask_1.sum().clamp(min=1)

        # --- Calculate Initial Disparity for Metrics ---
        init_mean_0 = (initial_probs * mask_0).sum() / n_0
        init_mean_1 = (initial_probs * mask_1).sum() / n_1
        initial_disparity = abs(init_mean_0 - init_mean_1).item()

        history = {'disparity': [], 'prob_loss': [], 'cost_loss': []}

        # Optimization Loop
        for step in range(self.cfg['n_steps']):
            optimizer.zero_grad()
            
            z_new = z_init + delta
            W_new = dist.transform(z_new)
            probs = self.outcome_model.predict_proba_tensor(W_new, Z)
            
            # Losses
            loss_validity = ((probs - target_prob) ** 2).mean()
            loss_cost = (delta ** 2).sum(dim=1).mean()
            
            mean_0 = (probs * mask_0).sum() / n_0
            mean_1 = (probs * mask_1).sum() / n_1
            disparity_val = (mean_0 - mean_1) ** 2
            
            loss = loss_validity + lambda_cost * loss_cost + lambda_fair * disparity_val
            
            loss.backward()
            optimizer.step()
            
            # History
            history['disparity'].append(abs(mean_0.item() - mean_1.item()))
            history['prob_loss'].append(loss_validity.item())
            history['cost_loss'].append(loss_cost.item())

        # Final Calculations
        with torch.no_grad():
            W_recourse = dist.transform(z_init + delta)
            final_probs = self.outcome_model.predict_proba_tensor(W_recourse, Z)
            
            final_mean_0 = (final_probs * mask_0).sum() / n_0
            final_mean_1 = (final_probs * mask_1).sum() / n_1
            final_disparity = abs(final_mean_0 - final_mean_1).item()
            
            # Calculate norm for cost metrics
            delta_norm = torch.norm(delta, p=2, dim=1)

        return {
            'W_recourse': W_recourse,
            'delta': delta,
            'delta_norm': delta_norm,
            'history': history,
            'initial_probs': initial_probs,
            'final_probs': final_probs,
            'initial_disparity': initial_disparity,
            'final_disparity': final_disparity
        }