import matplotlib.pyplot as plt
import numpy as np
import torch

def compute_metrics(results, X_input, name):
    """Compute metrics with focus on minority group outcomes."""
    init_probs = results['initial_probs'].cpu().numpy()
    final_probs = results['final_probs'].cpu().numpy()
    X_np = X_input.cpu().numpy()
    
    improvement = final_probs - init_probs
    success = final_probs > 0.5
    
    # Group masks
    is_minority = X_np == 1
    is_majority = X_np == 0
    
    metrics = {
        'name': name,
        # Overall metrics
        'mean_improvement': improvement.mean(),
        'success_rate': success.mean(),
        'init_disparity': results['initial_disparity'],
        'final_disparity': results['final_disparity'],
        'disparity_reduction': results['initial_disparity'] - results['final_disparity'],
        'disparity_reduction_pct': (results['initial_disparity'] - results['final_disparity']) / max(results['initial_disparity'], 1e-6) * 100,
        'mean_cost': results['delta_norm'].mean().item(),
        
        # Majority (X=0) metrics - fairness anchor
        'improvement_majority': improvement[is_majority].mean() if is_majority.sum() > 0 else np.nan,
        'success_majority': success[is_majority].mean() if is_majority.sum() > 0 else np.nan,
        'final_prob_majority': final_probs[is_majority].mean() if is_majority.sum() > 0 else np.nan,
        
        # Minority (X=1) metrics - PRIMARY FOCUS
        'improvement_minority': improvement[is_minority].mean() if is_minority.sum() > 0 else np.nan,
        'success_minority': success[is_minority].mean() if is_minority.sum() > 0 else np.nan,
        'final_prob_minority': final_probs[is_minority].mean() if is_minority.sum() > 0 else np.nan,
        
        # Cost by group
        'cost_majority': results['delta_norm'][torch.tensor(is_majority)].mean().item() if is_majority.sum() > 0 else np.nan,
        'cost_minority': results['delta_norm'][torch.tensor(is_minority)].mean().item() if is_minority.sum() > 0 else np.nan,
    }
    return metrics

def plot_results(results_nofair, results_fair, metrics_nofair, metrics_fair, X_recourse_input):
    """
    Generates comparison plots for fairness recourse.
    
    Args:
        results_nofair (dict): Output from RecourseOptimizer with lambda_fair=0
        results_fair (dict): Output from RecourseOptimizer with lambda_fair>0
        metrics_nofair (dict): Dictionary with 'success_majority', 'success_minority' keys
        metrics_fair (dict): Dictionary with 'success_majority', 'success_minority' keys
        X_recourse_input (Tensor): The sensitive attribute tensor for the recourse batch
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Optimization trajectories
    ax = axes[0, 0]
    ax.plot(results_nofair['history']['disparity'], label='No Fairness', color='gray', linewidth=2)
    ax.plot(results_fair['history']['disparity'], label='With Fairness', color='green', linewidth=2)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Disparity')
    ax.set_title('Disparity During Optimization')
    ax.legend()
    ax.axhline(0, color='black', linestyle=':', alpha=0.3)  # Zero disparity reference

    ax = axes[0, 1]
    ax.plot(results_nofair['history']['prob_loss'], label='No Fairness', color='gray', linewidth=2)
    ax.plot(results_fair['history']['prob_loss'], label='With Fairness', color='green', linewidth=2)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Probability Loss')
    ax.set_title('Probability Loss During Optimization')
    ax.legend()

    ax = axes[0, 2]
    ax.plot(results_nofair['history']['cost_loss'], label='No Fairness', color='gray', linewidth=2)
    ax.plot(results_fair['history']['cost_loss'], label='With Fairness', color='green', linewidth=2)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Cost Loss')
    ax.set_title('Cost During Optimization')
    ax.legend()

    # Row 2: Outcome comparisons
    # Ensure X_np is available
    X_np = X_recourse_input.cpu().numpy()

    ax = axes[1, 0]
    # Initial vs Final probabilities by group (No Fairness)
    init_probs = results_nofair['initial_probs'].cpu().numpy()
    final_probs = results_nofair['final_probs'].cpu().numpy()
    ax.scatter(init_probs[X_np==0], final_probs[X_np==0], alpha=0.4, c='blue', label='Majority (X=0)', s=20)
    ax.scatter(init_probs[X_np==1], final_probs[X_np==1], alpha=0.4, c='red', label='Minority (X=1)', s=20)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='No change')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Initial P(Y=1)')
    ax.set_ylabel('Final P(Y=1)')
    ax.set_title('Without Fairness Constraint (λ=0)')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1, 1]
    # Initial vs Final probabilities by group (With Fairness)
    init_probs = results_fair['initial_probs'].cpu().numpy()
    final_probs = results_fair['final_probs'].cpu().numpy()
    ax.scatter(init_probs[X_np==0], final_probs[X_np==0], alpha=0.4, c='blue', label='Majority (X=0)', s=20)
    ax.scatter(init_probs[X_np==1], final_probs[X_np==1], alpha=0.4, c='red', label='Minority (X=1)', s=20)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='No change')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Initial P(Y=1)')
    ax.set_ylabel('Final P(Y=1)')
    ax.set_title('With Fairness Constraint')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


    # Add mean markers to the scatter plots
    for ax, results, title in [(axes[1,0], results_nofair, 'No Fairness'), 
                               (axes[1,1], results_fair, 'With Fairness')]:
        final_probs = results['final_probs'].cpu().numpy()
        
        # Plot group means as larger markers
        if (X_np==0).sum() > 0:
            mean_maj = final_probs[X_np==0].mean()
            ax.axhline(mean_maj, color='blue', linestyle='--', alpha=0.7, linewidth=2)
        else:
            mean_maj = 0
            
        if (X_np==1).sum() > 0:
            mean_min = final_probs[X_np==1].mean()
            ax.axhline(mean_min, color='red', linestyle='--', alpha=0.7, linewidth=2)
        else:
            mean_min = 0
        
        # Annotate the disparity
        disparity = abs(mean_maj - mean_min)
        ax.annotate(f'Disparity: {disparity:.3f}', 
                    xy=(0.02, 0.92), xycoords='axes fraction',
                    fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax = axes[1, 2]
    # Bar chart: Success rate by group
    x_pos = np.arange(2)
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, 
                   [metrics_nofair['success_majority'], metrics_nofair['success_minority']], 
                   width, label='No Fairness', color='gray', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, 
                   [metrics_fair['success_majority'], metrics_fair['success_minority']], 
                   width, label='With Fairness', color='green', alpha=0.7)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate by Group')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Majority (X=0)', 'Minority (X=1)'])
    ax.legend()
    ax.set_ylim(0, 1.1)  # Extra space for labels

    plt.tight_layout()
    plt.savefig('fairness_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved as fairness_comparison.png")