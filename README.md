# Counterfactual Flows

A modular pipeline for generating and evaluating counterfactuals using **conditional normalizing flows (Zuko)**.

## Features
- Train a conditional flow model \( P(W \mid X, Z) \)
- Generate counterfactuals \( W' \sim P_\theta(W \mid X,Z) \)
- Evaluate success and distribution shift (Wasserstein, KS)

## Usage
```bash
python -m counterfactual_flows.cli train config/adult.yaml
python -m counterfactual_flows.cli cf config/adult.yaml
python -m counterfactual_flows.cli eval config/adult.yaml
