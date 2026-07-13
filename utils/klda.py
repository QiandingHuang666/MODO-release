"""
KLDA (Kernel LDA via Random Fourier Features) plugin.
Adapted from ~/exp/KLDA for CLIP image features.

Pipeline: img → CLIP → raw feature (d) → RFF (D) → kernel LDA (Mahalanobis)

Note: batch_update expects (X, y=int) where X is ALL samples of ONE class.
      Accumulate features externally, then call batch_update once per class.
"""
import torch
from torch import nn
from collections import defaultdict


class KLDAPlugin(nn.Module):
    def __init__(self, feature_dim, D=5000, sigma=1e-4, num_ensembles=5, seed=0, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.D = D
        self.sigma_param = sigma
        self.num_ensembles = num_ensembles
        self.seed = seed
        self.device = torch.device("cpu")

        self.omegas = []
        self.bs_params = []
        for i in range(num_ensembles):
            torch.manual_seed(seed + i)
            omega = torch.normal(0,
                torch.sqrt(torch.tensor(2.0 * sigma, dtype=torch.float32)),
                (feature_dim, D))
            b = torch.rand(D) * 2 * torch.pi
            self.omegas.append(omega)
            self.bs_params.append(b)

        self.class_means = [defaultdict(lambda: torch.zeros(D)) for _ in range(num_ensembles)]
        self.sigma_mats = [torch.zeros((D, D)) for _ in range(num_ensembles)]
        self.sigma_invs = [None] * num_ensembles
        self.class_mean_matrices = [None] * num_ensembles

        self.anchor_means = {}
        self.anchor_covs = {}
        self._fitted = False

    def to(self, device):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.omegas = [w.to(self.device) for w in self.omegas]
        self.bs_params = [b.to(self.device) for b in self.bs_params]
        self.sigma_mats = [s.to(self.device) for s in self.sigma_mats]
        for i in range(self.num_ensembles):
            for k in list(self.class_means[i].keys()):
                self.class_means[i][k] = self.class_means[i][k].to(self.device)
        return self

    def _compute_rff(self, X, ens_idx):
        scaling = torch.sqrt(torch.tensor(2.0 / self.D, dtype=torch.float32, device=X.device))
        return scaling * torch.cos(X @ self.omegas[ens_idx] + self.bs_params[ens_idx])

    def reset_parameters(self):
        pass

    def parameters_for_optimization(self):
        return []

    def register_anchor_stats(self, class_id, mean_x, cov_x):
        self.anchor_means[int(class_id)] = mean_x.detach().clone()
        self.anchor_covs[int(class_id)] = cov_x.detach().clone()

    def batch_update(self, X, y):
        """Update with ALL samples of a SINGLE class. Matches original KLDA API.
        Args:
            X: (n_samples, feature_dim) — all samples of class y
            y: int — class label
        """
        X = X.to(self.device)
        for i in range(self.num_ensembles):
            phi_X = self._compute_rff(X, i)  # (n, D)
            self.class_means[i][y] = torch.mean(phi_X, dim=0)
            centered = phi_X - self.class_means[i][y]
            self.sigma_mats[i] = self.sigma_mats[i].to(self.device)
            self.sigma_mats[i] += centered.t() @ centered

    def fit(self, class_ids):
        """Compute Σ⁻¹ and class_mean_matrix for all ensembles."""
        for i in range(self.num_ensembles):
            self.sigma_mats[i] = self.sigma_mats[i].to(self.device)
            self.sigma_invs[i] = torch.pinverse(
                self.sigma_mats[i] + 1e-4 * torch.eye(self.D, device=self.device))
            means = [self.class_means[i][c].to(self.device) for c in class_ids]
            self.class_mean_matrices[i] = torch.stack(means).to(self.device)

    def transform(self, features):
        return features

    def forward(self, features):
        return self.transform(features)

    def predict_logits(self, features):
        features = features.to(self.device)
        batch_size = features.shape[0]
        num_classes = self.class_mean_matrices[0].shape[0]
        total_probs = torch.zeros(batch_size, num_classes, device=self.device)
        for i in range(self.num_ensembles):
            phi_x = self._compute_rff(features, i)
            cm = self.class_mean_matrices[i]
            sinv = self.sigma_invs[i]
            diff = cm.unsqueeze(0) - phi_x.unsqueeze(1)
            logits = -torch.einsum('bcd,dd,bcd->bc', diff, sinv, diff)
            total_probs += torch.softmax(logits, dim=1)
        return total_probs / self.num_ensembles

    def transform_statistics(self, class_id):
        return self.anchor_means[class_id], self.anchor_covs[class_id]

    def separation_loss(self, class_ids):
        return torch.zeros((), device=self.device)
