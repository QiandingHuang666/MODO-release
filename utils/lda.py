"""
LDA (Linear Discriminant Analysis) transform plugin.

Drop-in replacement for SingleDOLPlugin with identical interface.
Two modes:
  - "full_dim": d→d, take all eigenvectors of S_W^{-1}S_B, pad with
                orthogonal complement to full dimension. Fair comparison
                with DOL (both are d→d transforms).
  - "reduced":  d→(C-1), standard LDA projection onto the C-1 dimensional
                discriminant subspace.

LDA is solved in closed form each task (no gradient descent).
"""
import torch
from torch import nn


class SingleLDAPlugin(nn.Module):
    def __init__(self, feature_dim, alpha=1.0, lambda_sep=1.0, eps=1e-8,
                 loss_mode="full_dim", random_scale=0.1, **kwargs):
        """
        loss_mode:
            "full_dim"  - d→d LDA: eigenvectors of S_W^{-1}S_B padded to d
            "reduced"   - d→(C-1) LDA: standard discriminant subspace
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.lambda_sep = lambda_sep
        self.eps = eps
        self.loss_mode = loss_mode
        self.random_scale = random_scale

        # W is stored as a d×d matrix (full_dim) or d×k (reduced).
        # Initialized as identity (full_dim) for compatibility.
        self.register_buffer(
            "weight", torch.eye(feature_dim)
        )
        self.register_buffer(
            "bias", torch.zeros(feature_dim)
        )

        self.anchor_means = {}
        self.anchor_covs = {}
        self._W = None  # actual projection matrix, set by _solve_lda

    def reset_parameters(self):
        # W will be recomputed in _solve_lda, nothing to reset here
        pass

    def _solve_lda(self):
        """Solve LDA in closed form from cached anchor stats."""
        class_ids = sorted(self.anchor_means.keys())
        if len(class_ids) < 2:
            self._W = torch.eye(
                self.feature_dim,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
            return

        d = self.feature_dim
        device = self.weight.device
        dtype = self.weight.dtype

        # Gather means
        means = torch.stack(
            [self.anchor_means[c].to(device, dtype) for c in class_ids], dim=0
        )  # (C, d)
        global_mean = means.mean(dim=0)  # (d,)

        C = len(class_ids)

        # S_B: between-class scatter (C, d, d) summed
        diff_b = means - global_mean.unsqueeze(0)  # (C, d)
        S_B = diff_b.t().mm(diff_b)  # (d, d)

        # S_W: within-class scatter, pooled from per-class covariances
        S_W = torch.zeros(d, d, device=device, dtype=dtype)
        for c in class_ids:
            S_W += self.anchor_covs[c].to(device, dtype)

        # Regularize S_W for numerical stability
        S_W += self.eps * torch.eye(d, device=device, dtype=dtype)

        # Solve generalized eigenvalue problem: S_W^{-1} S_B
        # Use eigh on the symmetric pair for stability
        try:
            # Method: solve S_W^{-1} S_B via Cholesky
            L = torch.linalg.cholesky(S_W)
            # S_W^{-1} S_B = L^{-T} (L^{-1} S_B L^{-T}) L^T  -- transform
            L_inv_SB = torch.linalg.solve_triangular(L, S_B, upper=False)
            M = L_inv_SB.t().mm(L_inv_SB)  # symmetric (d, d)
            eigvals, eigvecs = torch.linalg.eigh(M)
            # eigvecs are in the transformed space; map back
            W_raw = torch.linalg.solve_triangular(
                L.t(), eigvecs, upper=True
            )  # (d, d), columns are eigenvectors of S_W^{-1} S_B
        except Exception:
            # Fallback: direct solve
            S_W_inv = torch.linalg.pinv(S_W)
            mat = S_W_inv.mm(S_B)
            # Symmetrize for eigh
            mat_sym = (mat + mat.t()) / 2.0
            eigvals, eigvecs = torch.linalg.eigh(mat_sym)
            W_raw = eigvecs

        # Sort by descending eigenvalue
        idx = torch.argsort(eigvals, descending=True)
        W_raw = W_raw[:, idx]  # (d, d)

        # Normalize columns to unit norm (standard LDA convention)
        norms = W_raw.norm(dim=0, keepdim=True).clamp(min=self.eps)
        W_raw = W_raw / norms

        if self.loss_mode == "reduced":
            # d → (C-1)
            k = min(C - 1, d)
            self._W = W_raw[:, :k]  # (d, k)
        else:
            # full_dim: d → d, already full
            self._W = W_raw  # (d, d)

    def forward(self, features):
        return self.transform(features)

    def transform(self, features):
        if self._W is None:
            self._solve_lda()
        return features.mm(self._W)  # (n, d) @ (d, k) → (n, k)

    def parameters_for_optimization(self):
        return []  # closed-form, no gradient descent

    def register_anchor_stats(self, class_id, mean_x, cov_x):
        self.anchor_means[int(class_id)] = mean_x.detach().clone()
        self.anchor_covs[int(class_id)] = cov_x.detach().clone()
        # Invalidate cached W so it gets recomputed next transform call
        self._W = None

    def transform_statistics(self, class_id):
        if self._W is None:
            self._solve_lda()
        mean_x = self.anchor_means[class_id].to(
            self.weight.device, dtype=self.weight.dtype
        )
        cov_x = self.anchor_covs[class_id].to(
            self.weight.device, dtype=self.weight.dtype
        )
        mean_z = self._W.t().mm(mean_x.unsqueeze(1)).squeeze(1)
        cov_z = self._W.t().mm(cov_x).mm(self._W)
        return mean_z, cov_z

    def separation_loss(self, class_ids):
        # No training loss for LDA (closed-form)
        return torch.zeros(
            (), device=self.weight.device, dtype=self.weight.dtype
        )


class IdentitySingleLDAPlugin(nn.Module):
    """No-op fallback, mirrors IdentitySingleDOLPlugin interface."""
    def __init__(self, feature_dim, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.anchor_means = {}
        self.anchor_covs = {}

    def reset_parameters(self):
        pass

    def forward(self, features):
        return features

    def transform(self, features):
        return features

    def parameters_for_optimization(self):
        return []

    def register_anchor_stats(self, class_id, mean_x, cov_x):
        self.anchor_means[int(class_id)] = mean_x.detach().clone()
        self.anchor_covs[int(class_id)] = cov_x.detach().clone()

    def transform_statistics(self, class_id):
        return self.anchor_means[class_id], self.anchor_covs[class_id]

    def separation_loss(self, class_ids):
        return torch.zeros(())
