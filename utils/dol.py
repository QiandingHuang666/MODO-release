import torch
from torch import nn


class AffineDOL(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.weight = nn.Parameter(torch.eye(feature_dim))

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.copy_(torch.eye(self.feature_dim, device=self.weight.device, dtype=self.weight.dtype))

    def forward(self, x):
        return torch.matmul(x, self.weight.t())


class SingleDOLPlugin(nn.Module):
    def __init__(self, feature_dim, alpha=1.0, lambda_sep=1.0, eps=1e-8):
        super().__init__()
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.lambda_sep = lambda_sep
        self.eps = eps
        self.layer = AffineDOL(feature_dim)
        self.anchor_means = {}
        self.anchor_covs = {}

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, features):
        return self.layer(features)

    def transform(self, features):
        return self.layer(features)

    def parameters_for_optimization(self):
        return self.layer.parameters()

    def register_anchor_stats(self, class_id, mean_x, cov_x):
        self.anchor_means[int(class_id)] = mean_x.detach().clone()
        self.anchor_covs[int(class_id)] = cov_x.detach().clone()

    def transform_statistics(self, class_id):
        mean_x = self.anchor_means[class_id].to(
            self.layer.weight.device, dtype=self.layer.weight.dtype
        )
        cov_x = self.anchor_covs[class_id].to(
            self.layer.weight.device, dtype=self.layer.weight.dtype
        )
        mean_z = self.layer(mean_x.unsqueeze(0)).squeeze(0)
        cov_z = self.layer.weight @ cov_x @ self.layer.weight.t()
        return mean_z, cov_z

    def separation_loss(self, class_ids):
        if len(class_ids) < 2:
            return torch.zeros(
                (), device=self.layer.weight.device, dtype=self.layer.weight.dtype
            )

        means = {}
        covs = {}
        for class_id in class_ids:
            if class_id not in self.anchor_means:
                continue
            mean_z, cov_z = self.transform_statistics(class_id)
            means[class_id] = mean_z
            covs[class_id] = cov_z

        valid_ids = sorted(means.keys())
        if len(valid_ids) < 2:
            return torch.zeros(
                (), device=self.layer.weight.device, dtype=self.layer.weight.dtype
            )

        losses = []
        for idx, class_i in enumerate(valid_ids):
            for class_j in valid_ids[idx + 1:]:
                delta = means[class_i] - means[class_j]
                distance = torch.norm(delta, p=2)
                unit = delta / (distance + self.eps)
                radius_i = torch.sqrt(torch.clamp(unit @ covs[class_i] @ unit, min=0.0))
                radius_j = torch.sqrt(torch.clamp(unit @ covs[class_j] @ unit, min=0.0))
                ratio = distance.pow(2) / (
                    (self.alpha * (radius_i + radius_j)).pow(2) + self.eps
                )
                losses.append(torch.exp(-ratio))

        return self.lambda_sep * torch.stack(losses).mean()


class IdentitySingleDOLPlugin(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.anchor_means = {}
        self.anchor_covs = {}

    def reset_parameters(self):
        return None

    def forward(self, features):
        return features

    def transform(self, features):
        return features

    def parameters_for_optimization(self):
        return []

    def register_anchor_stats(self, class_id, mean_x, cov_x):
        self.anchor_means[int(class_id)] = mean_x.detach().clone()
        self.anchor_covs[int(class_id)] = cov_x.detach().clone()

    def separation_loss(self, class_ids):
        if class_ids and class_ids[0] in self.anchor_means:
            ref = self.anchor_means[class_ids[0]]
            return torch.zeros((), device=ref.device, dtype=ref.dtype)
        return torch.zeros(())
