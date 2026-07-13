"""
KLDA Learner: CLIP features → Random Fourier Features → Kernel LDA.
Vectorized per-class covariance using scatter/gather for D=5000 speed.
"""
import torch
import logging
from torch.utils.data import DataLoader
from models.clean_baseline import CleanClipBaseLearner
from utils.klda import KLDAPlugin
from utils.toolkit import get_attribute


class KLDALearner(CleanClipBaseLearner):
    def __init__(self, args, network_type="clip"):
        super().__init__(args, network_type=network_type)
        self.use_dol = True
        self.use_lda = False

        D = get_attribute(args, "klda_D", 5000)
        sigma = get_attribute(args, "klda_sigma", 1e-4)
        num_ensembles = get_attribute(args, "klda_ensembles", 5)
        seed = args.get("seed", [1993])
        seed = seed[0] if isinstance(seed, list) else seed

        self._klda_D = D
        self._klda_sigma = sigma
        self._klda_ensembles = num_ensembles
        self._klda_seed = seed
        self._klda_plugin = None
        self._classifier_ready = False
        self._batch_size = get_attribute(args, "batch_size", 128)

    def _fit_classifier_stage(self):
        device = self._device
        network = self._network

        all_indices = list(range(self._total_classes))
        dataset = self.data_manager.get_dataset(all_indices, source="train", mode="test")
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=False, num_workers=4)

        feats_all, labels_all = [], []
        network.eval()
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(device)
                features = network.encode_image(inputs) if hasattr(network, 'encode_image') else network._network_encode_image(inputs)
                feats_all.append(features.cpu())
                labels_all.append(targets)
        feats_all = torch.cat(feats_all, dim=0).to(device)
        labels_all = torch.cat(labels_all, dim=0).to(device)
        N, d = feats_all.shape
        C = len(all_indices)

        self._klda_plugin = KLDAPlugin(
            feature_dim=network.feature_dim,
            D=self._klda_D,
            sigma=self._klda_sigma,
            num_ensembles=self._klda_ensembles,
            seed=self._klda_seed,
        ).to(device)

        for i in range(self._klda_ensembles):
            phi = self._klda_plugin._compute_rff(feats_all, i)  # (N, D)
            D_rff = phi.shape[1]

            # Compute class means via scatter_mean
            means = torch.zeros(C, D_rff, device=device)
            counts = torch.zeros(C, device=device)
            means.scatter_add_(0, labels_all.unsqueeze(-1).expand(-1, D_rff), phi)
            counts.scatter_add_(0, labels_all, torch.ones(N, device=device))
            mask = counts > 0
            means[mask] /= counts[mask].unsqueeze(-1)

            # Center features by class mean
            centered = phi - means[labels_all]  # (N, D)

            # Within-class scatter = X_centeredᵀ @ X_centered
            sigma_mat = centered.T @ centered  # (D, D)

            self._klda_plugin.sigma_mats[i] = sigma_mat
            for c in all_indices:
                if counts[c] > 0:
                    self._klda_plugin.class_means[i][c] = means[c]

        self._klda_plugin.fit(all_indices)
        self._classifier_ready = True
        logging.info("KLDA fitted: D={}, ensembles={}, classes={}".format(
            self._klda_D, self._klda_ensembles, self._total_classes))

    def _run_dol_stage(self):
        return

    def _predict_logits(self, features):
        return self._klda_plugin.predict_logits(features)

    def _transform_features(self, features):
        return features
