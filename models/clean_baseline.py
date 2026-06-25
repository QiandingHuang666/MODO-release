import logging
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.dol import IdentitySingleDOLPlugin, SingleDOLPlugin
from utils.lda import SingleLDAPlugin
from utils.inc_net import SimpleClipNet, SimpleVitNet
from utils.toolkit import get_attribute, tensor2numpy


num_workers = 8


class CleanClipBaseLearner(BaseLearner):
    def __init__(self, args, network_type="clip"):
        super().__init__(args)
        self.args = args
        self.network_type = network_type
        self._network = SimpleClipNet(args, True) if network_type == "clip" else SimpleVitNet(args, True)

        self.batch_size = get_attribute(args, "batch_size", 48)
        self.init_lr = get_attribute(args, "init_lr", 0.01)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 1e-8)
        self.train_epoch = get_attribute(args, "train_epoch", get_attribute(args, "tuned_epoch", 10))

        self.use_dol = get_attribute(args, "use_dol", False)
        self.use_lda = get_attribute(args, "use_lda", False)
        self.lda_mode = get_attribute(args, "lda_mode", "full_dim")
        self.dol_epoch = get_attribute(args, "dol_epoch", self.train_epoch)
        self.dol_lr = get_attribute(args, "dol_lr", self.init_lr)
        self.dol_alpha = get_attribute(args, "dol_alpha", 1.0)
        self.dol_lambda_sep = get_attribute(args, "dol_lambda_sep", 1.0)
        self.dol_eps = get_attribute(args, "dol_eps", 1e-8)

        self._classifier_ready = False
        self._class_means_torch = None
        self._class_covs_torch = {}
        self._text_prototypes = None
        self._gda_cov_inv = None
        self._gda_log_prior = None

        feature_dim = self._network.feature_dim
        if self.use_lda:
            self.dol = SingleLDAPlugin(
                feature_dim=feature_dim,
                loss_mode=self.lda_mode,
                eps=self.dol_eps,
            )
        elif self.use_dol:
            self.dol = SingleDOLPlugin(
                feature_dim=feature_dim,
                alpha=self.dol_alpha,
                lambda_sep=self.dol_lambda_sep,
                eps=self.dol_eps,
            )
        else:
            self.dol = IdentitySingleDOLPlugin(feature_dim=feature_dim)

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        train_eval_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="test"
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )

        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers
        )
        self.train_eval_loader = DataLoader(
            train_eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        self._network.to(self._device)
        self.dol.to(self._device)
        self._prepare_stage_statistics()
        self._run_dol_stage()
        self._fit_classifier_stage()

    def _network_encode_image(self, inputs):
        if hasattr(self._network, "convnet"):
            return self._network.convnet.encode_image(inputs)
        return self._network.encode_image(inputs)

    def _network_encode_text(self, texts):
        if hasattr(self._network, "convnet"):
            return self._network.convnet.encode_text(texts)
        return self._network.encode_text(texts)

    def _extract_features_and_labels(self, loader):
        features, labels = [], []
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device)
                image_features = self._network_encode_image(inputs)
                features.append(image_features)
                labels.append(targets.to(self._device))
        return torch.cat(features, dim=0), torch.cat(labels, dim=0)

    def _prepare_stage_statistics(self):
        self.dol.reset_parameters()
        raw_features, labels = self._extract_features_and_labels(self.train_eval_loader)

        class_means = []
        class_covs = {}
        for class_idx in range(self._total_classes):
            cls_features = raw_features[labels == class_idx]
            if cls_features.shape[0] == 0:
                # Old class not in current loader — reuse existing anchor stats
                if class_idx in self.dol.anchor_means:
                    cls_mean = self.dol.anchor_means[class_idx]
                    cls_cov = self.dol.anchor_covs[class_idx]
                else:
                    continue
            else:
                cls_mean = cls_features.mean(dim=0)
                if cls_features.shape[0] > 1:
                    centered = cls_features - cls_mean
                    cls_cov = centered.t().mm(centered) / (cls_features.shape[0] - 1)
                else:
                    cls_cov = torch.eye(raw_features.shape[1], device=self._device, dtype=raw_features.dtype)
            class_means.append(cls_mean)
            class_covs[class_idx] = cls_cov
            self.dol.register_anchor_stats(class_idx, cls_mean, cls_cov)

        self._class_means_torch = torch.stack(class_means, dim=0)
        self._class_covs_torch = class_covs
        self._raw_train_features = raw_features
        self._raw_train_labels = labels
        self._text_prototypes = self._build_text_prototypes()
        self._classifier_ready = False

    def _build_text_prototypes(self):
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes]
        text_features = []
        with torch.no_grad():
            for label in total_labels:
                texts = [t.format(label) for t in templates]
                texts = self._network.tokenizer(texts).to(self._device)
                class_embeddings = self._network_encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
        return torch.stack(text_features, dim=0)

    def _run_dol_stage(self):
        if not self.use_dol and not self.use_lda:
            return

        # LDA: closed-form solve, no gradient descent
        if self.use_lda:
            logging.info("LDA: solving closed-form ({})".format(self.lda_mode))
            self.dol._solve_lda()
            if self.dol._W is not None:
                logging.info(
                    "LDA: W shape = {}".format(self.dol._W.shape)
                )
            return

        optimizer = optim.Adam(self.dol.parameters_for_optimization(), lr=self.dol_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.dol_epoch, eta_min=self.min_lr
        )
        class_ids = list(range(self._total_classes))
        prog_bar = tqdm(range(self.dol_epoch))
        self.dol.train()
        for epoch in prog_bar:
            loss = self.dol.separation_loss(class_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            prog_bar.set_description(
                "Task {}, DOL Epoch {}/{} => Loss {:.3f}".format(
                    self._cur_task, epoch + 1, self.dol_epoch, loss.item()
                )
            )
        self.dol.eval()

    def _transform_features(self, features):
        return self.dol.transform(features) if (self.use_dol or self.use_lda) else features

    def _transform_class_means(self):
        return self._transform_features(self._class_means_torch)

    def _fit_classifier_stage(self):
        raise NotImplementedError

    def _predict_logits(self, transformed_features):
        raise NotImplementedError

    def _eval_cnn(self, loader):
        self._network.eval()
        self.dol.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                raw_features = self._network_encode_image(inputs)
                transformed = self._transform_features(raw_features)
                outputs = self._predict_logits(transformed)
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _compute_accuracy(self, model, loader):
        self._network.eval()
        self.dol.eval()
        correct, total = 0, 0
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                raw_features = self._network_encode_image(inputs)
                transformed = self._transform_features(raw_features)
                outputs = self._predict_logits(transformed)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


class PrototypeClassifierLearner(CleanClipBaseLearner):
    def _fit_classifier_stage(self):
        self.prototype = self._transform_class_means()
        self._classifier_ready = True

    def _predict_logits(self, transformed_features):
        image_features = transformed_features / transformed_features.norm(dim=-1, keepdim=True)
        proto = self.prototype / self.prototype.norm(dim=-1, keepdim=True)
        return image_features @ proto.t()


class TextPrototypeClassifierLearner(CleanClipBaseLearner):
    def _fit_classifier_stage(self):
        optimizer = optim.SGD(
            self._build_alignment_parameters(),
            lr=self.init_lr,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.train_epoch, eta_min=self.min_lr
        )
        text_proto = self._text_prototypes / self._text_prototypes.norm(dim=-1, keepdim=True)
        prog_bar = tqdm(range(self.train_epoch))
        for epoch in prog_bar:
            total_loss = 0.0
            correct, total = 0, 0
            for _, inputs, targets in self.train_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                raw_features = self._network_encode_image(inputs)
                transformed = self._transform_features(raw_features)
                aligned_features = self._align_features(transformed)
                aligned_features = aligned_features / aligned_features.norm(dim=-1, keepdim=True)
                logits = aligned_features @ text_proto.t()
                loss = nn.functional.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += logits.argmax(dim=1).eq(targets).sum().item()
                total += targets.shape[0]
            scheduler.step()
            prog_bar.set_description(
                "Task {}, Epoch {}/{} => Loss {:.3f}, Train_acc {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.train_epoch,
                    total_loss / max(len(self.train_loader), 1),
                    100.0 * correct / max(total, 1),
                )
            )
        self.prototype = text_proto
        self._classifier_ready = True

    def _build_alignment_parameters(self):
        feature_dim = self._network.feature_dim
        self.image_align = nn.Linear(feature_dim, feature_dim, bias=False).to(self._device)
        with torch.no_grad():
            self.image_align.weight.copy_(torch.eye(feature_dim, device=self._device))
        return self.image_align.parameters()

    def _align_features(self, transformed_features):
        return self.image_align(transformed_features)

    def _predict_logits(self, transformed_features):
        image_features = self._align_features(transformed_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_proto = self.prototype / self.prototype.norm(dim=-1, keepdim=True)
        return image_features @ text_proto.t()


class GDAClassifierLearner(CleanClipBaseLearner):
    def _fit_classifier_stage(self):
        d = self._network.feature_dim
        # Use transformed class means for ALL classes (same as PrototypeClassifierLearner).
        # _class_means_torch contains means for every class 0..total_classes-1,
        # including old classes via DOL anchor stats. This avoids NaN prototypes
        # that occurred when old classes had no data in _raw_train_features.
        self.prototype = self._transform_class_means()

        # Pooled shared covariance in DOL-transformed space.
        # Use stored per-class anchor covariances (_class_covs_torch has entries
        # for all classes), transform them through the current DOL layer.
        covs_z = []
        for class_idx in range(self._total_classes):
            cov_x = self._class_covs_torch[class_idx].to(
                device=self._device, dtype=self.prototype.dtype
            )
            if hasattr(self.dol, "layer"):
                # cov_z = W @ cov_x @ W^T  (transform covariance through DOL)
                cov_z = self.dol.layer.weight @ cov_x @ self.dol.layer.weight.t()
            else:
                cov_z = cov_x
            covs_z.append(cov_z)

        cov_pooled = torch.stack(covs_z, dim=0).mean(dim=0)  # (D, D)
        cov = cov_pooled + 1e-4 * torch.eye(d, device=self._device)

        # Regularized precision matrix with shrinkage toward identity
        self._gda_cov_inv = d * torch.linalg.pinv(
            self._total_classes * cov + cov.trace() * torch.eye(d, device=self._device)
        )

        prior = torch.ones(self.prototype.shape[0], device=self._device) / self.prototype.shape[0]
        self._gda_log_prior = prior.log()
        self._classifier_ready = True

    def _predict_logits(self, transformed_features):
        logits = torch.einsum("bd,dc->bc", transformed_features, self._gda_cov_inv @ self.prototype.t())
        bias = self._gda_log_prior - 0.5 * torch.einsum(
            "nd,dc,nc->n", self.prototype, self._gda_cov_inv, self.prototype
        )
        return logits + bias.unsqueeze(0)


class ParametricClassifierLearner(CleanClipBaseLearner):
    def __init__(self, args, head_type="fc"):
        super().__init__(args, network_type="clip")
        self.head_type = head_type
        self.classifier = None

    def _build_classifier(self):
        input_dim = self._network.feature_dim
        if self.head_type == "fc":
            return nn.Linear(input_dim, self._total_classes).to(self._device)
        hidden_dim = get_attribute(self.args, "mlp_hidden_dim", input_dim)
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self._total_classes),
        ).to(self._device)

    def _fit_classifier_stage(self):
        self.classifier = self._build_classifier()
        optimizer = optim.SGD(self.classifier.parameters(), lr=self.init_lr, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.train_epoch, eta_min=self.min_lr
        )
        prog_bar = tqdm(range(self.train_epoch))
        self.classifier.train()
        for epoch in prog_bar:
            total_loss = 0.0
            correct, total = 0, 0
            for _, inputs, targets in self.train_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                with torch.no_grad():
                    raw_features = self._network_encode_image(inputs)
                    features = self._transform_features(raw_features)
                logits = self.classifier(features)
                loss = nn.functional.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += logits.argmax(dim=1).eq(targets).sum().item()
                total += targets.shape[0]
            scheduler.step()
            prog_bar.set_description(
                "Task {}, Epoch {}/{} => Loss {:.3f}, Train_acc {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.train_epoch,
                    total_loss / max(len(self.train_loader), 1),
                    100.0 * correct / max(total, 1),
                )
            )
        self.classifier.eval()
        self._classifier_ready = True

    def _predict_logits(self, transformed_features):
        return self.classifier(transformed_features)
