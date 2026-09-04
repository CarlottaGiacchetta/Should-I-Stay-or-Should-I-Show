from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchmetrics.classification import BinaryAUROC, MulticlassAUROC


def _build_optimizer(optimizer: str, params, lr: float, weight_decay: float):
    optimizer = optimizer.lower()
    if optimizer == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if optimizer == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{optimizer}'. Choose 'adam' or 'adamw'.")


class ConvNeXtTinyBackbone(nn.Module):
    """ConvNeXt-Tiny feature extractor (768-dim pooled features) shared by all
    the image models."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        print(f"\n\n[ConvNeXtTinyBackbone] pretrained={pretrained}, weights={weights}")
        model = convnext_tiny(weights=weights)

        self.features = model.features
        self.avgpool = model.avgpool
        self.out_dim = model.classifier[2].in_features

        if freeze_backbone:
            print("Freezing ConvNeXt backbone")
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class LabelClassifier(pl.LightningModule):
    """
    Tabular label model f(x) = P(Y | X = x), used by the ClasswiseRisk
    estimator.

    Args:
        in_dim:       feature dimension of X.
        num_classes:  |Y|.
        hidden:       width of the first hidden layer (the second is hidden // 2).
        lr / weight_decay: optimization.
        dropout:      dropout after the first hidden layer (halved after the second).
        optimizer:    'adam' or 'adamw'.
    """

    def __init__(
            self,
            in_dim: int,
            num_classes: int,
            hidden: int = 128,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            dropout: float = 0.0,
            optimizer: str = "adam",
    ):
        super().__init__()
        self.save_hyperparameters()

        hidden2 = max(hidden // 2, 1)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden2, num_classes),
        )
        self.train_auc = MulticlassAUROC(num_classes=num_classes)
        self.val_auc = MulticlassAUROC(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the logits over Y. Shape: (B, |Y|)."""
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns P(Y=y'|X=x). Shape: (B, |Y|)."""
        return F.softmax(self.forward(x), dim=-1)

    def _step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch["x"], batch["y"]
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            acc = (logits.argmax(dim=-1) == y).float().mean()

        if stage == "train":
            auc = self.train_auc(probs, y)
        elif stage == "val":
            auc = self.val_auc(probs, y)
        else:
            auc = None

        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_epoch=True)
        if auc is not None:
            self.log(f"{stage}/auc", auc, prog_bar=True, on_epoch=True)

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return _build_optimizer(
            self.hparams.optimizer,
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class ImageLabelClassifier(pl.LightningModule):
    """Image counterpart of LabelClassifier: ConvNeXt-Tiny backbone plus a
    two-layer classification head."""

    def __init__(
        self,
        num_classes: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        head_hidden: int = 512,
        head_hidden2: int = 256,
        dropout: float = 0.3,
        label_smoothing: float = 0.05,
        optimizer: str = "adamw",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = ConvNeXtTinyBackbone(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        feat_dim = self.backbone.out_dim

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden2, num_classes),
        )

        self.train_auc = MulticlassAUROC(num_classes=num_classes)
        self.val_auc = MulticlassAUROC(num_classes=num_classes)

    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)

    def predict_proba(self, x):
        return F.softmax(self.forward(x), dim=-1)

    def _step(self, batch, stage: str):
        x, y = batch["x"], batch["y"]
        logits = self.forward(x)

        loss = F.cross_entropy(
            logits,
            y,
            label_smoothing=self.hparams.label_smoothing,
        )

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            acc = (logits.argmax(dim=-1) == y).float().mean()

        if stage == "train":
            auc = self.train_auc(probs, y)
        elif stage == "val":
            auc = self.val_auc(probs, y)
        else:
            auc = None

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        if auc is not None:
            self.log(f"{stage}_auc", auc, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return _build_optimizer(
            self.hparams.optimizer,
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class ErrorHeadModel(pl.LightningModule):
    """
    Tabular class-conditional error model for one regime d:
    g_d(x)_{y'} = P(A_d != y' | X = x, Y = y'), one binary head per class.

    Args:
        in_dim:       feature dimension of X.
        num_classes:  |Y|, i.e. the number of binary heads.
        regime:       the disclosure value d the model is trained for.
        hidden:       width of the first hidden layer (the second is hidden // 2).
        lr / weight_decay: optimization.
        dropout:      dropout after the first hidden layer (halved after the second).
        optimizer:    'adam' or 'adamw'.
        pos_weight:   positive-class weight for the BCE loss (errors are the
                      minority class); None disables it.
    """

    def __init__(
            self,
            in_dim: int,
            num_classes: int,
            regime: int,
            hidden: int = 128,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            dropout: float = 0.0,
            optimizer: str = "adam",
            pos_weight: float | None = None,
    ):
        super().__init__()
        assert regime in (0, 1), "regime must be 0 or 1"
        self.save_hyperparameters()
        hidden2 = max(hidden // 2, 1)
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.heads = nn.Linear(hidden2, num_classes)

        if pos_weight is not None:
            self.register_buffer("_pos_weight", torch.tensor(float(pos_weight)))
        else:
            self._pos_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the per-head logits. Shape: (B, |Y|)."""
        return self.heads(self.trunk(x))

    def predict_error_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns g_d(x)_{y'} = P(A_d != y' | X=x, Y=y'). Shape: (B, |Y|).
        The weighted BCE shifts the optimal logit by log(pos_weight), which is
        undone here before reading the output as a probability."""
        logits = self.forward(x)
        if self._pos_weight is not None:
            logits = logits - torch.log(self._pos_weight)
        return torch.sigmoid(logits)

    def _step(self, batch, stage: str) -> torch.Tensor:
        x, y, err = batch["x"], batch["y"], batch["err"].float()
        logits = self.forward(x)
        head_logit = logits.gather(1, y.unsqueeze(1)).squeeze(1)

        loss = F.binary_cross_entropy_with_logits(
            head_logit,
            err,
            pos_weight=self._pos_weight,
        )

        with torch.no_grad():
            pred_err = (head_logit > 0).float()
            acc = (pred_err == err).float().mean()

        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/err_head_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return _build_optimizer(
            self.hparams.optimizer,
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class ImageErrorHeadModel(pl.LightningModule):
    """Image counterpart of ErrorHeadModel."""

    def __init__(
        self,
        num_classes: int,
        regime: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        head_hidden: int = 512,
        head_hidden2: int = 256,
        dropout: float = 0.3,
        pos_weight: float | None = None,
        optimizer: str = "adamw",
    ):
        super().__init__()
        assert regime in (0, 1)
        self.save_hyperparameters()

        self.backbone = ConvNeXtTinyBackbone(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        feat_dim = self.backbone.out_dim

        self.trunk = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.heads = nn.Linear(head_hidden2, num_classes)

        if pos_weight is not None:
            self.register_buffer("_pos_weight", torch.tensor(float(pos_weight)))
        else:
            self._pos_weight = None

        self.train_auc = BinaryAUROC()
        self.val_auc = BinaryAUROC()

    def forward(self, x):
        z = self.backbone(x)
        z = self.trunk(z)
        return self.heads(z)

    def predict_error_proba(self, x):
        logits = self.forward(x)
        if self._pos_weight is not None:
            logits = logits - torch.log(self._pos_weight)
        return torch.sigmoid(logits)

    def _step(self, batch, stage: str):
        x, y, err = batch["x"], batch["y"], batch["err"].float()
        logits = self.forward(x)
        head_logit = logits.gather(1, y.unsqueeze(1)).squeeze(1)

        loss = F.binary_cross_entropy_with_logits(
            head_logit,
            err,
            pos_weight=self._pos_weight,
        )

        with torch.no_grad():
            prob = torch.sigmoid(head_logit)
            pred_err = (head_logit > 0).float()
            acc = (pred_err == err).float().mean()

        if stage == "train":
            auc = self.train_auc(prob, err.long())
        elif stage == "val":
            auc = self.val_auc(prob, err.long())
        else:
            auc = None

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_err_head_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        if auc is not None:
            self.log(f"{stage}_auc", auc, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return _build_optimizer(
            self.hparams.optimizer,
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class VoIEstimator(pl.LightningModule):
    """
    ClasswiseRisk estimator. Combines the label model f with the two
    class-conditional error heads g_0, g_1 into the conditional risks
        r_d(x) = sum_{y'} f(x)_{y'} * g_d(x)_{y'}
    and the value of information VoI(x) = r_0(x) - r_1(x).
    """

    def __init__(
            self,
            label_model: LabelClassifier,
            error_model_d0: ErrorHeadModel,
            error_model_d1: ErrorHeadModel,
    ):
        super().__init__()
        K = label_model.hparams.num_classes
        assert error_model_d0.hparams.num_classes == K
        assert error_model_d1.hparams.num_classes == K
        assert error_model_d0.hparams.regime == 0
        assert error_model_d1.hparams.regime == 1

        self.f = label_model
        self.g0 = error_model_d0
        self.g1 = error_model_d1

    @torch.no_grad()
    def conditional_risk(self, x: torch.Tensor, d: int) -> torch.Tensor:
        """r_d(x) = sum_{y'} f(x)_{y'} * g_d(x)_{y'}. Shape: (B,)."""
        assert d in (0, 1)
        f_probs = self.f.predict_proba(x)
        g = self.g0 if d == 0 else self.g1
        err_probs = g.predict_error_proba(x)
        return (f_probs * err_probs).sum(dim=-1)

    @torch.no_grad()
    def voi(self, x: torch.Tensor) -> torch.Tensor:
        """VoI(x) = r_0(x) - r_1(x). Shape: (B,)."""
        return self.conditional_risk(x, 0) - self.conditional_risk(x, 1)


class BinaryRegimeRiskModel(pl.LightningModule):
    """
    Tabular scalar risk model for one regime d: h_d(x) = P(A_d != Y | X = x),
    the building block of the DirectRisk T-learner.
    """

    def __init__(
            self,
            in_dim: int,
            regime: int,
            hidden: int = 128,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            dropout: float = 0.0,
            optimizer: str = "adam",
            pos_weight: float | None = None,
    ):
        super().__init__()
        assert regime in (0, 1), "regime must be 0 or 1"
        self.save_hyperparameters()
        hidden2 = max(hidden // 2, 1)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden2, 1),
        )

        if pos_weight is not None:
            self.register_buffer("_pos_weight", torch.tensor(float(pos_weight)))
        else:
            self._pos_weight = None
        self.train_auc = BinaryAUROC()
        self.val_auc = BinaryAUROC()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_error_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns h_d(x) = P(A_d != Y | X=x), undoing the log(pos_weight)
        shift introduced by the weighted BCE. Shape: (B,)."""
        logits = self.forward(x)
        if self._pos_weight is not None:
            logits = logits - torch.log(self._pos_weight)
        return torch.sigmoid(logits)

    def _step(self, batch, stage: str) -> torch.Tensor:
        x, err = batch["x"], batch["err"].float()
        logits = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            err,
            pos_weight=self._pos_weight,
        )
        probs = torch.sigmoid(logits)

        with torch.no_grad():
            pred_err = (logits > 0).float()
            acc = (pred_err == err).float().mean()

        if stage == "train":
            auc = self.train_auc(probs, err.long())
        elif stage == "val":
            auc = self.val_auc(probs, err.long())
        else:
            auc = None

        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/err_acc", acc, prog_bar=True, on_epoch=True)
        if auc is not None:
            self.log(f"{stage}/auc", auc, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return _build_optimizer(
            self.hparams.optimizer,
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class ImageBinaryRegimeRiskModel(pl.LightningModule):
    """Image counterpart of BinaryRegimeRiskModel."""

    def __init__(
        self,
        regime: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        head_hidden: int = 512,
        head_hidden2: int = 256,
        pos_weight: float | None = None,
        optimizer: str = "adamw",
    ):
        super().__init__()
        assert regime in (0, 1)
        self.save_hyperparameters()

        self.backbone = ConvNeXtTinyBackbone(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        feat_dim = self.backbone.out_dim

        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden2, 1),
        )

        if pos_weight is not None:
            self.register_buffer("_pos_weight", torch.tensor(float(pos_weight)))
        else:
            self._pos_weight = None

        self.train_auc = BinaryAUROC()
        self.val_auc = BinaryAUROC()

    def forward(self, x):
        z = self.backbone(x)
        return self.net(z).squeeze(-1)

    def predict_error_proba(self, x):
        logits = self.forward(x)
        if self._pos_weight is not None:
            logits = logits - torch.log(self._pos_weight)
        return torch.sigmoid(logits)

    def _step(self, batch, stage: str):
        x, err = batch["x"], batch["err"].float()
        logits = self.forward(x)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            err,
            pos_weight=self._pos_weight,
        )

        probs = torch.sigmoid(logits)

        with torch.no_grad():
            pred_err = (logits > 0).float()
            acc = (pred_err == err).float().mean()

        if stage == "train":
            auc = self.train_auc(probs, err.long())
        elif stage == "val":
            auc = self.val_auc(probs, err.long())
        else:
            auc = None

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_err_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        if auc is not None:
            self.log(f"{stage}_auc", auc, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return _build_optimizer(
            self.hparams.optimizer,
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class RegimeRiskTLearner(pl.LightningModule):
    """
    DirectRisk estimator. Pairs the two regime-risk models into
        r_d(x) = P(A_d != Y | X = x)
    and the value of information VoI(x) = r_0(x) - r_1(x).
    """

    def __init__(
            self,
            risk_model_d0: BinaryRegimeRiskModel,
            risk_model_d1: BinaryRegimeRiskModel,
    ):
        super().__init__()
        assert risk_model_d0.hparams.regime == 0
        assert risk_model_d1.hparams.regime == 1
        self.g0 = risk_model_d0
        self.g1 = risk_model_d1

    @torch.no_grad()
    def conditional_risk(self, x: torch.Tensor, d: int) -> torch.Tensor:
        """r_d(x) = P(A_d != Y | X = x). Shape: (B,)."""
        assert d in (0, 1)
        g = self.g0 if d == 0 else self.g1
        return g.predict_error_proba(x)

    @torch.no_grad()
    def voi(self, x: torch.Tensor) -> torch.Tensor:
        """VoI(x) = r_0(x) - r_1(x). Shape: (B,)."""
        return self.conditional_risk(x, 0) - self.conditional_risk(x, 1)


class SynthLabelClassifier(pl.LightningModule):
    """Label model f(x) = P(Y | X = x) for the synthetic scenario: plain ReLU
    MLP, no normalization and no dropout."""

    def __init__(
            self,
            in_dim: int,
            num_classes: int,
            hidden: int = 128,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the logits over Y. Shape: (B, |Y|)."""
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns P(Y=y'|X=x). Shape: (B, |Y|)."""
        return F.softmax(self.forward(x), dim=-1)

    def _step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch["x"], batch["y"]
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)

        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == y).float().mean()

        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class SynthErrorHeadModel(pl.LightningModule):
    """Class-conditional error model g_d for the synthetic scenario: GELU MLP
    with one binary head per class, no normalization and no dropout."""

    def __init__(
            self,
            in_dim: int,
            num_classes: int,
            regime: int,
            hidden: int = 128,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            pos_weight: float | None = None,
    ):
        super().__init__()
        assert regime in (0, 1), "regime must be 0 or 1"
        self.save_hyperparameters()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.heads = nn.Linear(hidden, num_classes)

        if pos_weight is not None:
            self.register_buffer("_pos_weight", torch.tensor(float(pos_weight)))
        else:
            self._pos_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the per-head logits. Shape: (B, |Y|)."""
        return self.heads(self.trunk(x))

    def predict_error_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns g_d(x)_{y'} = P(A_d != y' | X=x, Y=y'). Shape: (B, |Y|)."""
        logits = self.forward(x)
        if self._pos_weight is not None:
            logits = logits - torch.log(self._pos_weight)
        return torch.sigmoid(logits)

    def _step(self, batch, stage: str) -> torch.Tensor:
        x, y, err = batch["x"], batch["y"], batch["err"].float()
        logits = self.forward(x)
        head_logit = logits.gather(1, y.unsqueeze(1)).squeeze(1)

        loss = F.binary_cross_entropy_with_logits(
            head_logit,
            err,
            pos_weight=self._pos_weight,
        )

        with torch.no_grad():
            pred_err = (head_logit > 0).float()
            acc = (pred_err == err).float().mean()

        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/err_head_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class SynthBinaryRegimeRiskModel(pl.LightningModule):
    """Scalar risk model h_d(x) = P(A_d != Y | X = x) for the synthetic
    scenario: GELU MLP with dropout, no normalization."""

    def __init__(
            self,
            in_dim: int,
            regime: int,
            hidden: int = 128,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            dropout: float = 0.0,
            pos_weight: float | None = None,
    ):
        super().__init__()
        assert regime in (0, 1), "regime must be 0 or 1"
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        if pos_weight is not None:
            self.register_buffer("_pos_weight", torch.tensor(float(pos_weight)))
        else:
            self._pos_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_error_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns h_d(x) = P(A_d != Y | X=x). Shape: (B,)."""
        logits = self.forward(x)
        if self._pos_weight is not None:
            logits = logits - torch.log(self._pos_weight)
        return torch.sigmoid(logits)

    def _step(self, batch, stage: str) -> torch.Tensor:
        x, err = batch["x"], batch["err"].float()
        logits = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            err,
            pos_weight=self._pos_weight,
        )
        with torch.no_grad():
            pred_err = (logits > 0).float()
            acc = (pred_err == err).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/err_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
