from __future__ import annotations

from typing import Sequence

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from src.datasets import TabularInteractionDataset


def standardize_from_train(
        x: torch.Tensor,
        train_idx: Sequence[int],
        eps: float = 1e-6,
) -> torch.Tensor:
    """Standardize the features using the statistics of the train split only."""
    idx = torch.tensor(list(train_idx), dtype=torch.long, device=x.device)
    x_train = x[idx]
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    return (x - mean) / std


class RegimeRiskDataModule(pl.LightningDataModule):
    """
    Batches (x, err_d) for the binary regime-risk T-learner (DirectRisk).

    For regime d the target is 1{A_d != Y}. Unlike ErrorDataModule, this does
    not condition on Y through separate heads: it learns P(A_d != Y | X)
    directly.
    """

    def __init__(
            self,
            dataset: TabularInteractionDataset,
            train_idx: Sequence[int],
            val_idx: Sequence[int],
            regime: int,
            batch_size: int = 128,
            num_workers: int = 0,
    ):
        super().__init__()
        assert regime in (0, 1)
        self.dataset = dataset
        self.regime = regime
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_idx = list(train_idx)
        self.val_idx = list(val_idx)

    def _collate(self, batch):
        err_key = "err0" if self.regime == 0 else "err1"
        return {
            "x": torch.stack([b["x"] for b in batch]).float(),
            "err": torch.stack([b[err_key] for b in batch]).float(),
        }

    def train_dataloader(self):
        return DataLoader(
            Subset(self.dataset, self.train_idx),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate,
        )

    def val_dataloader(self):
        return DataLoader(
            Subset(self.dataset, self.val_idx),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate,
        )

    def compute_pos_weight(
            self,
            eps: float = 1e-3,
            cap: float = 10.0,
    ) -> float:
        """BCE pos_weight = n_neg / n_pos on the train split of this regime,
        clamped to [1/cap, cap]."""
        A = self.dataset.A0 if self.regime == 0 else self.dataset.A1
        Y = self.dataset.Y
        idx = torch.tensor(self.train_idx, dtype=torch.long)
        a = A[idx]
        y = Y[idx]
        n_pos = (a != y).sum().item()
        n_neg = (a == y).sum().item()
        n_total = n_pos + n_neg
        p_err = max(min(n_pos / n_total, 1.0 - eps), eps)
        pw = (1.0 - p_err) / p_err
        return float(min(max(pw, 1.0 / cap), cap))
