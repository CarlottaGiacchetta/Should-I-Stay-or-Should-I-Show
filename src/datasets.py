from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
import torch
import pytorch_lightning as pl
import pandas as pd


def generate_synthetic_data(
    n_samples=1000,
    n_features=5,
    n_hidden=3,
    n_redundant=0,
    n_informative=5,
    n_classes=2,
    random_state=42,
    splits=(0.7, 0.1, 0.2),
    class_sep=1,
    return_torch: bool = False,
):
    """Build the synthetic scenario: features X, ground truth y and the
    decisions of two simulated humans. human_0 only observes the first
    (n_features - n_hidden) features, human_1 observes all of them, so a_1 is
    the decision taken with the extra information disclosure provides.
    X is returned restricted to the features human_0 can observe.
    """
    assert sum(splits) == 1, "Splits must sum to 1"
    assert n_informative + n_redundant <= n_features, (
        "Number of informative and redundant features must be less than or equal to total features"
    )
    assert n_hidden < n_features, "Number of hidden features must be less than total features"

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=n_redundant,
        n_informative=n_informative,
        n_classes=n_classes,
        random_state=random_state,
        class_sep=class_sep,
    )

    human_0 = LogisticRegression(
        C=1.0,
        solver="saga",
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )
    human_1 = LogisticRegression(
        C=1.0,
        solver="saga",
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )

    atts_0 = [el for el in range(n_features - n_hidden)]
    atts_1 = [el for el in range(n_features)]

    human_0.fit(X[:, atts_0], y)
    human_1.fit(X[:, atts_1], y)

    a_0 = human_0.predict(X[:, atts_0])
    a_1 = human_1.predict(X)

    print(f"Human 0 accuracy: {human_0.score(X[:, atts_0], y)}")
    print(f"Human 1 accuracy: {human_1.score(X[:, atts_1], y)}")

    if return_torch:
        return (
            torch.from_numpy(X[:, atts_0]).float(),
            torch.from_numpy(y).long(),
            torch.from_numpy(a_0).long(),
            torch.from_numpy(a_1).long(),
        )
    return X[:, atts_0], y, a_0, a_1


def email_sampled(
        emb_path: str = "Datasets/email/embeddings_final.csv",
        return_torch: bool = False,
        val_frac: float | None = 0.1,
        random_state: int = 42,
        splits=(0.7, 0.15, 0.15),
):
    """
    Load the email embeddings CSV and build the train/cal/test splits at the
    StimID level, so that augmented variants of the same email never end up in
    two different splits. Everything is derived from `random_state`, so the
    splits are reproducible from the seed alone.

    For every row the decisions a_0 (no help) and a_1 (with help) are sampled
    from the vote counts of the human annotators; rows without annotations are
    dropped. Calibration and test only keep non-augmented rows.

    val_frac carves an extra validation split out of train (4-way split
    train/val/cal/test) for hyperparameter tuning; None disables it and
    `tr_idx` is then the whole train split.

    Returns X, y, a_0, a_1, train_idx, cal_idx, test_idx, tr_idx, val_idx,
    stim_ids.
    """
    if val_frac is not None and not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be None or in (0, 1), got {val_frac}")
    if len(splits) != 3 or not np.isclose(sum(splits), 1.0):
        raise ValueError(f"splits must be a 3-tuple summing to 1. Got: {splits}")

    df = pd.read_csv(emb_path)

    required_cols = [
        "StimID", "augmentation", "GroundTruth",
        "human_0_fraudulent", "human_0_legitimate",
        "human_1_fraudulent", "human_1_legitimate",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {emb_path}: {missing_cols}\nFound: {df.columns.tolist()}")

    emb_cols = [c for c in df.columns if c.startswith("v51_emb_")]
    if not emb_cols:
        raise ValueError("No v51_emb_* column found.")

    X = df[emb_cols].to_numpy(dtype=np.float32)
    y = np.where(df["GroundTruth"].astype(str) == "fraudulent", 1, 0)

    rng = np.random.default_rng(random_state)

    def _sample_decisions(p_fraud_col: str, p_legit_col: str) -> np.ndarray:
        """Sample one decision per row from the annotators' vote split.
        Returns -1 where no vote is available."""
        p_f   = df[p_fraud_col].fillna(0.0).to_numpy(dtype=np.float64)
        p_l   = df[p_legit_col].fillna(0.0).to_numpy(dtype=np.float64)
        total = p_f + p_l
        valid = total > 0
        decisions = np.full(len(df), -1, dtype=np.int64)
        for i in np.where(valid)[0]:
            decisions[i] = int(rng.random() < p_f[i] / total[i])
        return decisions

    a_0 = _sample_decisions("human_0_fraudulent", "human_0_legitimate")
    a_1 = _sample_decisions("human_1_fraudulent", "human_1_legitimate")

    print(f"[Sampling] human_0 -> fraud:{(a_0==1).sum()} legit:{(a_0==0).sum()} missing:{(a_0==-1).sum()}")
    print(f"[Sampling] human_1 -> fraud:{(a_1==1).sum()} legit:{(a_1==0).sum()} missing:{(a_1==-1).sum()}")

    valid_mask = pd.Series((a_0 >= 0) & (a_1 >= 0), index=df.index)
    n_dropped  = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"[Missing] Dropped {n_dropped} rows without annotations.")

    origin_mask = (df["augmentation"] == "original") & valid_mask
    origin_df   = df[origin_mask].copy()
    if len(origin_df) == 0:
        raise ValueError("No valid original row left after dropping the missing annotations.")

    origin_stimids = origin_df["StimID"].dropna().unique()
    n_stimids = len(origin_stimids)
    if n_stimids < 3:
        raise ValueError(f"Too few original StimIDs: {n_stimids}")

    rng_split     = np.random.default_rng(random_state + 1)
    perm          = rng_split.permutation(n_stimids)
    stimids_perm  = origin_stimids[perm]

    n_train = max(1, int(splits[0] * n_stimids))
    n_cal   = max(1, int(splits[1] * n_stimids))
    n_test  = n_stimids - n_train - n_cal
    if n_test < 1:
        n_test = 1
        if n_cal > 1: n_cal -= 1
        else:         n_train -= 1

    train_stimids = set(stimids_perm[:n_train])
    cal_stimids   = set(stimids_perm[n_train:n_train + n_cal])
    test_stimids  = set(stimids_perm[n_train + n_cal:])

    if train_stimids & cal_stimids or train_stimids & test_stimids or cal_stimids & test_stimids:
        raise RuntimeError("StimID leakage across the global splits.")

    train_idx = df.index[df["StimID"].isin(train_stimids) & valid_mask].to_numpy()
    cal_idx   = df.index[df["StimID"].isin(cal_stimids)   & origin_mask].to_numpy()
    test_idx  = df.index[df["StimID"].isin(test_stimids)  & origin_mask].to_numpy()

    if val_frac is not None:
        train_orig_stimids = (
            df[df["StimID"].isin(train_stimids) & origin_mask]["StimID"].dropna().unique()
        )
        n_train_stimids = len(train_orig_stimids)
        if n_train_stimids < 2:
            raise ValueError(f"Too few original StimIDs in train: {n_train_stimids}")

        perm_inner    = rng_split.permutation(n_train_stimids)
        n_val_stimids = min(max(1, int(round(val_frac * n_train_stimids))), n_train_stimids - 1)

        val_stimids = set(train_orig_stimids[perm_inner[:n_val_stimids]])
        tr_stimids  = set(train_orig_stimids[perm_inner[n_val_stimids:]])

        val_idx = df.index[df["StimID"].isin(val_stimids) & origin_mask].to_numpy()
        tr_idx  = df.index[df["StimID"].isin(tr_stimids)  & valid_mask].to_numpy()
    else:
        val_idx = np.array([], dtype=train_idx.dtype)
        tr_idx = train_idx

    assert len(np.intersect1d(train_idx, cal_idx))  == 0, "train/cal leakage"
    assert len(np.intersect1d(train_idx, test_idx)) == 0, "train/test leakage"
    assert len(np.intersect1d(cal_idx,   test_idx)) == 0, "cal/test leakage"
    assert len(np.intersect1d(tr_idx,    val_idx))  == 0, "tr/val leakage"
    if val_frac is not None:
        assert (df.loc[val_idx,  "augmentation"] == "original").all(), "val contains augmented rows"
        assert not (set(df.loc[tr_idx, "StimID"]) & set(df.loc[val_idx, "StimID"])), "tr/val StimID leakage"
    assert (df.loc[cal_idx,  "augmentation"] == "original").all(), "cal contains augmented rows"
    assert (df.loc[test_idx, "augmentation"] == "original").all(), "test contains augmented rows"
    assert (a_0[train_idx] >= 0).all(), "missing a_0 in train_idx"
    assert (a_1[train_idx] >= 0).all(), "missing a_1 in train_idx"

    print(f"\nHUMAN BASELINE ACCURACY")
    for name, idx in [("train", train_idx), ("cal", cal_idx), ("test", test_idx),
                      ("all", np.concatenate([train_idx, cal_idx, test_idx]))]:
        if len(idx) == 0:
            continue
        acc_0 = (a_0[idx] == y[idx]).mean()
        acc_1 = (a_1[idx] == y[idx]).mean()
        print(f"  {name:5s}: human_0 = {acc_0:.4f} | human_1 = {acc_1:.4f} | delta = {acc_1 - acc_0:+.4f}")

    print(f"\nLoaded {len(df)} rows ({n_dropped} dropped) | seed: {random_state}")
    print(f"Total StimID: {df['StimID'].nunique()} | valid original: {origin_df['StimID'].nunique()}")
    print(f"\nGLOBAL SPLITS")
    print(f"  Train  StimID: {len(train_stimids):4d} | rows: {len(train_idx)}")
    print(f"  Cal    StimID: {len(cal_stimids):4d} | rows: {len(cal_idx)}")
    print(f"  Test   StimID: {len(test_stimids):4d} | rows: {len(test_idx)}")
    if val_frac is not None:
        print(f"\nINTERNAL TRAIN/VAL")
        print(f"  tr    StimID: {len(tr_stimids):4d} | rows: {len(tr_idx)}")
        print(f"  val   StimID: {len(val_stimids):4d} | rows: {len(val_idx)}")
    else:
        print(f"\nNo internal train/val split (val_frac=None): tr rows: {len(tr_idx)}")
    print(f"\nClass distribution (y):")
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        vals, cnts = np.unique(y[idx], return_counts=True)
        print(f"  {name:5s}: { {int(v): int(c) for v,c in zip(vals,cnts)} }")

    stim_ids = df["StimID"].to_numpy()

    if return_torch:
        X   = torch.from_numpy(X)
        y   = torch.from_numpy(y)
        a_0 = torch.from_numpy(a_0)
        a_1 = torch.from_numpy(a_1)

    return X, y, a_0, a_1, train_idx, cal_idx, test_idx, tr_idx, val_idx, stim_ids


@dataclass
class TabularInteractionDataset(Dataset):
    """
    Tabular scenario, one item per instance.

    X:  (N, in_dim) float
    A0: (N,) long   -- human decision without disclosure (d=0)
    A1: (N,) long   -- human decision with disclosure (d=1)
    Y:  (N,) long   -- ground truth
    """
    X: torch.Tensor
    A0: torch.Tensor
    A1: torch.Tensor
    Y: torch.Tensor

    def __post_init__(self):
        if self.X.dtype != torch.float32:
            self.X = self.X.float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> dict:
        a0 = self.A0[idx]
        a1 = self.A1[idx]
        y = self.Y[idx]
        return {
            "x": self.X[idx],
            "a0": a0,
            "a1": a1,
            "y": y,
            "err0": (a0 != y).long(),
            "err1": (a1 != y).long(),
        }


class LabelDataModule(pl.LightningDataModule):
    """Batches (x, y) for the label model f of the ClasswiseRisk estimator."""

    def __init__(
            self,
            dataset: TabularInteractionDataset,
            train_idx: Sequence[int],
            val_idx: Sequence[int],
            batch_size: int = 128,
            num_workers: int = 0,
    ):
        super().__init__()
        self.dataset = dataset
        self.train_idx = list(train_idx)
        self.val_idx = list(val_idx)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _collate(self, batch):
        return {
            "x": torch.stack([b["x"] for b in batch]).float(),
            "y": torch.stack([b["y"] for b in batch]).long(),
        }

    def train_dataloader(self):
        return DataLoader(
            Subset(self.dataset, self.train_idx),
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=self._collate,
        )

    def val_dataloader(self):
        return DataLoader(
            Subset(self.dataset, self.val_idx),
            batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=self._collate,
        )


class ErrorDataModule(pl.LightningDataModule):
    """Batches (x, y, err_d) for the class-conditional error head g_d of the
    ClasswiseRisk estimator, with err_d = 1{A_d != Y}."""

    def __init__(
            self,
            dataset: TabularInteractionDataset,
            train_idx: Sequence[int],
            val_idx: Sequence[int],
            regime: int,
            batch_size: int = 32,
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
            "y": torch.stack([b["y"] for b in batch]).long(),
            "err": torch.stack([b[err_key] for b in batch]).long(),
        }

    def train_dataloader(self):
        return DataLoader(
            Subset(self.dataset, self.train_idx),
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=self._collate,
        )

    def val_dataloader(self):
        return DataLoader(
            Subset(self.dataset, self.val_idx),
            batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=self._collate,
        )

    def compute_pos_weight(
            self,
            eps: float = 1e-3,
            cap: float = 10.0,
    ) -> float:
        """
        BCE pos_weight = n_neg / n_pos on the train split of this regime,
        clamped to [1/cap, cap]. The positive class is the error event
        (A_d != Y), usually the minority one, so the weight is typically > 1;
        clamping avoids destabilizing the loss under extreme imbalance.
        """
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
