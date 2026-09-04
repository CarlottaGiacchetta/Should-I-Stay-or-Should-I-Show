import torch
import numpy as np
import pandas as pd
import os
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from torchvision.transforms import InterpolationMode
from collections import Counter
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


FILL = 0


def aug_identity(img):
    return img


def aug_hflip(img):
    return TF.hflip(img)


def aug_vflip(img):
    return TF.vflip(img)


def aug_rot45(img):
    return TF.rotate(img, 45, interpolation=InterpolationMode.BILINEAR, fill=FILL)


def aug_rot30(img):
    return TF.rotate(img, 30, interpolation=InterpolationMode.BILINEAR, fill=FILL)


def aug_rot15(img):
    return TF.rotate(img, 15, interpolation=InterpolationMode.BILINEAR, fill=FILL)


def aug_shift_right(img):
    return TF.affine(
        img, angle=0, translate=[8, 0], scale=1.0, shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR, fill=FILL
    )


def aug_shift_left(img):
    return TF.affine(
        img, angle=0, translate=[-8, 0], scale=1.0, shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR, fill=FILL
    )


def aug_shift_up(img):
    return TF.affine(
        img, angle=0, translate=[0, -8], scale=1.0, shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR, fill=FILL
    )


def aug_shift_down(img):
    return TF.affine(
        img, angle=0, translate=[0, 8], scale=1.0, shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR, fill=FILL
    )


def aug_shear_x_pos(img):
    return TF.affine(
        img, angle=0, translate=[0, 0], scale=1.0, shear=[8.0, 0.0],
        interpolation=InterpolationMode.BILINEAR, fill=FILL
    )


def aug_shear_x_neg(img):
    return TF.affine(
        img, angle=0, translate=[0, 0], scale=1.0, shear=[-8.0, 0.0],
        interpolation=InterpolationMode.BILINEAR, fill=FILL
    )


def aug_shear_y_pos(img):
    return TF.affine(
        img, angle=0, translate=[0, 0], scale=1.0, shear=[0.0, 8.0],
        interpolation=InterpolationMode.BILINEAR, fill=FILL
    )


def aug_shear_y_neg(img):
    return TF.affine(
        img, angle=0, translate=[0, 0], scale=1.0, shear=[0.0, -8.0],
        interpolation=InterpolationMode.BILINEAR, fill=FILL
    )


AUGMENTATIONS = [
    ("orig", aug_identity),
    ("hflip", aug_hflip),
    ("vflip", aug_vflip),
    ("rot45", aug_rot45),
    ("rot30", aug_rot30),
    ("rot15", aug_rot15),
    ("shift_right", aug_shift_right),
    ("shift_left", aug_shift_left),
    ("shift_up", aug_shift_up),
    ("shift_down", aug_shift_down),
    ("shear_x_pos", aug_shear_x_pos),
    ("shear_x_neg", aug_shear_x_neg),
    ("shear_y_pos", aug_shear_y_pos),
    ("shear_y_neg", aug_shear_y_neg),
]


def apply_n_random_augs(img, n=3):
    """Compose n augmentations drawn at random from AUGMENTATIONS."""
    fns = [fn for _, fn in AUGMENTATIONS]
    idxs = torch.randint(low=0, high=len(fns), size=(n,))
    for i in idxs:
        img = fns[int(i)](img)
    return img


def _print_split_class_dist(name, idx, labels):
    split_labels = labels[idx]
    counts = Counter(split_labels)
    total = len(split_labels)
    print(f"\nClass distribution in {name} (n={total}):")
    for cls, c in sorted(counts.items()):
        frac = c / total
        print(f"  class {cls:2d}: {c:4d}  ({frac:6.2%})")


def build_dataset(alpha=.01, noise_level=110,
                  path_preds="vgg19_epoch10_preds.csv",
                  path_human_only="human_only_classification.csv",
                  path_human_support="predictions_lenient.csv"):
    """Build the two tables of human response distributions, one per regime.

    The conformal threshold at level `alpha` is computed on the classifier
    softmax scores and turned into a prediction-set size for every image. The
    d=0 table aggregates the responses given without any support, the d=1 table
    those given while seeing the conformal prediction set of the same size.
    Returns (df_0, df_1), each with one row per image and one column per class.
    """
    mp = pd.read_csv(path_preds)
    ho = pd.read_csv(path_human_only)
    ho = ho[ho['noise_level'] == noise_level].reset_index(drop=True)
    ho["prediction"] = ho["participant_classification"].copy()
    hp = pd.read_csv(path_human_support)
    hp = hp[hp["prediction"] != "none"].copy().reset_index(drop=True)
    Y = mp['category'].values
    logits_cols = sorted(['knife', 'keyboard', 'elephant',
       'bicycle', 'airplane', 'clock', 'oven', 'chair', 'bear', 'boat', 'cat',
       'bottle', 'truck', 'car', 'bird', 'dog'])
    logits = mp[logits_cols].to_numpy()
    exp_logits = np.exp(logits)
    softmax = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    softmax = pd.DataFrame(softmax, columns=logits_cols)

    print(f"\n[DEBUG] Computing conformal threshold for alpha={alpha}...\n")
    conformal_score = 1 - softmax.to_numpy()[np.arange(len(mp)), softmax.columns.get_indexer(Y)]
    threshold = np.quantile(conformal_score, (1 - alpha) * (len(mp) + 1) / len(mp))
    q = 1 - threshold
    mask = softmax >= q

    mp['set'] = mask.sum(axis=1)
    print(mp['set'].value_counts())
    df = mp[["image_name", "set", "category"]].copy()

    df_1 = df.merge(hp[['image_name', 'prediction', 'set']],
                    on=['image_name', 'set'], how="inner").reset_index(drop=True)
    df_1 = pd.concat([df_1, pd.get_dummies(df_1["prediction"])], axis=1)
    tmp_1 = df_1.groupby(["image_name", "set"], as_index=False).agg({el: ["mean"] for el in logits_cols})
    tmp_1.columns = ["image_name", "set"] + logits_cols

    df_0 = df.merge(ho[['image_name', 'prediction']],
                    on=['image_name'], how="inner").reset_index(drop=True)
    df_0 = pd.concat([df_0, pd.get_dummies(df_0["prediction"])], axis=1)
    tmp_0 = df_0.groupby(["image_name"], as_index=False).agg({el: ["mean"] for el in logits_cols})
    tmp_0.columns = ["image_name"] + logits_cols

    return tmp_0, tmp_1


class ImageNet10kImageFolder(torch.utils.data.Dataset):
    """Images plus the two human decisions attached to each of them.

    a_0 and a_1 are sampled from the per-image response distributions df_0
    (no support) and df_1 (conformal prediction set shown). At training time
    the sampling is re-drawn at every epoch; at evaluation time it is seeded
    per index so that the decisions are fixed.
    """

    def __init__(self, root, df_0, df_1, prob_cols, id_col='id', transform=None,
                 training=True, seed=42):
        self.imgs = ImageFolder(root, transform=transform)
        self.training = training
        self.seed = seed
        self.class_to_idx = self.imgs.class_to_idx

        # the probability columns must follow the label indices assigned by
        # ImageFolder, otherwise probs_d[i, y] would read the wrong class
        expected_order = sorted(self.class_to_idx, key=self.class_to_idx.get)
        assert list(prob_cols) == expected_order, (
            f"prob_cols order {list(prob_cols)} does not match class_to_idx order "
            f"{expected_order}."
        )

        probs_df_0 = df_0.set_index(id_col)[prob_cols]
        probs_df_1 = df_1.set_index(id_col)[prob_cols]
        order = [self._id_from_path(p) for p, _ in self.imgs.samples]
        self.probs_0 = torch.tensor(probs_df_0.loc[order].values, dtype=torch.float32)
        self.probs_1 = torch.tensor(probs_df_1.loc[order].values, dtype=torch.float32)

        self.ids = list(order)
        set_size_df = df_1.set_index(id_col)["set"]
        self.pred_set_size = torch.tensor(
            set_size_df.loc[order].to_numpy(), dtype=torch.long
        )

    @staticmethod
    def _id_from_path(path):
        return os.path.splitext(os.path.basename(path))[0]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img, y = self.imgs[i]

        if self.training:
            a_0 = torch.multinomial(self.probs_0[i], 1).squeeze(0).long()
            a_1 = torch.multinomial(self.probs_1[i], 1).squeeze(0).long()
        else:
            g = torch.Generator().manual_seed(int(self.seed) + int(i))
            a_0 = torch.multinomial(self.probs_0[i], 1, generator=g).squeeze(0).long()
            a_1 = torch.multinomial(self.probs_1[i], 1, generator=g).squeeze(0).long()

        y = torch.tensor(y, dtype=torch.long)

        return {
            "x": img,
            "a0": a_0,
            "a1": a_1,
            "y": y,
            "err0": (a_0 != y).long(),
            "err1": (a_1 != y).long(),
            "id": self.ids[i],
            "pred_set_size": self.pred_set_size[i],
        }


def create_cp(
    base_folder="Datasets/imagenet1kH/",
    val_frac: float | None = 0.1,
    random_state: int = 42,
    splits=(0.7, 0.1, 0.2),
    batch_size: int = 32,
    alpha: float = 0.01,
):
    """Build the image scenario loaders (train/cal/val/test) for a conformal
    level `alpha`. The splits are stratified by class; the validation split is
    carved out of train. Returns the four loaders and the number of classes.
    """
    assert len(splits) == 3, "splits must be (train, calibration, test)"
    _, cal_frac, test_frac = splits
    assert abs(sum(splits) - 1.0) < 1e-8, "splits must sum to 1"
    assert val_frac is None or (0.0 <= val_frac < 1.0), "val_frac must be None or in [0, 1)"
    val_frac = val_frac or 0.0

    print(f"\n[INFO] Creating CP dataset with alpha={alpha}, splits={splits}, val_frac={val_frac}...\n")

    df_0, df_1 = build_dataset(
        alpha=alpha,
        noise_level=110,
        path_preds=base_folder + "vgg19_epoch10_preds.csv",
        path_human_only=base_folder + "human_only_classification.csv",
        path_human_support=base_folder + "predictions_lenient.csv",
    )

    logits_cols = sorted([
        'knife', 'keyboard', 'elephant', 'bicycle', 'airplane', 'clock',
        'oven', 'chair', 'bear', 'boat', 'cat', 'bottle', 'truck',
        'car', 'bird', 'dog'
    ])

    train_transform = T.Compose([
        T.Lambda(lambda img: apply_n_random_augs(img, n=3)),
        T.ToTensor(),
    ])

    eval_transform = T.Compose([
        T.ToTensor(),
    ])

    train_base_dataset = ImageNet10kImageFolder(
        root=base_folder + "phase_noise_110_IF",
        df_0=df_0,
        df_1=df_1,
        prob_cols=logits_cols,
        id_col='image_name',
        transform=train_transform,
        training=True,
        seed=random_state,
    )

    eval_base_dataset = ImageNet10kImageFolder(
        root=base_folder + "phase_noise_110_IF",
        df_0=df_0,
        df_1=df_1,
        prob_cols=logits_cols,
        id_col='image_name',
        transform=eval_transform,
        training=False,
        seed=random_state,
    )

    print(f"\n[INFO] Label to index map: {train_base_dataset.class_to_idx}\n")

    n = len(train_base_dataset)
    all_idx = np.arange(n)
    all_labels = np.array([int(train_base_dataset[i]["y"]) for i in range(n)], dtype=np.int64)

    trainval_idx, caltest_idx = train_test_split(
        all_idx,
        test_size=(cal_frac + test_frac),
        random_state=random_state,
        shuffle=True,
        stratify=all_labels,
    )

    test_relative = test_frac / (cal_frac + test_frac)
    cal_idx, test_idx = train_test_split(
        caltest_idx,
        test_size=test_relative,
        random_state=random_state,
        shuffle=True,
        stratify=all_labels[caltest_idx],
    )

    if val_frac > 0.0:
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=val_frac,
            random_state=random_state,
            shuffle=True,
            stratify=all_labels[trainval_idx],
        )
    else:
        train_idx = trainval_idx
        val_idx = np.array([], dtype=all_idx.dtype)

    _print_split_class_dist("TRAIN", train_idx, all_labels)
    _print_split_class_dist("CAL",   cal_idx,   all_labels)
    _print_split_class_dist("VAL",   val_idx,   all_labels)
    _print_split_class_dist("TEST",  test_idx,  all_labels)

    train_dataset = Subset(train_base_dataset, train_idx)
    cal_dataset = Subset(eval_base_dataset, cal_idx)
    val_dataset = Subset(eval_base_dataset, val_idx)
    test_dataset = Subset(eval_base_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=7)
    cal_loader   = DataLoader(cal_dataset,   batch_size=batch_size, shuffle=False, num_workers=7)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=7)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=7)

    print(f"Total samples:      {n}")
    print(f"Train samples:      {len(train_dataset)}")
    print(f"Calibration samples:{len(cal_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Final test samples: {len(test_dataset)}")

    K = len(logits_cols)

    return train_loader, cal_loader, val_loader, test_loader, K


def _error_collate(batch, regime: int):
    err_key = "err0" if regime == 0 else "err1"
    return {
        "x": torch.stack([b["x"] for b in batch]).float(),
        "y": torch.stack([b["y"] for b in batch]).long(),
        "err": torch.stack([b[err_key] for b in batch]).float(),
    }


def _risk_collate(batch, regime: int):
    err_key = "err0" if regime == 0 else "err1"
    return {
        "x": torch.stack([b["x"] for b in batch]).float(),
        "err": torch.stack([b[err_key] for b in batch]).float(),
    }


def compute_pos_weight_from_cp_train(
    train_loader: DataLoader,
    regime: int,
    eps: float = 1e-3,
    cap: float = 10.0,
) -> float:
    """BCE pos_weight = n_neg / n_pos over the image train loader for regime d,
    clamped to [1/cap, cap]."""
    assert regime in (0, 1), "regime must be 0 or 1"
    err_key = "err0" if regime == 0 else "err1"

    n_pos = 0
    n_neg = 0
    for batch in train_loader:
        err = batch[err_key].float()
        n_pos += (err == 1).sum().item()
        n_neg += (err == 0).sum().item()

    n_total = n_pos + n_neg
    if n_total == 0:
        raise RuntimeError("No samples in the CP train loader when computing pos_weight.")

    p_err = max(min(n_pos / n_total, 1.0 - eps), eps)
    pw = (1.0 - p_err) / p_err
    return float(min(max(pw, 1.0 / cap), cap))


def make_cp_error_loaders(
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    regime: int,
    batch_size: int,
):
    """Re-wrap the image loaders with the (x, y, err_d) collate used by the
    error heads, and compute pos_weight on train. Returns
    (train_loader, val_loader, pos_weight)."""
    pos_weight = compute_pos_weight_from_cp_train(train_loader, regime=regime)
    print(f"\nComputed pos_weight for regime {regime}: {pos_weight:.4f}")

    error_train_loader = DataLoader(
        train_loader.dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=getattr(train_loader, "num_workers", 0),
        collate_fn=lambda batch: _error_collate(batch, regime=regime),
    )

    if val_loader is None:
        return error_train_loader, None, pos_weight

    error_val_loader = DataLoader(
        val_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=getattr(val_loader, "num_workers", 0),
        collate_fn=lambda batch: _error_collate(batch, regime=regime),
    )

    return error_train_loader, error_val_loader, pos_weight


def make_cp_risk_loaders(
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    regime: int,
    batch_size: int,
):
    """Re-wrap the image loaders with the (x, err_d) collate used by the direct
    risk models, and compute pos_weight on train. Returns
    (train_loader, val_loader, pos_weight)."""
    pos_weight = compute_pos_weight_from_cp_train(train_loader, regime=regime)

    risk_train_loader = DataLoader(
        train_loader.dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=getattr(train_loader, "num_workers", 0),
        collate_fn=lambda batch: _risk_collate(batch, regime=regime),
    )

    if val_loader is None:
        return risk_train_loader, None, pos_weight

    risk_val_loader = DataLoader(
        val_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=getattr(val_loader, "num_workers", 0),
        collate_fn=lambda batch: _risk_collate(batch, regime=regime),
    )

    return risk_train_loader, risk_val_loader, pos_weight
