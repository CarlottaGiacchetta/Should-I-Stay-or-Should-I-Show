from pathlib import Path
import argparse
import json

import numpy as np
import torch
import pytorch_lightning as pl

torch.set_float32_matmul_precision("medium")

from src.datasets import (
    TabularInteractionDataset,
    LabelDataModule,
    ErrorDataModule,
    generate_synthetic_data,
    email_sampled,
)
from src.models import (
    LabelClassifier,
    ErrorHeadModel,
    VoIEstimator,
    BinaryRegimeRiskModel,
    RegimeRiskTLearner,
    ImageLabelClassifier,
    ImageErrorHeadModel,
    ImageBinaryRegimeRiskModel,
    SynthLabelClassifier,
    SynthErrorHeadModel,
    SynthBinaryRegimeRiskModel,
)
from src.methods import standardize_from_train, RegimeRiskDataModule
from src.utils import (
    _eval_coverage,
    build_run_name,
    aggregate_multirun_stats,
    plot_multirun_results,
    plot_model_comparison,
    _collect_image_policy_outputs,
    build_final_results_csv,
    PrintMetricsCallback,
    _fmt_float_for_name,
    alpha_dirname,
)
from src.cp_datasets_class import create_cp, make_cp_error_loaders, make_cp_risk_loaders

DATASET_SEED = 42

MODELS = ["ClasswiseRisk", "DirectRisk", "Random"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final run: trains and evaluates ClasswiseRisk, DirectRisk and Random "
                     "for --dataset, with fixed hyperparameters (train/cal/test split, "
                     "seed=--dataset_seed)."
    )
    parser.add_argument("--dataset", type=str, default="email", choices=["synth", "email", "cp"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                         help="Seeds for model initialization/training (multirun). "
                              "The dataset split always uses --dataset_seed.")
    parser.add_argument("--dataset_seed", type=int, default=DATASET_SEED,
                         help="Seed of the dataset split (train/val/cal/test), independent of "
                              "--seeds and kept fixed across a multirun so that every seed is "
                              "compared on exactly the same data.")
    parser.add_argument("--splits", type=float, nargs=3, default=[0.7, 0.1, 0.2],
                         help="Train/cal/test split ratios (must sum to 1.0)")
    parser.add_argument("--val_frac", type=float, default=0.1,
                         help="Fraction of train carved out for validation. Kept so that the split "
                              "matches the one used to select the hyperparameters; validation "
                              "itself is unused here, since the final training has no early stopping")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--config_path", type=str, default="configs/best_hparams.json",
                         help="JSON file with the selected hyperparameters, per dataset and model")

    # image scenario
    parser.add_argument("--alpha", type=float, default=0.01, help="conformal alpha (dataset=cp only)")
    parser.add_argument("--cp_feedback", type=str, default="lenient", choices=["lenient", "strict"])
    parser.add_argument("--freeze_backbone", action="store_true")

    # synthetic scenario (dataset construction only)
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_features", type=int, default=20)
    parser.add_argument("--n_hidden", type=int, default=11)
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--class_sep", type=float, default=1.0)

    # fallback hyperparameters, only used where no tuned combination applies:
    # the Random model (never trained) and the `elr` suffix of the run name
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--error_lr", type=float, default=1e-4)
    parser.add_argument("--risk_dropout", type=float, default=0.05)
    parser.add_argument("--risk_weight_decay", type=float, default=1e-4)

    # shared architecture/training options
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--risk_batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--error_hidden_dim", type=int, default=256)
    parser.add_argument("--risk_hidden_dim", type=int, default=128)

    return parser.parse_args()


HPARAM_KEYS = '{"epochs": ..., "optimizer": ..., "lr": ..., "dropout": ..., "weight_decay": ...}'


def load_tuned_hparams(config_path, dataset, model, alpha=None):
    """Read the selected hyperparameters for (dataset, model) from the JSON
    config. The image scenario keys them by conformal alpha first, since a
    different alpha reshapes the feedback the models are trained on."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found.")
    config = json.loads(config_path.read_text())

    if dataset == "cp":
        alpha_key = _fmt_float_for_name(alpha)
        try:
            return config[dataset][alpha_key][model]
        except KeyError:
            raise KeyError(
                f"No hyperparameters for dataset=cp alpha={alpha} model={model} in "
                f'{config_path}. Add config["cp"]["{alpha_key}"]["{model}"] = {HPARAM_KEYS}.'
            )

    try:
        return config[dataset][model]
    except KeyError:
        raise KeyError(
            f"No hyperparameters for dataset={dataset} model={model} in {config_path}. "
            f'Add config["{dataset}"]["{model}"] = {HPARAM_KEYS}.'
        )


def build_dataset(dataset, splits, val_frac, batch_size, alpha, n_samples, n_features, n_hidden,
                  n_classes, class_sep, dataset_seed=DATASET_SEED):
    """Build the data once, always with seed=dataset_seed, and return a bundle
    of everything the per-seed training needs."""
    if dataset == "cp":
        print(f"[DEBUG] Loading CP image dataset (4-way split train/val/cal/test, "
              f"seed={dataset_seed})")
        train_loader, cal_loader, val_loader, test_loader, K = create_cp(
            val_frac=val_frac,
            random_state=dataset_seed,
            splits=splits,
            batch_size=batch_size,
            alpha=alpha,
        )
        return {
            "dataset_type": "image",
            "train_loader": train_loader,
            "cal_loader": cal_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "K": K,
        }

    if dataset == "email":
        data_dir = Path(__file__).resolve().parent / "Datasets" / "email"
        print(f"[DEBUG] Loading email dataset (4-way split train/val/cal/test, "
              f"seed={dataset_seed})")
        X, y, a_0, a_1, train_idx, cal_idx, test_idx, tr_idx, val_idx, stim_ids = email_sampled(
            emb_path=data_dir / "embeddings_aug.csv",
            return_torch=True,
            val_frac=val_frac,
            splits=splits,
            random_state=dataset_seed,
        )
        X = X.float()
        return {
            "dataset_type": "tabular",
            "X": X, "y": y, "a_0": a_0, "a_1": a_1,
            "train_idx": tr_idx, "val_idx": val_idx, "cal_idx": cal_idx, "test_idx": test_idx,
            "ids": stim_ids,
            "in_dim": X.shape[1], "K": int(y.max().item()) + 1,
        }

    if dataset == "synth":
        X, y, a_0, a_1 = generate_synthetic_data(
            n_samples=n_samples, n_features=n_features, n_hidden=n_hidden,
            n_informative=n_features, n_classes=n_classes,
            random_state=dataset_seed, splits=splits, class_sep=class_sep,
            return_torch=True,
        )
        X = X.float()
        N = X.shape[0]
        rng = np.random.default_rng(dataset_seed)
        perm = rng.permutation(N)
        n_tr = int(splits[0] * N)
        n_cal = int(splits[1] * N)
        train_idx = perm[:n_tr]
        cal_idx = perm[n_tr:n_tr + n_cal]
        test_idx = perm[n_tr + n_cal:]
        return {
            "dataset_type": "tabular",
            "X": X, "y": y, "a_0": a_0, "a_1": a_1,
            "train_idx": train_idx, "cal_idx": cal_idx, "test_idx": test_idx,
            "ids": None,
            "in_dim": X.shape[1], "K": int(y.max().item()) + 1,
        }

    raise ValueError("Invalid dataset. Choose 'synth', 'email', or 'cp'.")


def train_cp_one_seed(bundle, model, hparams, hidden_dim, error_hidden_dim, risk_hidden_dim,
                       risk_batch_size, freeze_backbone):
    """Train one VoI estimator on the image scenario, for a single seed and a
    fixed hyperparameter combination."""
    train_loader = bundle["train_loader"]
    val_loader = bundle["val_loader"]
    K = bundle["K"]

    if model == "Random":
        return {"dataset_type": "image", "voi_est": None, **bundle}

    if model == "ClasswiseRisk":
        f = ImageLabelClassifier(
            num_classes=K, lr=hparams["lr"], weight_decay=hparams["weight_decay"],
            pretrained=True, freeze_backbone=freeze_backbone, head_hidden=hidden_dim,
            dropout=hparams["dropout"], optimizer=hparams["optimizer"],
        )
        pl.Trainer(
            max_epochs=hparams["epochs"], enable_progress_bar=False, enable_model_summary=False,
            logger=False, enable_checkpointing=False,
            callbacks=[PrintMetricsCallback(name="CP-Label", every_n_epochs=10)],
        ).fit(f, train_dataloaders=train_loader, val_dataloaders=val_loader)

        error_models = {}
        for d in (0, 1):
            err_train_loader, err_val_loader, pos_weight = make_cp_error_loaders(
                train_loader=train_loader, val_loader=val_loader, regime=d,
                batch_size=train_loader.batch_size,
            )
            g = ImageErrorHeadModel(
                num_classes=K, regime=d, lr=hparams["lr"], weight_decay=hparams["weight_decay"],
                pretrained=True, freeze_backbone=freeze_backbone, head_hidden=error_hidden_dim,
                dropout=hparams["dropout"], pos_weight=pos_weight, optimizer=hparams["optimizer"],
            )
            pl.Trainer(
                max_epochs=hparams["epochs"], enable_progress_bar=False, enable_model_summary=False,
                logger=False, enable_checkpointing=False,
                callbacks=[PrintMetricsCallback(name=f"CP-Error-d{d}", every_n_epochs=10)],
            ).fit(g, train_dataloaders=err_train_loader, val_dataloaders=err_val_loader)
            error_models[d] = g

        voi_est = VoIEstimator(
            label_model=f, error_model_d0=error_models[0], error_model_d1=error_models[1],
        ).eval()

    elif model == "DirectRisk":
        risk_models = {}
        for d in (0, 1):
            risk_train_loader, risk_val_loader, pos_weight = make_cp_risk_loaders(
                train_loader=train_loader, val_loader=val_loader, regime=d,
                batch_size=risk_batch_size,
            )
            g = ImageBinaryRegimeRiskModel(
                regime=d, lr=hparams["lr"], weight_decay=hparams["weight_decay"],
                dropout=hparams["dropout"], pretrained=True, freeze_backbone=freeze_backbone,
                head_hidden=risk_hidden_dim, pos_weight=pos_weight, optimizer=hparams["optimizer"],
            )
            pl.Trainer(
                max_epochs=hparams["epochs"], enable_progress_bar=False, enable_model_summary=False,
                logger=False, enable_checkpointing=False,
                callbacks=[PrintMetricsCallback(name=f"CP-Risk-d{d}", every_n_epochs=10)],
            ).fit(g, train_dataloaders=risk_train_loader, val_dataloaders=risk_val_loader)
            risk_models[d] = g

        voi_est = RegimeRiskTLearner(risk_model_d0=risk_models[0], risk_model_d1=risk_models[1]).eval()

    else:
        raise ValueError("Invalid model. Choose 'ClasswiseRisk', 'DirectRisk' or 'Random'.")

    return {"dataset_type": "image", "voi_est": voi_est, **bundle}


def _empty_idx_like(idx):
    return np.array([], dtype=idx.dtype)


def train_tabular_tuned_one_seed(bundle, model, hparams, batch_size, risk_batch_size,
                                  hidden_dim, error_hidden_dim, risk_hidden_dim):
    """Train one VoI estimator on the email scenario, for a single seed and the
    fixed hyperparameter combination read from the config (shared by the label
    model and its error/risk heads). No validation split: fit on train only."""
    X, y, a_0, a_1 = bundle["X"], bundle["y"], bundle["a_0"], bundle["a_1"]
    train_idx, cal_idx, test_idx = bundle["train_idx"], bundle["cal_idx"], bundle["test_idx"]
    val_idx = bundle["val_idx"]
    IN_DIM, K = bundle["in_dim"], bundle["K"]

    ds = TabularInteractionDataset(X=X, A0=a_0, A1=a_1, Y=y)
    X_eval = X

    if model == "Random":
        print("Using the random baseline (no VoI estimate)...")
        voi_est = None

    elif model == "ClasswiseRisk":
        label_dm = LabelDataModule(ds, train_idx, val_idx, batch_size=batch_size)
        f = LabelClassifier(
            in_dim=IN_DIM, num_classes=K, hidden=hidden_dim, lr=hparams["lr"],
            weight_decay=hparams["weight_decay"], dropout=hparams["dropout"],
            optimizer=hparams["optimizer"],
        )
        pl.Trainer(
            max_epochs=hparams["epochs"], enable_progress_bar=False, enable_model_summary=False,
            logger=False, enable_checkpointing=False,
            callbacks=[PrintMetricsCallback(name="Tabular-Label", every_n_epochs=5)],
        ).fit(f, datamodule=label_dm)

        error_models = {}
        for d in (0, 1):
            err_dm = ErrorDataModule(ds, train_idx, val_idx, regime=d, batch_size=batch_size)
            pw = err_dm.compute_pos_weight()
            g = ErrorHeadModel(
                in_dim=IN_DIM, num_classes=K, regime=d, hidden=error_hidden_dim, lr=hparams["lr"],
                weight_decay=hparams["weight_decay"], dropout=hparams["dropout"],
                optimizer=hparams["optimizer"], pos_weight=pw,
            )
            pl.Trainer(
                max_epochs=hparams["epochs"], enable_progress_bar=False, enable_model_summary=False,
                logger=False, enable_checkpointing=False,
                callbacks=[PrintMetricsCallback(name=f"Tabular-Error-d{d}", every_n_epochs=5)],
            ).fit(g, datamodule=err_dm)
            error_models[d] = g

        voi_est = VoIEstimator(
            label_model=f, error_model_d0=error_models[0], error_model_d1=error_models[1],
        ).eval()

    elif model == "DirectRisk":
        X_eval = standardize_from_train(X, train_idx)
        risk_ds = TabularInteractionDataset(X=X_eval, A0=a_0, A1=a_1, Y=y)

        risk_models = {}
        for d in (0, 1):
            risk_dm = RegimeRiskDataModule(risk_ds, train_idx, val_idx, regime=d,
                                           batch_size=risk_batch_size)
            pw = risk_dm.compute_pos_weight()
            g = BinaryRegimeRiskModel(
                in_dim=IN_DIM, regime=d, hidden=risk_hidden_dim, lr=hparams["lr"],
                weight_decay=hparams["weight_decay"], dropout=hparams["dropout"],
                optimizer=hparams["optimizer"], pos_weight=pw,
            )
            pl.Trainer(
                max_epochs=hparams["epochs"], enable_progress_bar=False, enable_model_summary=False,
                logger=False, enable_checkpointing=False,
                callbacks=[PrintMetricsCallback(name=f"Tabular-Risk-d{d}", every_n_epochs=5)],
            ).fit(g, datamodule=risk_dm)
            risk_models[d] = g

        voi_est = RegimeRiskTLearner(risk_model_d0=risk_models[0], risk_model_d1=risk_models[1]).eval()

    else:
        raise ValueError("Invalid model. Choose 'ClasswiseRisk', 'DirectRisk' or 'Random'.")

    return {
        "dataset_type": "tabular",
        "voi_est": voi_est,
        "X_eval": X_eval,
        "y": y, "a_0": a_0, "a_1": a_1,
        "cal_idx": cal_idx, "test_idx": test_idx,
        "ids": bundle["ids"],
    }


SYNTH_BATCH_SIZE = 32
SYNTH_ERROR_LR = 1e-4


def train_synth_one_seed(bundle, model, hparams, hidden_dim, error_hidden_dim, risk_hidden_dim,
                          risk_batch_size):
    """Train one VoI estimator on the synthetic scenario. The models used here
    are the simpler Synth* variants (plain MLPs, always Adam), so the
    'dropout'/'optimizer' hyperparameters are only read where they apply."""
    X, y, a_0, a_1 = bundle["X"], bundle["y"], bundle["a_0"], bundle["a_1"]
    train_idx, cal_idx, test_idx = bundle["train_idx"], bundle["cal_idx"], bundle["test_idx"]
    IN_DIM, K = bundle["in_dim"], bundle["K"]

    ds = TabularInteractionDataset(X=X, A0=a_0, A1=a_1, Y=y)
    X_eval = X
    no_val = _empty_idx_like(train_idx)

    if model == "Random":
        print("Using the random baseline (no VoI estimate)...")
        voi_est = None

    elif model == "ClasswiseRisk":
        label_dm = LabelDataModule(ds, train_idx, no_val, batch_size=SYNTH_BATCH_SIZE)
        f = SynthLabelClassifier(
            in_dim=IN_DIM, num_classes=K, hidden=hidden_dim, lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
        )
        pl.Trainer(
            max_epochs=hparams["epochs"], enable_progress_bar=False, enable_model_summary=False,
            logger=False, enable_checkpointing=False,
            callbacks=[PrintMetricsCallback(name="Tabular-Label", every_n_epochs=3, stage="train")],
        ).fit(f, train_dataloaders=label_dm.train_dataloader())

        error_models = {}
        for d in (0, 1):
            err_dm = ErrorDataModule(ds, train_idx, no_val, regime=d, batch_size=SYNTH_BATCH_SIZE)
            pw = err_dm.compute_pos_weight()
            g = SynthErrorHeadModel(
                in_dim=IN_DIM, num_classes=K, regime=d, hidden=error_hidden_dim,
                lr=SYNTH_ERROR_LR, weight_decay=hparams["weight_decay"], pos_weight=pw,
            )
            pl.Trainer(
                max_epochs=hparams["epochs"], enable_progress_bar=False, enable_model_summary=False,
                logger=False, enable_checkpointing=False,
                callbacks=[PrintMetricsCallback(name=f"Tabular-Error-d{d}", every_n_epochs=3, stage="train")],
            ).fit(g, train_dataloaders=err_dm.train_dataloader())
            error_models[d] = g

        voi_est = VoIEstimator(
            label_model=f, error_model_d0=error_models[0], error_model_d1=error_models[1],
        ).eval()

    elif model == "DirectRisk":
        X_eval = standardize_from_train(X, train_idx)
        risk_ds = TabularInteractionDataset(X=X_eval, A0=a_0, A1=a_1, Y=y)

        risk_models = {}
        for d in (0, 1):
            risk_dm = RegimeRiskDataModule(risk_ds, train_idx, no_val, regime=d,
                                           batch_size=risk_batch_size)
            pw = risk_dm.compute_pos_weight()
            g = SynthBinaryRegimeRiskModel(
                in_dim=IN_DIM, regime=d, hidden=risk_hidden_dim, lr=hparams["lr"],
                weight_decay=hparams["weight_decay"], dropout=hparams["dropout"],
                pos_weight=pw,
            )
            pl.Trainer(
                max_epochs=hparams["epochs"], enable_progress_bar=False, enable_model_summary=False,
                logger=False, enable_checkpointing=False,
                callbacks=[PrintMetricsCallback(name=f"Tabular-Risk-d{d}", every_n_epochs=3, stage="train")],
            ).fit(g, train_dataloaders=risk_dm.train_dataloader())
            risk_models[d] = g

        voi_est = RegimeRiskTLearner(risk_model_d0=risk_models[0], risk_model_d1=risk_models[1]).eval()

    else:
        raise ValueError("Invalid model. Choose 'ClasswiseRisk', 'DirectRisk' or 'Random'.")

    return {
        "dataset_type": "tabular",
        "voi_est": voi_est,
        "X_eval": X_eval,
        "y": y, "a_0": a_0, "a_1": a_1,
        "cal_idx": cal_idx, "test_idx": test_idx,
        "ids": bundle["ids"],
    }


def save_trained_model(voi_est, comparison_dir, run_name, seed):
    """Save the trained estimator under <comparison_dir>/models/, one file per
    model and seed. Random trains nothing, so there is nothing to save."""
    if voi_est is None:
        return
    models_dir = comparison_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / f"{run_name}_seed{seed}.pt"
    torch.save(voi_est.state_dict(), save_path)
    print(f"[run] saved model state_dict to {save_path}")


def evaluate_tabular(run_output, *, model, dataset, results_dir, random_state, run_name):
    """Evaluate the policy on the test set and write the budget-accuracy rows
    plus the per-instance decision log."""
    voi_est = run_output["voi_est"]
    X_eval, y = run_output["X_eval"], run_output["y"]
    a_0, a_1 = run_output["a_0"], run_output["a_1"]
    cal_idx, test_idx = run_output["cal_idx"], run_output["test_idx"]

    X_cal, X_te = X_eval[cal_idx], X_eval[test_idx]
    a0_te, a1_te, y_te = a_0[test_idx], a_1[test_idx], y[test_idx]

    if model == "Random":
        r0_t = r1_t = voi_cal_t = voi_test_t = None
    else:
        voi_cal_t, voi_test_t = voi_est.voi(X_cal), voi_est.voi(X_te)
        r0_t, r1_t = voi_est.conditional_risk(X_te, 0), voi_est.conditional_risk(X_te, 1)

    obs_r0 = (a0_te != y_te).float().mean().item()
    obs_r1 = (a1_te != y_te).float().mean().item()
    if model != "Random":
        print("\n--- estimated vs observed regime risks (test set) ---")
        print(f"  r0_hat avg : {r0_t.mean().item():.3f}    observed r0 : {obs_r0:.3f}")
        print(f"  r1_hat avg : {r1_t.mean().item():.3f}    observed r1 : {obs_r1:.3f}")
        print(f"  mean VoI   : {voi_test_t.mean().item():+.3f}    "
              f"frac VoI>0  : {(voi_test_t > 0).float().mean().item():.2%}")

    base_results_dir = Path(results_dir)
    filepath = base_results_dir / dataset / f"{run_name}_multirun.csv"
    final_csv_path = base_results_dir / dataset / "final_results.csv"

    if model == "Random":
        accs = _eval_coverage(
            voi_test=None, voi_cal=None, a0_test=a0_te.numpy(), a1_test=a1_te.numpy(),
            y_test=y_te.numpy(), label="coverage-accuracy curve (random)", verbose=True,
            csv_path=filepath, dataset=dataset, model_type=model, seed=random_state,
            random_mode=True,
        )
    else:
        accs = _eval_coverage(
            voi_test=voi_test_t.numpy(), voi_cal=voi_cal_t.numpy(),
            a0_test=a0_te.numpy(), a1_test=a1_te.numpy(), y_test=y_te.numpy(),
            label="coverage-accuracy curve", verbose=True, csv_path=filepath,
            dataset=dataset, model_type=model, seed=random_state,
        )
        ids_all = run_output.get("ids")
        ids_te = ids_all[test_idx] if ids_all is not None else np.array(test_idx)
        build_final_results_csv(
            voi_test=voi_test_t.numpy(), voi_cal=voi_cal_t.numpy(),
            a0_test=a0_te.numpy(), a1_test=a1_te.numpy(), y_test=y_te.numpy(),
            ids_test=ids_te, csv_path=final_csv_path,
            dataset=dataset, model_type=model, seed=random_state,
        )

    print("\nsummary (coverage -> accuracy):")
    for c, a in accs.items():
        print(f"  {c:>4.2f}: {a:.4f}")


def evaluate_image(run_output, *, model, dataset, results_dir, random_state, run_name,
                   cp_feedback, alpha):
    """Image counterpart of `evaluate_tabular`, which also logs the conformal
    prediction-set size of each test instance."""
    voi_est = run_output["voi_est"]
    cal_loader, test_loader = run_output["cal_loader"], run_output["test_loader"]

    if model != "Random":
        try:
            device = next(voi_est.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model == "Random":
        test_out = _collect_image_policy_outputs(None, test_loader, model, device)
        voi_cal_t = voi_test_t = r0_t = r1_t = None
    else:
        cal_out = _collect_image_policy_outputs(voi_est, cal_loader, model, device)
        test_out = _collect_image_policy_outputs(voi_est, test_loader, model, device)
        voi_cal_t, voi_test_t = cal_out["voi"], test_out["voi"]
        r0_t, r1_t = test_out["r0"], test_out["r1"]

    a0_te, a1_te, y_te = test_out["a0"], test_out["a1"], test_out["y"]
    obs_r0 = (a0_te != y_te).float().mean().item()
    obs_r1 = (a1_te != y_te).float().mean().item()
    if model != "Random":
        print("\n--- estimated vs observed regime risks (test set, images) ---")
        print(f"  r0_hat avg : {r0_t.mean().item():.3f}    observed r0 : {obs_r0:.3f}")
        print(f"  r1_hat avg : {r1_t.mean().item():.3f}    observed r1 : {obs_r1:.3f}")
        print(f"  mean VoI   : {voi_test_t.mean().item():+.3f}    "
              f"frac VoI>0  : {(voi_test_t > 0).float().mean().item():.2%}")

    base_results_dir = Path(results_dir)
    cp_results_dir = base_results_dir / dataset / cp_feedback / alpha_dirname(alpha)
    filepath = cp_results_dir / f"{run_name}_multirun.csv"
    final_csv_path = cp_results_dir / "final_results.csv"

    if model == "Random":
        accs = _eval_coverage(
            voi_test=None, voi_cal=None, a0_test=a0_te.numpy(), a1_test=a1_te.numpy(),
            y_test=y_te.numpy(), label="coverage-accuracy curve (random)", verbose=True,
            csv_path=filepath, dataset=dataset, model_type=model, seed=random_state,
            random_mode=True,
        )
    else:
        accs = _eval_coverage(
            voi_test=voi_test_t.numpy(), voi_cal=voi_cal_t.numpy(),
            a0_test=a0_te.numpy(), a1_test=a1_te.numpy(), y_test=y_te.numpy(),
            label="coverage-accuracy curve", verbose=True, csv_path=filepath,
            dataset=dataset, model_type=model, seed=random_state,
        )
        pred_set_size_te = test_out.get("pred_set_size")
        build_final_results_csv(
            voi_test=voi_test_t.numpy(), voi_cal=voi_cal_t.numpy(),
            a0_test=a0_te.numpy(), a1_test=a1_te.numpy(), y_test=y_te.numpy(),
            ids_test=test_out["ids"], csv_path=final_csv_path,
            dataset=dataset, model_type=model, seed=random_state,
            pred_set_size_test=(pred_set_size_te.numpy() if pred_set_size_te is not None else None),
        )

    print("\nsummary (coverage -> accuracy):")
    for c, a in accs.items():
        print(f"  {c:>4.2f}: {a:.4f}")


def main():
    args = parse_args()
    splits = tuple(args.splits)

    print(f"{'#'*80}\nFINAL RUN\nDataset: {args.dataset}, Models: {MODELS}, "
          f"Seeds: {args.seeds}\n{'#'*80}\n")

    dataset_bundle = build_dataset(
        dataset=args.dataset, splits=splits, val_frac=args.val_frac,
        batch_size=args.batch_size, alpha=args.alpha,
        n_samples=args.n_samples, n_features=args.n_features, n_hidden=args.n_hidden,
        n_classes=args.n_classes, class_sep=args.class_sep,
        dataset_seed=args.dataset_seed,
    )

    base_results_dir = Path(args.results_dir)
    if args.dataset == "cp":
        comparison_dir = base_results_dir / args.dataset / args.cp_feedback / alpha_dirname(args.alpha)
    else:
        comparison_dir = base_results_dir / args.dataset
    final_csv_path = comparison_dir / "final_results.csv"
    if final_csv_path.exists():
        final_csv_path.unlink()

    aggregated_csv_by_model = {}

    for model in MODELS:
        print(f"\n{'#'*80}\nMODEL={model}\n{'#'*80}\n")

        needs_tuned_hparams = model in ("ClasswiseRisk", "DirectRisk")
        if needs_tuned_hparams:
            hparams = load_tuned_hparams(args.config_path, args.dataset, model, alpha=args.alpha)
            print(f"[run] using tuned hyperparameters for dataset={args.dataset} "
                  f"{'alpha=' + str(args.alpha) + ' ' if args.dataset == 'cp' else ''}"
                  f"model={model}: {hparams}")
        else:
            hparams = dict(
                epochs=args.epochs, optimizer="adam", lr=args.lr,
                dropout=args.risk_dropout, weight_decay=args.risk_weight_decay,
            )

        run_name = build_run_name(
            dataset=args.dataset, model=model,
            n_samples=args.n_samples, n_features=args.n_features, n_hidden=args.n_hidden,
            n_classes=args.n_classes, class_sep=args.class_sep,
            epochs=hparams["epochs"], batch_size=args.batch_size,
            risk_batch_size=args.risk_batch_size,
            lr=hparams["lr"], error_lr=args.error_lr,
            hidden_dim=args.hidden_dim, error_hidden_dim=args.error_hidden_dim,
            risk_hidden_dim=args.risk_hidden_dim,
            risk_dropout=hparams["dropout"], risk_weight_decay=hparams["weight_decay"],
            cp_feedback=args.cp_feedback, freeze_backbone=args.freeze_backbone,
        )
        if needs_tuned_hparams:
            run_name += "_tuned"

        csv_path = comparison_dir / f"{run_name}_multirun.csv"
        if csv_path.exists():
            csv_path.unlink()

        for seed in args.seeds:
            print(f"\n{'='*80}\nMODEL={model} SEED={seed} "
                  f"(dataset split fixed at seed={args.dataset_seed})\n{'='*80}\n")
            pl.seed_everything(seed, workers=True)

            if args.dataset == "cp":
                run_output = train_cp_one_seed(
                    dataset_bundle, model, hparams,
                    hidden_dim=args.hidden_dim, error_hidden_dim=args.error_hidden_dim,
                    risk_hidden_dim=args.risk_hidden_dim, risk_batch_size=args.risk_batch_size,
                    freeze_backbone=args.freeze_backbone,
                )
                save_trained_model(run_output.get("voi_est"), comparison_dir, run_name, seed)
                evaluate_image(
                    run_output, model=model, dataset=args.dataset, results_dir=args.results_dir,
                    random_state=seed, run_name=run_name, cp_feedback=args.cp_feedback,
                    alpha=args.alpha,
                )
            elif args.dataset == "email":
                run_output = train_tabular_tuned_one_seed(
                    dataset_bundle, model, hparams,
                    batch_size=args.batch_size, risk_batch_size=args.risk_batch_size,
                    hidden_dim=args.hidden_dim, error_hidden_dim=args.error_hidden_dim,
                    risk_hidden_dim=args.risk_hidden_dim,
                )
                save_trained_model(run_output.get("voi_est"), comparison_dir, run_name, seed)
                evaluate_tabular(
                    run_output, model=model, dataset=args.dataset, results_dir=args.results_dir,
                    random_state=seed, run_name=run_name,
                )
            else:
                run_output = train_synth_one_seed(
                    dataset_bundle, model, hparams,
                    hidden_dim=args.hidden_dim, error_hidden_dim=args.error_hidden_dim,
                    risk_hidden_dim=args.risk_hidden_dim,
                    risk_batch_size=args.risk_batch_size,
                )
                save_trained_model(run_output.get("voi_est"), comparison_dir, run_name, seed)
                evaluate_tabular(
                    run_output, model=model, dataset=args.dataset, results_dir=args.results_dir,
                    random_state=seed, run_name=run_name,
                )

        print(f"\nMODEL={model} COMPLETE - Ran {len(args.seeds)} seeds")

        if len(args.seeds) > 1:
            print("Computing aggregate statistics...")
            aggregate_multirun_stats(input_csv=csv_path, verbose=True)

            print("\nGenerating plots...")
            aggregated_csv = Path(csv_path).parent / f"{Path(csv_path).stem}_aggregated.csv"
            plot_multirun_results(aggregated_csv=aggregated_csv, show_ci=True)
            aggregated_csv_by_model[model] = aggregated_csv

    if len(aggregated_csv_by_model) > 1:
        print("\nGenerating combined model-comparison plot...")
        try:
            plot_model_comparison(
                aggregated_csv_by_model=aggregated_csv_by_model,
                combined_csv_path=comparison_dir / "comparison.csv",
                plot_output_path=comparison_dir / "comparison.pdf",
                dataset=args.dataset,
            )
        except Exception as e:
            # the per-model results and plots are already saved, so a failure
            # here (e.g. no LaTeX installation for text.usetex) is not fatal
            print(f"[run] WARNING: comparison plot failed ({e}); per-model results are unaffected.")

    print(f"\n{'#'*80}\nFINAL RUN COMPLETE - Ran {MODELS} for dataset={args.dataset}\n{'#'*80}\n")


if __name__ == "__main__":
    main()
