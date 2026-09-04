import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl


from pathlib import Path
from matplotlib.lines import Line2D
from sklearn.metrics import balanced_accuracy_score


def _fmt_float_for_name(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def alpha_dirname(alpha: float) -> str:
    """0.05 -> 'alpha05', 0.01 -> 'alpha01'. Keeps the image results (and the
    hyperparameters tuned for them) separated by conformal alpha."""
    return f"alpha{round(alpha * 100):02d}"


def build_run_name(
        dataset,
        model,
        n_samples,
        n_features,
        n_hidden,
        n_classes,
        class_sep,
        epochs,
        batch_size,
        risk_batch_size,
        lr,
        error_lr,
        hidden_dim,
        error_hidden_dim,
        risk_hidden_dim,
        risk_dropout,
        risk_weight_decay,
        cp_feedback,
        freeze_backbone,
):
    """Encode model, hyperparameters and dataset options into the file name
    prefix shared by every artifact of a run."""
    if model == "Random":
        parts = [f"res_{model}"]
    else:
        parts = [
            f"res_{model}",
            f"ep{epochs}",
            f"bs{batch_size}",
            f"rbs{risk_batch_size}",
            f"lr{_fmt_float_for_name(lr)}",
            f"elr{_fmt_float_for_name(error_lr)}",
            f"hd{hidden_dim}",
            f"ehd{error_hidden_dim}",
            f"rhd{risk_hidden_dim}",
            f"rdrop{_fmt_float_for_name(risk_dropout)}",
            f"rwd{_fmt_float_for_name(risk_weight_decay)}",
        ]

    if dataset == "synth":
        parts.extend([
            f"n{n_samples}",
            f"f{n_features}",
            f"h{n_hidden}",
            f"c{n_classes}",
            f"sep{_fmt_float_for_name(class_sep)}",
        ])

    if dataset == "cp":
        parts.append(f"{cp_feedback}")
        parts.append("freeze" if freeze_backbone else "nofreeze")

    return "_".join(parts)


@torch.no_grad()
def _collect_image_policy_outputs(voi_est, loader, model_type: str, device: torch.device):
    """Run one image loader through the estimator and collect everything the
    evaluation needs. For model_type='Random' there is no estimator at all
    (voi_est is None) and only the observed decisions are collected."""
    if voi_est is not None:
        voi_est.eval()
        voi_est.to(device)

    all_voi = []
    all_r0 = []
    all_r1 = []
    all_a0 = []
    all_a1 = []
    all_y = []
    all_ids = []
    all_pred_set_size = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].cpu()
        a0 = batch["a0"].cpu()
        a1 = batch["a1"].cpu()

        if model_type != "Random":
            all_voi.append(voi_est.voi(x).detach().cpu())
            all_r0.append(voi_est.conditional_risk(x, 0).detach().cpu())
            all_r1.append(voi_est.conditional_risk(x, 1).detach().cpu())

        all_y.append(y)
        all_a0.append(a0)
        all_a1.append(a1)

        if "id" in batch:
            all_ids.extend(list(batch["id"]))
        if "pred_set_size" in batch:
            all_pred_set_size.append(batch["pred_set_size"].cpu())

    out = {
        "y": torch.cat(all_y),
        "a0": torch.cat(all_a0),
        "a1": torch.cat(all_a1),
    }

    if model_type != "Random":
        out["voi"] = torch.cat(all_voi)
        out["r0"] = torch.cat(all_r0)
        out["r1"] = torch.cat(all_r1)

    if all_ids:
        out["ids"] = all_ids
    if all_pred_set_size:
        out["pred_set_size"] = torch.cat(all_pred_set_size)

    return out


def _score(y_true, preds, metric: str) -> float:
    if metric == "accuracy":
        return float(np.mean(preds == y_true))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, preds))
    raise ValueError(f"Unknown metric '{metric}'. Choose 'accuracy' or 'balanced_accuracy'.")


def _eval_coverage(
        voi_test,
        voi_cal,
        a0_test,
        a1_test,
        y_test,
        label,
        verbose,
        csv_path: str | Path | None = None,
        dataset: str = "unknown",
        model_type: str = "unknown",
        seed: int = 42,
        coverages=np.linspace(0, 1, 11),
        random_mode: bool = False,
        metric: str = "accuracy",
):
    """Budget-accuracy curve of the disclosure policy.

    At every budget level the threshold is the (1 - budget)-quantile of the VoI
    on the calibration set, and help is requested where the VoI is above it and
    strictly positive. Budget 0 is always the a_0 decision; budget 1 asks for
    help on every instance with positive VoI. With random_mode=True the policy
    is replaced by an independent Bernoulli(budget) draw per instance, which is
    the Random baseline. Returns {budget: score} and optionally appends the
    same rows to `csv_path`.
    """
    accs = {}
    n_total = len(y_test)
    rows = []

    if verbose:
        print(f"\n--- {label} (seed={seed}) ---")

    rng = np.random.default_rng(seed) if random_mode else None

    for cov in coverages:
        if cov == 0:
            acc = _score(y_test, a0_test, metric)
            thr = float("nan")
            n_sel = 0
            if verbose:
                print(f"Coverage: {cov:.2f}, Threshold: N/A, {metric} {acc:.4f}")

        elif cov == 1:
            if random_mode or voi_test is None:
                sel = np.ones(n_total, dtype=bool)
            else:
                sel = voi_test > 0

            preds  = np.where(sel, a1_test, a0_test)
            acc    = _score(y_test, preds, metric)
            thr    = 0.0
            n_sel  = int(sel.sum())
            if verbose:
                print(f"Coverage: {cov:.2f}, Threshold: 0 (voi>0), "
                    f"Selected: {n_sel}/{n_total}, {metric} {acc:.4f}")

        else:
            if random_mode:
                sel = rng.binomial(1, cov, size=n_total).astype(bool)
                preds = np.where(sel, a1_test, a0_test)
                thr = float("nan")
                n_sel = int(sel.sum())
                acc = _score(y_test, preds, metric)

                if verbose:
                    eff_cov = n_sel / n_total
                    print(f"Coverage target: {cov:.2f}, Empirical: {eff_cov:.3f}, "
                        f"Random selected: {n_sel}, {metric} {acc:.4f}")
            else:
                thr = float(np.quantile(voi_cal, 1 - cov))
                sel = (voi_test >= thr) & (voi_test > 0)
                preds = np.where(sel, a1_test, a0_test)
                acc = _score(y_test, preds, metric)
                n_sel = int(sel.sum())
                if verbose:
                    print(f"Coverage: {cov:.2f}, Threshold: {thr:.6f}, "
                          f"Selected: {n_sel}, {metric} {acc:.4f}")

        accs[cov] = acc
        rows.append({
            "dataset": dataset,
            "model_type": model_type,
            "seed": seed,
            "coverage": cov,
            "threshold": thr,
            "n_selected": n_sel,
            "n_total": n_total,
            "accuracy": acc,
        })

    if csv_path is not None:
        _append_rows_csv(csv_path, rows)

    return accs


def build_final_results_csv(
        voi_test,
        voi_cal,
        a0_test,
        a1_test,
        y_test,
        ids_test,
        csv_path: str | Path,
        dataset: str,
        model_type: str,
        seed: int,
        coverages=np.linspace(0, 1, 11),
        pred_set_size_test=None,
):
    """
    Per-instance decision log on the test set. For every instance and every
    budget level it records whether the calibrated policy asked for human help
    (same threshold as `_eval_coverage`), the ground truth, the prediction
    actually used and, for the image scenario, the conformal prediction-set
    size of that instance.
    """
    n_total = len(y_test)
    rows = []

    for cov in coverages:
        if cov == 0:
            sel = np.zeros(n_total, dtype=bool)
            thr = float("nan")
        elif cov == 1:
            sel = voi_test > 0
            thr = 0.0
        else:
            thr = float(np.quantile(voi_cal, 1 - cov))
            sel = (voi_test >= thr) & (voi_test > 0)

        preds = np.where(sel, a1_test, a0_test)

        for i in range(n_total):
            rows.append({
                "dataset": dataset,
                "model_type": model_type,
                "seed": seed,
                "id": ids_test[i],
                "coverage": cov,
                "threshold": thr,
                "asked_for_help": bool(sel[i]),
                "groundtruth": y_test[i],
                "prediction": preds[i],
                "pred_set_size": (
                    int(pred_set_size_test[i]) if pred_set_size_test is not None else np.nan
                ),
            })

    _append_rows_csv(csv_path, rows)


def _append_rows_csv(csv_path: str | Path, rows: list[dict]) -> None:
    """Append rows to a CSV, writing the header if the file does not exist."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    fieldnames = list(rows[0].keys())
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def aggregate_multirun_stats(
        input_csv: str | Path,
        output_csv: str | Path | None = None,
        verbose: bool = True,
):
    """Aggregate the per-seed budget-accuracy rows of one model into mean, std
    and 95% confidence interval per budget level, and write them next to the
    input CSV as *_aggregated.csv."""
    input_path = Path(input_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_csv}")

    df = pd.read_csv(input_path)

    if "seed" not in df.columns:
        raise ValueError(f"{input_csv} has no 'seed' column; it is not a multirun output.")

    group_cols = ["dataset", "model_type", "coverage"]

    missing_cols = [c for c in group_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns in the CSV: {missing_cols}. Found: {df.columns.tolist()}"
        )

    numeric_cols = ["seed", "coverage", "threshold", "n_selected", "n_total", "accuracy"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if verbose:
        print("\nColumn dtypes after conversion:")
        print(df.dtypes)

    df = df.dropna(subset=["coverage", "accuracy"])

    stats = (
        df.groupby(group_cols, dropna=False)["accuracy"]
        .agg(
            mean="mean",
            std="std",
            min="min",
            max="max",
            count="count",
        )
        .reset_index()
    )

    threshold_stats = (
        df.groupby(group_cols, dropna=False)[["threshold", "n_selected", "n_total"]]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={
            "threshold": "threshold_mean",
            "n_selected": "n_selected_mean",
        })
    )

    stats = stats.merge(threshold_stats, on=group_cols, how="left")
    stats["ci_95"] = 1.96 * stats["std"] / np.sqrt(stats["count"])

    stats = stats[
        [
            "dataset", "model_type", "coverage",
            "mean", "std", "ci_95", "min", "max", "count",
            "threshold_mean", "n_selected_mean", "n_total"
        ]
    ]

    if output_csv is None:
        output_csv = input_path.parent / f"{input_path.stem}_aggregated.csv"
    else:
        output_csv = Path(output_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_csv, index=False)

    if verbose:
        print(f"\n{'='*80}")
        print("AGGREGATE STATISTICS OVER MULTIPLE SEEDS")
        print(f"{'='*80}")
        print(f"Input: {input_csv}")
        print(f"Output: {output_csv}")
        print(f"Number of seeds: {df['seed'].nunique()}")
        print(f"Seeds: {sorted(df['seed'].dropna().unique())}")
        print(f"{'='*80}\n")

        print("Coverage | Mean Acc | Std | CI 95% | Min | Max | N Seeds")
        print("-" * 80)
        for _, row in stats.iterrows():
            print(
                f"{row['coverage']:>8.2f} | "
                f"{row['mean']:>8.4f} | "
                f"{row['std']:>6.4f} | "
                f"±{row['ci_95']:>6.4f} | "
                f"{row['min']:>6.4f} | "
                f"{row['max']:>6.4f} | "
                f"{int(row['count']):>7}"
            )
        print(f"\n{'='*80}\n")

    return stats


def plot_multirun_results(
        aggregated_csv: str | Path,
        output_path: str | Path = None,
        show_ci: bool = True,
        figsize: tuple = (12, 8),
):
    """Budget-accuracy curve of a single model, with the 95% confidence band
    across seeds."""
    df = pd.read_csv(aggregated_csv)

    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'font.size': 14
    })

    sns.set(
        font_scale=1.2,
        style="ticks",
        rc={
            "text.usetex": False,
            'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amsmath} \usepackage{bm}',
            "font.family": "serif"
        }
    )
    sns.color_palette("colorblind")

    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=df,
        x="coverage",
        y="mean",
        marker='o',
        markersize=8,
        linewidth=2.5,
        label=f"{df['model_type'].iloc[0]} (n={int(df['count'].iloc[0])} seeds)",
        ax=ax
    )

    if show_ci:
        ax.fill_between(
            df["coverage"],
            df["mean"] - df["ci_95"],
            df["mean"] + df["ci_95"],
            alpha=0.2,
            label=r"95\% CI"
        )

    ax.set_xlabel(r"Coverage", fontsize=16)
    ax.set_ylabel(r"Accuracy", fontsize=16)
    ax.set_title(r"Coverage-Accuracy Curve", fontsize=18, pad=15)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', frameon=True, shadow=True)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(df["min"].min() - 0.02, df["max"].max() + 0.02)

    plt.tight_layout()

    if output_path is None:
        aggregated_path = Path(aggregated_csv)
        output_path = aggregated_path.parent / f"{aggregated_path.stem}.pdf"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.close()


_COMPARISON_MODEL_ORDER = ["DirectRisk", "ClasswiseRisk", "Random"]
_COMPARISON_MARKERS = {"DirectRisk": "s", "ClasswiseRisk": "^", "Random": "D"}
_COMPARISON_COLORBLIND_PALETTE = sns.color_palette("colorblind")
_COMPARISON_COLORS = {
    "DirectRisk": _COMPARISON_COLORBLIND_PALETTE[1],
    "ClasswiseRisk": _COMPARISON_COLORBLIND_PALETTE[2],
    "Random": _COMPARISON_COLORBLIND_PALETTE[3],
}


def _tex_tt(s: str) -> str:
    escaped = str(s).replace("_", r"\_")
    return rf"\texttt{{{escaped}}}"


def plot_model_comparison(
        aggregated_csv_by_model: dict,
        combined_csv_path: str | Path,
        plot_output_path: str | Path,
        dataset: str = "",
        figsize: tuple = (12, 8),
):
    """
    Merge the per-model aggregated CSVs into a single budget-accuracy figure,
    plus the combined CSV behind it.

    aggregated_csv_by_model: {model_type: path}. Missing models are skipped.
    The human_0/human_1 reference lines are drawn from the Random rows at
    budget 0 (always a_0) and budget 1 (always a_1).
    """
    dfs = []
    for model_type, csv_path in aggregated_csv_by_model.items():
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"[plot_model_comparison] skipping {model_type}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df["model_type"] = model_type
        dfs.append(df)

    if not dfs:
        print("[plot_model_comparison] no aggregated csvs found, skipping comparison plot")
        return

    combined_df = pd.concat(dfs, ignore_index=True)

    combined_csv_path = Path(combined_csv_path)
    combined_csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"[plot_model_comparison] saved combined csv to {combined_csv_path}")

    df = combined_df
    err_col = "std" if "std" in df.columns else ("ci_95" if "ci_95" in df.columns else None)

    human_0 = human_1 = None
    df_random = df[df["model_type"] == "Random"]
    if not df_random.empty:
        if (df_random["coverage"] == 0).any():
            human_0 = df_random.loc[df_random["coverage"] == 0, "mean"].mean()
        if (df_random["coverage"] == 1).any():
            human_1 = df_random.loc[df_random["coverage"] == 1, "mean"].mean()

    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "axes.titlesize": 46, "axes.labelsize": 40,
        "xtick.labelsize": 34, "ytick.labelsize": 34,
        "legend.fontsize": 34, "font.size": 34,
        "text.usetex": True, "font.family": "serif",
    })
    sns.set_theme(style="ticks", palette="colorblind")

    fig, ax = plt.subplots(figsize=figsize)

    for model_type in _COMPARISON_MODEL_ORDER:
        g = df[df["model_type"] == model_type].copy()
        if g.empty:
            continue
        g = g.sort_values("coverage")

        sns.lineplot(
            data=g, x="coverage", y="mean", ax=ax,
            label=_tex_tt(model_type),
            marker=_COMPARISON_MARKERS.get(model_type, "o"),
            markersize=12, linewidth=3.0,
            color=_COMPARISON_COLORS[model_type],
            errorbar=None,
        )
        if err_col is not None:
            ax.fill_between(
                g["coverage"], g["mean"] - g[err_col], g["mean"] + g[err_col],
                alpha=0.18, color=_COMPARISON_COLORS[model_type],
            )

    if human_0 is not None:
        ax.axhline(y=human_0, color="black", linestyle=":", linewidth=1.5, label=_tex_tt("human_0"))
    if human_1 is not None:
        ax.axhline(y=human_1, color="black", linestyle="--", linewidth=1.5, label=_tex_tt("human_1"))

    title = "Budget-Accuracy Curve"
    if dataset:
        title += f" ({dataset})"

    ax.set_xlabel("Budget", fontsize=30)
    ax.set_ylabel("Accuracy", fontsize=30)
    ax.set_title(title, fontsize=32)
    ax.tick_params(axis="both", which="major", labelsize=28)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(-0.05, 1.05)

    # matplotlib fills an ncol=2 legend column by column, so the two groups are
    # padded to the same length to keep them in their own column
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))

    left_labels = [_tex_tt("human_0"), _tex_tt("human_1")]
    right_labels = [_tex_tt(m) for m in _COMPARISON_MODEL_ORDER]

    left_items = [(lab, label_to_handle[lab]) for lab in left_labels if lab in label_to_handle]
    right_items = [(lab, label_to_handle[lab]) for lab in right_labels if lab in label_to_handle]

    n_rows = max(len(left_items), len(right_items))
    blank_handle = Line2D([], [], linestyle="none", color="none")

    def _pad(items, n):
        items = list(items)
        items += [("", blank_handle)] * (n - len(items))
        return items

    left_items = _pad(left_items, n_rows)
    right_items = _pad(right_items, n_rows)

    ax.legend(
        [h for _, h in left_items] + [h for _, h in right_items],
        [lab for lab, _ in left_items] + [lab for lab, _ in right_items],
        loc="lower right", handlelength=2.5, fontsize=24, ncol=2, columnspacing=1.0,
    )

    plt.tight_layout()

    plot_output_path = Path(plot_output_path)
    plot_output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_output_path, bbox_inches="tight")
    plt.close()
    print(f"[plot_model_comparison] saved comparison plot to {plot_output_path}")


class PrintMetricsCallback(pl.Callback):
    """
    Periodically print `trainer.callback_metrics`.

    stage='val' hooks the end of the validation epoch, for runs that have a
    validation split (hyperparameter tuning); stage='train' hooks the end of
    the training epoch, for the final runs that train without validation.
    """

    def __init__(self, name="model", every_n_epochs=1, stage="val"):
        super().__init__()
        assert stage in ("val", "train")
        self.name = name
        self.every_n_epochs = every_n_epochs
        self.stage = stage

    def _fmt(self, x):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    def _print(self, trainer):
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs != 0:
            return

        metrics = trainer.callback_metrics

        if not metrics:
            print(f"[{self.name}] epoch {epoch:03d} | no metrics in callback_metrics")
            return

        parts = [f"{k}={self._fmt(v)}" for k, v in metrics.items()]
        print(f"[{self.name}] epoch {epoch:03d} | " + " | ".join(parts))

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.stage == "val":
            self._print(trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.stage == "train":
            self._print(trainer)
