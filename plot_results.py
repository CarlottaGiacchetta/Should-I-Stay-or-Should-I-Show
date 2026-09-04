import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


MODEL_ORDER = ["DirectRisk", "ClasswiseRisk", "Random"]

MARKERS = {
    "DirectRisk": "s",
    "ClasswiseRisk": "^",
    "Random": "D",
}

# fixed colour per model name, so a curve keeps its colour whichever models
# happen to be present in a figure
_COLORBLIND_PALETTE = sns.color_palette("colorblind")
COLORS = {
    "DirectRisk": _COLORBLIND_PALETTE[1],
    "ClasswiseRisk": _COLORBLIND_PALETTE[2],
    "Random": "#8e44ad",
}

# Layout on disk, matching run.py's output:
#   - cp:    <results_dir>/cp/<cp_feedback>/alpha{XX}/
#   - email: <results_dir>/email/
#   - synth: <results_dir>/synth/<task>/
# run.py already writes comparison.csv/comparison.pdf itself; this script
# exists to redo just the plot (e.g. after changing the styling, or when
# run.py's own comparison plot failed for lack of a LaTeX installation) from
# the per-model *_aggregated.csv files, whose names embed every hyperparameter
# and are therefore matched by glob rather than hardcoded.

DEFAULT_RESULTS_DIR = Path("results")
MODELS = ("DirectRisk", "ClasswiseRisk", "Random")


def _alpha_dirname(alpha: float) -> str:
    """0.05 -> 'alpha05', 0.01 -> 'alpha01' (matches run.py's alpha_dirname)."""
    return f"alpha{round(alpha * 100):02d}"


def find_aggregated_csv(directory: Path, model_type: str) -> Path:
    """Locate the aggregated results csv of `model_type` inside `directory`.

    Accepts both the raw pipeline output (res_<Model>_<hparams>_aggregated.csv)
    and a manually simplified <Model>.csv. When several files match, the most
    recently modified one wins.
    """
    candidates = sorted(directory.glob(f"res_{model_type}_*_aggregated.csv"))
    if candidates:
        if len(candidates) > 1:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            print(
                f"[warn] {len(candidates)} aggregated files found for {model_type} "
                f"in {directory}, using most recent: {candidates[0].name}"
            )
        return candidates[0]

    simple = directory / f"{model_type}.csv"
    if simple.exists():
        return simple

    raise FileNotFoundError(
        f"No aggregated csv or {model_type}.csv found for {model_type} in {directory}"
    )


def resolve_model_paths(base_dir: Path, models) -> dict[str, Path]:
    """Aggregated csv of every model in `models`, looked up inside `base_dir`."""
    return {model_type: find_aggregated_csv(base_dir, model_type) for model_type in models}


def build_cp_paths(
    alpha: float, cp_feedback: str = "lenient", results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> tuple[Path, Path, dict[str, Path]]:
    base_dir = Path(results_dir) / "cp" / cp_feedback / _alpha_dirname(alpha)
    combined_csv_path = base_dir / "comparison.csv"
    plot_output_path = base_dir / "comparison.pdf"

    def _model_paths():
        if not base_dir.is_dir():
            raise FileNotFoundError(f"No results directory for alpha={alpha}: {base_dir}")
        return resolve_model_paths(base_dir, MODELS)

    return combined_csv_path, plot_output_path, _model_paths


def build_email_paths(results_dir: str | Path = DEFAULT_RESULTS_DIR) -> tuple[Path, Path, dict[str, Path]]:
    base_dir = Path(results_dir) / "email"
    combined_csv_path = base_dir / "comparison.csv"
    plot_output_path = base_dir / "comparison.pdf"

    def _model_paths():
        if not base_dir.is_dir():
            raise FileNotFoundError(f"No results directory for email: {base_dir}")
        return resolve_model_paths(base_dir, MODELS)

    return combined_csv_path, plot_output_path, _model_paths


def build_synth_paths(
    task: str, results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> tuple[Path, Path, dict[str, Path]]:
    synth_dir = Path(results_dir) / "synth"
    base_dir = synth_dir / task
    combined_csv_path = synth_dir / f"combined_aggregated_{task}.csv"
    plot_output_path = synth_dir / f"combined_aggregated_{task}.pdf"

    def _model_paths():
        if not base_dir.is_dir():
            raise FileNotFoundError(f"No results directory for synth task={task}: {base_dir}")
        return resolve_model_paths(base_dir, MODELS)

    return combined_csv_path, plot_output_path, _model_paths


def tex_tt(s: str) -> str:
    s = str(s).replace("_", r"\_")
    return rf"\texttt{{{s}}}"


def load_model_csvs_from_paths(
    model_paths: dict[str, str | Path],
    task: str | None = None,
) -> pd.DataFrame:
    """Concatenate the per-model aggregated csvs into a single dataframe."""
    dfs = []

    for model_type, csv_path in model_paths.items():
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing file for {model_type}: {csv_path}")

        df = pd.read_csv(csv_path)
        df["model_type"] = model_type
        df["task"] = task

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# panels of the 2x2 --dataset all figure, in reading order
ALL_PANELS = ("synth_bin", "synth_multi", "email", "cp")
PANEL_TITLES = {
    "synth_bin": "(a) Synthetic, Binary",
    "synth_multi": "(b) Synthetic, Multiclass",
    "cp": "(d) ImageNet-16H",
    "email": "(c) Email",
}

# groups of panels sharing one y range; panels not listed keep their own
# autoscaled range, which is the default since each scenario reaches a
# different accuracy level
SHARED_Y_GROUPS: tuple[tuple[str, ...], ...] = ()

# LEGEND_NCOL = None puts every entry of the shared legend on one row
LEGEND_FONTSIZE = 34
LEGEND_NCOL = None

TITLE_FONTSIZE = 46
LABEL_FONTSIZE = 40


def _apply_style():
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.titlesize": 46,
        "axes.labelsize": 40,
        "xtick.labelsize": 34,
        "ytick.labelsize": 34,
        "legend.fontsize": 34,
        "font.size": 34,
        "text.usetex": True,
        "font.family": "serif",
    })
    sns.set_theme(style="ticks", palette="colorblind")


def _err_col(df: pd.DataFrame) -> str | None:
    if "std" in df.columns:
        return "std"
    if "ci_95" in df.columns:
        return "ci_95"
    return None


def _human_levels(df: pd.DataFrame) -> tuple[float | None, float | None]:
    """human_0/human_1: Random's accuracy at budget 0 (never ask) / 1 (always ask)."""
    df_random = df[df["model_type"] == "Random"]
    human_0 = human_1 = None
    if not df_random.empty:
        if (df_random["coverage"] == 0).any():
            human_0 = df_random.loc[df_random["coverage"] == 0, "mean"].mean()
        if (df_random["coverage"] == 1).any():
            human_1 = df_random.loc[df_random["coverage"] == 1, "mean"].mean()
    return human_0, human_1


def _draw_panel(ax, df: pd.DataFrame) -> dict:
    """Draw every model curve + the human_0/human_1 baselines onto `ax`.

    Returns {legend label: handle} and leaves the axes without a legend, so the
    caller decides where it goes: per-axes in _plot_, one shared figure-level
    legend in _plot_all_.
    """
    err_col = _err_col(df)

    for model_type in MODEL_ORDER:
        g = df[df["model_type"] == model_type].copy()
        if g.empty:
            continue
        g = g.sort_values("coverage")

        sns.lineplot(
            data=g,
            x="coverage",
            y="mean",
            ax=ax,
            label=tex_tt(model_type),
            marker=MARKERS.get(model_type, "o"),
            markersize=12,
            linewidth=3.0,
            color=COLORS[model_type],
            errorbar=None,
        )

        if err_col is not None:
            ax.fill_between(
                g["coverage"],
                g["mean"] - g[err_col],
                g["mean"] + g[err_col],
                alpha=0.18,
                color=COLORS[model_type],
            )

    human_0, human_1 = _human_levels(df)
    if human_0 is not None:
        ax.axhline(y=human_0, color="black", linestyle=":", linewidth=2.5,
                   label=tex_tt("human_0"))
    if human_1 is not None:
        ax.axhline(y=human_1, color="black", linestyle="--", linewidth=2.5,
                   label=tex_tt("human_1"))

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(-0.05, 1.05)

    handles, labels = ax.get_legend_handles_labels()
    # seaborn attaches a legend to the axes as soon as `label=` is passed
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    return dict(zip(labels, handles))


def _legend_two_columns(label_to_handle: dict) -> tuple[list, list]:
    """Left column human_0/human_1/Random, right column the other models.

    matplotlib's ncol=2 legend fills column-major (the first ceil(n/2) handles
    go in column 1, the rest in column 2), so a plain concatenation only lands
    on this exact split when both groups are the same size. Pad the shorter
    group with invisible placeholders so the two groups always occupy exactly
    the left/right columns, regardless of how many models are in the data.
    """
    left_labels = [tex_tt("human_0"), tex_tt("human_1"), tex_tt("Random")]
    right_labels = [tex_tt(m) for m in MODEL_ORDER if m != "Random"]

    left_items = [(lab, label_to_handle[lab]) for lab in left_labels if lab in label_to_handle]
    right_items = [(lab, label_to_handle[lab]) for lab in right_labels if lab in label_to_handle]

    n_rows = max(len(left_items), len(right_items))
    blank_handle = Line2D([], [], linestyle="none", color="none")

    def _pad(items: list, n: int) -> list:
        items = list(items)
        items += [("", blank_handle)] * (n - len(items))
        return items

    left_items = _pad(left_items, n_rows)
    right_items = _pad(right_items, n_rows)

    return (
        [h for _, h in left_items] + [h for _, h in right_items],
        [lab for lab, _ in left_items] + [lab for lab, _ in right_items],
    )


def _legend_flat(label_to_handle: dict) -> tuple[list, list]:
    """Single row-major legend (used for the shared --dataset all legend)."""
    order = [tex_tt("human_0"), tex_tt("human_1")] + [tex_tt(m) for m in MODEL_ORDER]
    items = [(lab, label_to_handle[lab]) for lab in order if lab in label_to_handle]
    return [h for _, h in items], [lab for lab, _ in items]


def _save(fig, output_path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_path}")


def _plot_(aggregated_csv, output_path=None, figsize=(12, 8), legend_loc="lower right"):
    df = pd.read_csv(aggregated_csv)

    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    label_to_handle = _draw_panel(ax, df)

    task = df["task"].iloc[0] if "task" in df.columns and not df.empty else None
    title = "Budget-Accuracy Curve"
    if task == "bin":
        title += " (Binary)"
    elif task == "multi":
        title += " (Multiclass)"

    ax.set_xlabel("Budget", fontsize=30)
    ax.set_ylabel("Accuracy", fontsize=30)
    ax.set_title(title, fontsize=32)
    ax.tick_params(axis="both", which="major", labelsize=28)

    handles, labels = _legend_two_columns(label_to_handle)
    ax.legend(
        handles,
        labels,
        loc=legend_loc,
        handlelength=2.5,
        fontsize=24,
        ncol=2,
        columnspacing=1.0,
    )

    fig.tight_layout()

    if output_path is None:
        output_path = Path(aggregated_csv).with_suffix(".pdf")
    _save(fig, output_path)


def _plot_all_(panel_dfs: dict, output_path, figsize=(22, 16)):
    """2x2 grid -- synth bin / synth multi on top, cp / email below -- with one
    figure title, one pair of axis labels and one legend for the whole figure.

    Each panel keeps its own y range (accuracy levels differ a lot between the
    synthetic and the real datasets) but shares the x axis, which is a budget
    in [0, 1] everywhere. Panels with no results are left blank.
    """
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)

    label_to_handle: dict = {}
    for ax, key in zip(axes.flat, ALL_PANELS):
        df = panel_dfs.get(key)
        if df is None or df.empty:
            ax.set_axis_off()
            continue
        label_to_handle.update(_draw_panel(ax, df))
        ax.set_title(PANEL_TITLES[key], fontsize=36)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", which="major", labelsize=30)

    # one shared y range per SHARED_Y_GROUPS group, as the union of the ranges
    # matplotlib autoscaled per panel
    ax_by_key = {key: ax for ax, key in zip(axes.flat, ALL_PANELS)}
    for group in SHARED_Y_GROUPS:
        group_axes = [ax_by_key[key] for key in group
                      if key in ax_by_key and key in panel_dfs and ax_by_key[key].get_visible()]
        if len(group_axes) < 2:
            continue
        lows, highs = zip(*(ax.get_ylim() for ax in group_axes))
        for ax in group_axes:
            ax.set_ylim(min(lows), max(highs))

    handles, labels = _legend_flat(label_to_handle)
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=LEGEND_NCOL or max(len(labels), 1),
        handlelength=1.6,
        handletextpad=0.4,
        markerscale=1.3,
        fontsize=LEGEND_FONTSIZE,
        columnspacing=0.8,
        borderpad=0.6,
        borderaxespad=0.0,
    )

    # the bottom margin is stacked from the legend's measured top edge rather
    # than from a fixed figure fraction, since the legend keeps its size in
    # points while the figure size changes
    fig.canvas.draw()
    legend_top = legend.get_window_extent().transformed(fig.transFigure.inverted()).y1
    xlabel_h = (LABEL_FONTSIZE / 72) / fig.get_figheight()
    xlabel_y = legend_top + 0.45 * xlabel_h
    bottom = xlabel_y + 1.6 * xlabel_h

    fig.tight_layout(rect=(0.04, bottom, 1.0, 0.96))

    # centre the shared labels on the panels rather than on the figure: the
    # tight_layout rect and the "Accuracy" label eat into the left margin
    drawn = [ax for ax in axes.flat if ax.axison]
    boxes = [ax.get_position() for ax in (drawn or list(axes.flat))]
    panel_x = (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2
    panel_y = (min(b.y0 for b in boxes) + max(b.y1 for b in boxes)) / 2
    fig.supxlabel("Budget", fontsize=LABEL_FONTSIZE, x=panel_x, y=xlabel_y)
    fig.supylabel("Accuracy", fontsize=LABEL_FONTSIZE, x=0.005, y=panel_y)
    legend.set_bbox_to_anchor((panel_x, 0.005), transform=fig.transFigure)

    _save(fig, output_path)


def build_and_plot_from_paths(
    model_paths: dict[str, str | Path],
    combined_csv_path: str | Path,
    plot_output_path: str | Path | None = None,
    task: str | None = None,
    legend_loc: str = "lower right",
):
    combined_df = load_model_csvs_from_paths(model_paths, task=task)

    combined_csv_path = Path(combined_csv_path)
    combined_csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Saved combined csv to {combined_csv_path}")

    _plot_(combined_csv_path, output_path=plot_output_path, legend_loc=legend_loc)


def parse_args():
    parser = argparse.ArgumentParser(
        description="(Re)plot the DirectRisk/ClasswiseRisk/Random budget-accuracy "
                     "comparison for a run.py output directory. Reuses comparison.csv "
                     "if run.py already wrote one there; otherwise rebuilds it from "
                     "the per-model *_aggregated.csv files."
    )
    parser.add_argument(
        "--dataset", choices=["cp", "email", "synth", "all"], default="cp",
        help="Which scenario to plot. Paths are auto-discovered on disk "
             "(<results_dir>/cp/<cp_feedback>/alpha{XX}, <results_dir>/email, "
             "or <results_dir>/synth/<task>) — no need to pass file paths manually. "
             "'all' draws a single 2x2 figure (synth bin | synth multi on top, "
             "cp | email below) with one shared title, axis labels and legend.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="CP alpha level (e.g. 0.01, 0.02, 0.05). Ignored for --dataset email/synth.",
    )
    parser.add_argument(
        "--cp_feedback", choices=["lenient", "strict"], default="lenient",
        help="Ignored for --dataset email/synth.",
    )
    parser.add_argument(
        "--task", choices=["bin", "multi"], default="bin",
        help="Synth task variant (binary or multiclass). Ignored for --dataset cp/email.",
    )
    parser.add_argument(
        "--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR),
        help="Base results dir, same one passed as --results_dir to run.py.",
    )
    parser.add_argument(
        "--skip-missing", action="store_true",
        help="--dataset all only: draw the panels that do have results instead of "
             "failing when some are missing (the others are left blank).",
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Rebuild comparison.csv from the per-model *_aggregated.csv even if "
             "comparison.csv already exists.",
    )
    return parser.parse_args()


def build_paths_for(dataset: str, args, task: str | None = None):
    """(combined csv, plot pdf, model-paths callable, task) for one scenario."""
    if dataset == "cp":
        combined, plot_path, get_paths = build_cp_paths(
            alpha=args.alpha, cp_feedback=args.cp_feedback, results_dir=args.results_dir,
        )
        return combined, plot_path, get_paths, None
    if dataset == "email":
        combined, plot_path, get_paths = build_email_paths(results_dir=args.results_dir)
        return combined, plot_path, get_paths, None
    combined, plot_path, get_paths = build_synth_paths(task=task, results_dir=args.results_dir)
    return combined, plot_path, get_paths, task


def load_or_build_combined(combined_csv_path, get_model_paths, task, force_rebuild):
    """Read the already-combined csv, or rebuild (and cache) it from the
    per-model *_aggregated.csv files."""
    combined_csv_path = Path(combined_csv_path)
    if combined_csv_path.exists() and not force_rebuild:
        return pd.read_csv(combined_csv_path)

    combined_df = load_model_csvs_from_paths(get_model_paths(), task=task)
    combined_csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Saved combined csv to {combined_csv_path}")
    return combined_df


def plot_all(args) -> None:
    """--dataset all: one 2x2 figure out of the four scenarios."""
    specs = [
        ("synth_bin", "synth", "bin"),
        ("synth_multi", "synth", "multi"),
        ("cp", "cp", None),
        ("email", "email", None),
    ]

    panel_dfs, missing = {}, []
    for key, dataset, task in specs:
        combined_csv_path, _, get_paths, task = build_paths_for(dataset, args, task)
        try:
            panel_dfs[key] = load_or_build_combined(
                combined_csv_path, get_paths, task, args.force_rebuild,
            )
        except FileNotFoundError as e:
            missing.append(f"  - {PANEL_TITLES[key]}: {e}")

    if missing:
        report = "\n".join(missing)
        if not args.skip_missing:
            raise SystemExit(
                "Cannot build the 2x2 figure, these panels have no results yet:\n"
                f"{report}\n"
                "Run run.py for them, or pass --skip-missing to plot only the rest."
            )
        print(f"[warn] leaving these panels blank (--skip-missing):\n{report}")

    _plot_all_(panel_dfs, output_path=Path(args.results_dir) / "comparison_all.pdf")


if __name__ == "__main__":
    args = parse_args()

    if args.dataset == "all":
        plot_all(args)
        raise SystemExit(0)

    legend_loc = "lower left" if args.dataset == "email" else "lower right"
    combined_csv_path, plot_output_path, get_model_paths, task = build_paths_for(
        args.dataset, args, task=args.task,
    )

    if combined_csv_path.exists() and not args.force_rebuild:
        print(f"Found existing {combined_csv_path}, plotting directly from it "
              f"(pass --force-rebuild to recombine from the per-model csvs instead).")
        _plot_(combined_csv_path, output_path=plot_output_path, legend_loc=legend_loc)
    else:
        build_and_plot_from_paths(
            model_paths=get_model_paths(),
            combined_csv_path=combined_csv_path,
            plot_output_path=plot_output_path,
            task=task,
            legend_loc=legend_loc,
        )
