#!/usr/bin/env python3
"""Plot evaluation curves from the logs written by main.py, averaged over seeds.

main.py stores every run at

    ~/all_logs/<save_dir>/offline/<agent_name>/<env_name>/<cfg_name>/seed<seed>/

and writes the evaluation metrics to ``eval.csv`` there (columns are
``step`` plus ``evaluation/<metric>``).  This script reads all seeds of one or
more (agent, cfg_name) combinations for a given environment, aligns the curves
on a common step grid, smooths each seed with an exponential moving average,
and plots the mean with a std/sem band.

Examples
--------
Plot the default run for one environment:

    python plot.py --env_name antmaze-medium-play-v2

Compare several configs of the same agent:

    python plot.py --env_name antmaze-medium-play-v2 \
        --agent driftql --cfg default/ temp0.1/ --smooth 0.9

Compare several agents and choose the metric explicitly:

    python plot.py --env_name antmaze-medium-play-v2 \
        --agent driftql dql --metric evaluation/success

Discover what is available under ~/all_logs:

    python plot.py --env_name antmaze-medium-play-v2 --list

    
Usage
---------    
# 最简：自动发现该环境下所有实验并画 success 曲线
python plot.py --env_name antmaze-medium-play-v2

# 对比多个 config，平滑更强
python plot.py --env_name antmaze-medium-play-v2 \
    --agent driftql --cfg default/ temp0.1/ --smooth 0.9

# 指定指标、种子、输出与矢量图
python plot.py --env_name antmaze-medium-play-v2 \
    --metric evaluation/success evaluation/episode.return \
    --seeds 0 1 2 --also_pdf --out plots/antmaze_compare.png

# 先看看现有日志里有什么
python plot.py --env_name antmaze-medium-play-v2 --list
"""

import argparse
import csv
import glob
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #

# Categorical palette (validated: lightness band, chroma, adjacent CVD, and
# normal-vision separation all pass on the light surface).  Hues are assigned
# in this fixed order and never cycled.
PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e6e4df"
AXIS_COLOR = "#c9c7c0"

DEFAULT_ROOT = os.path.expanduser("~/all_logs")
DEFAULT_SAVE_DIR = "drift_logs"
DEFAULT_ENV = "antmaze-medium-play-v2"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Metrics that make sense as a primary y-axis, in order of preference.
DEFAULT_METRIC_PRIORITY = [
    "evaluation/success",
    "evaluation/episode.normalized_return",
    "evaluation/episode.return",
    "evaluation/normalized_return",
    "evaluation/return",
]


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot seed-averaged evaluation curves from main.py logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Where the runs live (mirrors main.py's path construction).
    p.add_argument("--root", default=DEFAULT_ROOT, help="Root that contains all log folders.")
    p.add_argument("--save_dir", default=DEFAULT_SAVE_DIR, help="Value of --save_dir used in main.py.")
    p.add_argument("--env_name", default=DEFAULT_ENV, help="Environment (dataset) name.")
    p.add_argument(
        "--agent",
        nargs="+",
        default=None,
        help="Agent directory name(s), e.g. driftql. Default: all agents found for this env.",
    )
    p.add_argument(
        "--cfg",
        nargs="+",
        default=None,
        help="Config directory name(s), e.g. default/ temp0.1/. Default: all found for each agent.",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Only use these seed indices. Default: all seed* directories.",
    )
    p.add_argument("--labels", nargs="+", default=None, help="Legend labels, one per (agent, cfg) group.")

    # What to read and how to aggregate.
    p.add_argument(
        "--metric",
        "-m",
        nargs="+",
        default=None,
        help="Metric column(s), e.g. evaluation/success evaluation/episode.return. "
        "Multiple metrics become subplots. Default: auto-detect.",
    )
    p.add_argument(
        "--smooth",
        type=float,
        default=0.85,
        help="EMA smoothing weight in [0, 1). Higher is smoother; 0 disables smoothing.",
    )
    p.add_argument(
        "--band",
        choices=["std", "sem", "none"],
        default="std",
        help="Shaded uncertainty band across seeds.",
    )
    p.add_argument(
        "--min_seeds",
        type=int,
        default=1,
        help="Only draw steps where at least this many seeds have data.",
    )

    # Figure cosmetics.
    p.add_argument("--title", default=None, help="Figure title. Default: the environment name.")
    p.add_argument("--xlabel", default="Environment steps", help="X-axis label.")
    p.add_argument("--xmax", type=float, default=None, help="Clip the x-axis at this step.")
    p.add_argument("--ymin", type=float, default=None, help="Lower y-axis limit.")
    p.add_argument("--ymax", type=float, default=None, help="Upper y-axis limit.")
    p.add_argument("--logy", action="store_true", help="Use a log scale on the y-axis.")
    p.add_argument(
        "--direct_labels",
        choices=["auto", "on", "off"],
        default="auto",
        help="Label each line at its right end. auto = on for 2-4 series.",
    )
    p.add_argument("--width", type=float, default=7.0, help="Figure width in inches.")
    p.add_argument("--height", type=float, default=4.0, help="Height in inches per metric panel.")
    p.add_argument("--dpi", type=int, default=200, help="Output resolution.")
    p.add_argument("--out", default=None, help="Output image path. Default: <script>/plots/<env>_<metric>.png")
    p.add_argument("--also_pdf", action="store_true", help="Also save a vector PDF next to the image.")
    p.add_argument("--list", action="store_true", help="List discovered groups, seeds, and metrics, then exit.")

    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


@dataclass
class SeedTable:
    """One seed's eval.csv, parsed into column arrays."""

    seed_dir: str
    seed: object
    columns: List[str]
    data: Dict[str, np.ndarray]


@dataclass
class RunGroup:
    """One (agent, cfg) combination, containing its seed directories."""

    agent: str
    cfg: str
    label: str
    seed_dirs: List[str] = field(default_factory=list)
    tables: List[SeedTable] = field(default_factory=list)

    @property
    def cfg_clean(self) -> str:
        return self.cfg.strip("/") or "default"


def _subdirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def _seed_index(name: str):
    m = re.fullmatch(r"seed(\d+)", name)
    return int(m.group(1)) if m else name


def build_groups(args: argparse.Namespace) -> List[RunGroup]:
    """Resolve the (agent, cfg, seed*) directories that main.py would have written."""
    offline_root = os.path.join(args.root, args.save_dir, "offline")
    if not os.path.isdir(offline_root):
        raise SystemExit(
            f"Log root not found: {offline_root}\n"
            f"Check --root / --save_dir (main.py uses ~/all_logs + save_dir + 'offline')."
        )

    available_agents = _subdirs(offline_root)
    agents = list(args.agent) if args.agent else available_agents
    if not agents:
        raise SystemExit(f"No agent directories under {offline_root}.")
    if args.agent:
        for agent in agents:
            if agent not in available_agents:
                raise SystemExit(
                    f"Unknown agent '{agent}'. Available under {offline_root}: {available_agents}"
                )

    groups: List[RunGroup] = []
    for agent in agents:
        env_root = os.path.join(offline_root, agent, args.env_name)
        if not os.path.isdir(env_root):
            print(f"[warn] no runs for agent '{agent}' / env '{args.env_name}': {env_root}", file=sys.stderr)
            continue

        available_cfgs = _subdirs(env_root)
        cfgs = list(args.cfg) if args.cfg else available_cfgs
        for cfg in cfgs:
            cfg_dir = os.path.join(env_root, cfg)
            if not os.path.isdir(cfg_dir):
                print(f"[warn] config directory does not exist: {cfg_dir}", file=sys.stderr)
                continue

            seed_dirs = sorted(glob.glob(os.path.join(cfg_dir, "seed*")))
            seed_dirs = [d for d in seed_dirs if os.path.isdir(d)]
            if args.seeds is not None:
                wanted = {int(s) for s in args.seeds}
                seed_dirs = [d for d in seed_dirs if _seed_index(os.path.basename(d)) in wanted]
            seed_dirs.sort(key=lambda d: _seed_index(os.path.basename(d)))

            groups.append(RunGroup(agent=agent, cfg=cfg, label="", seed_dirs=seed_dirs))

    if not groups:
        raise SystemExit(f"No (agent, cfg) groups found for env '{args.env_name}'.")

    # Labels: explicit, or derived from the directory names.
    if args.labels is not None:
        if len(args.labels) != len(groups):
            raise SystemExit(
                f"--labels got {len(args.labels)} value(s) but {len(groups)} group(s) were found."
            )
        for group, label in zip(groups, args.labels):
            group.label = label
    else:
        for group in groups:
            group.label = (
                group.agent if group.cfg_clean == "default" else f"{group.agent}:{group.cfg_clean}"
            )

    # Keep labels unique so the legend cannot merge two runs.
    seen: Dict[str, int] = {}
    for group in groups:
        if group.label in seen:
            seen[group.label] += 1
            group.label = f"{group.label} ({seen[group.label]})"
        else:
            seen[group.label] = 1

    return groups


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_eval_csv(path: str) -> Optional[SeedTable]:
    """Read one eval.csv into arrays. Returns None for missing/empty files."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return None

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        columns = [c for c in (reader.fieldnames or []) if c]
        rows = list(reader)
    if not rows or "step" not in columns:
        return None

    data = {
        col: np.array([_to_float(row.get(col, "")) for row in rows], dtype=float)
        for col in columns
    }
    seed_dir = os.path.dirname(path)
    return SeedTable(
        seed_dir=seed_dir,
        seed=_seed_index(os.path.basename(seed_dir)),
        columns=columns,
        data=data,
    )


def load_tables(group: RunGroup) -> None:
    group.tables = []
    for seed_dir in group.seed_dirs:
        table = read_eval_csv(os.path.join(seed_dir, "eval.csv"))
        if table is not None:
            group.tables.append(table)


def available_metrics(group: RunGroup) -> List[str]:
    cols = set()
    for table in group.tables:
        cols.update(table.columns)
    return sorted(c for c in cols if c.startswith("evaluation/"))


def pick_default_metric(columns: Sequence[str]) -> Optional[str]:
    for metric in DEFAULT_METRIC_PRIORITY:
        if metric in columns:
            return metric
    candidates = [c for c in columns if c.startswith("evaluation/")]
    return candidates[0] if candidates else None


def resolve_metric(columns: Sequence[str], requested: str) -> Optional[str]:
    """Map a user-facing metric name to an actual CSV column."""
    if requested in columns:
        return requested
    prefixed = f"evaluation/{requested}"
    if prefixed in columns:
        return prefixed
    tail = requested.split("/")[-1]
    matches = [c for c in columns if c.split("/")[-1] == tail]
    if len(matches) == 1:
        return matches[0]
    return None


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def interp_curve(steps: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate a seed's curve onto `grid`, leaving NaNs outside its range."""
    order = np.argsort(steps)
    steps, values = steps[order], values[order]

    # Duplicate steps can appear in hand-edited/resumed logs: average them.
    unique_steps, inverse = np.unique(steps, return_inverse=True)
    if unique_steps.size != steps.size:
        sums = np.zeros(unique_steps.size)
        counts = np.zeros(unique_steps.size)
        np.add.at(sums, inverse, values)
        np.add.at(counts, inverse, 1)
        steps, values = unique_steps, sums / counts

    out = np.interp(grid, steps, values)
    out[(grid < steps[0]) | (grid > steps[-1])] = np.nan
    return out


def ema_smooth(values: np.ndarray, weight: float) -> np.ndarray:
    """Exponential moving average, matching the usual RL plotting convention."""
    out = np.full(values.shape, np.nan)
    last = np.nan
    for i, value in enumerate(values):
        if not np.isfinite(value):
            out[i] = last  # carry the previous smoothed value across gaps
            continue
        last = value if not np.isfinite(last) else weight * last + (1.0 - weight) * value
        out[i] = last
    return out


def summarize(
    grid: np.ndarray,
    matrix: np.ndarray,
    seeds: Sequence[object],
    min_seeds: int,
    band: str,
) -> Optional[dict]:
    """Reduce a (seeds x steps) matrix to mean/std/sem and trim empty edges."""
    count = np.isfinite(matrix).sum(axis=0)
    safe_count = np.maximum(count, 1)

    mean = np.nansum(matrix, axis=0) / safe_count
    deviation = np.where(np.isfinite(matrix), matrix - mean[None, :], 0.0)
    std = np.sqrt((deviation ** 2).sum(axis=0) / np.maximum(count - 1, 1))
    sem = np.where(count > 0, std / np.sqrt(safe_count), np.nan)

    mean[count < max(1, min_seeds)] = np.nan
    finite = np.isfinite(mean)
    if not finite.any():
        return None

    start = int(np.argmax(finite))
    stop = int(len(finite) - np.argmax(finite[::-1]))

    grid = grid[start:stop]
    mean = mean[start:stop]
    std = std[start:stop]
    sem = sem[start:stop]
    count = count[start:stop]

    if band == "std":
        lo, hi = mean - std, mean + std
    elif band == "sem":
        lo, hi = mean - sem, mean + sem
    else:
        lo = hi = None

    finite = np.isfinite(mean)
    best_idx = int(np.nanargmax(mean))
    last_idx = int(np.flatnonzero(finite)[-1])

    return {
        "grid": grid,
        "mean": mean,
        "std": std,
        "sem": sem,
        "count": count,
        "lo": lo,
        "hi": hi,
        "seeds": list(seeds),
        "best_mean": float(mean[best_idx]),
        "best_step": float(grid[best_idx]),
        "last_mean": float(mean[last_idx]),
        "last_std": float(std[last_idx]),
        "last_step": float(grid[last_idx]),
    }


def aggregate_group(
    group: RunGroup,
    metric: str,
    smooth: float,
    min_seeds: int,
    band: str,
) -> Optional[dict]:
    curves: List[Tuple[np.ndarray, np.ndarray]] = []
    seeds: List[object] = []

    for table in group.tables:
        key = resolve_metric(table.columns, metric)
        if key is None:
            continue
        steps = table.data["step"]
        values = table.data[key]
        ok = np.isfinite(steps) & np.isfinite(values)
        if not ok.any():
            continue
        curves.append((steps[ok], values[ok]))
        seeds.append(table.seed)

    if not curves:
        return None

    grid = np.unique(np.concatenate([steps for steps, _ in curves]))
    matrix = np.vstack([interp_curve(steps, values, grid) for steps, values in curves])
    if smooth > 0.0:
        matrix = np.vstack([ema_smooth(row, smooth) for row in matrix])

    return summarize(grid, matrix, seeds, min_seeds, band)


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def metric_display_name(metric: str) -> str:
    name = metric.split("/")[-1].replace("_", " ").replace(".", " ").strip()
    return (name[:1].upper() + name[1:]) if name else metric


def step_formatter(x: float, _pos) -> str:
    if x == 0:
        return "0"
    if abs(x) >= 1e6:
        return f"{x / 1e6:g}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:g}k"
    return f"{x:g}"


def _spread_positions(values: Sequence[float], min_gap: float) -> np.ndarray:
    """Nudge label positions apart so direct labels never overlap."""
    out = np.asarray(values, dtype=float).copy()
    order = np.argsort(out)
    for left, right in zip(order[:-1], order[1:]):
        if out[right] - out[left] < min_gap:
            out[right] = out[left] + min_gap
    return out


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS_COLOR,
            "axes.labelcolor": TEXT_PRIMARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": TEXT_PRIMARY,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": TEXT_PRIMARY,
            "font.family": "DejaVu Sans",
            "font.size": 10,
        }
    )


def fmt_value(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def make_figure(
    args: argparse.Namespace,
    groups: Sequence[RunGroup],
    metrics: Sequence[str],
    results: Dict[str, List[Optional[dict]]],
    out_path: str,
) -> None:
    apply_style()

    n_panels = len(metrics)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(args.width, args.height * n_panels),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    # One color per group, fixed regardless of which groups have data: color
    # follows the entity, never its rank.
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(groups))]
    label_axis = None

    for panel_idx, metric in enumerate(metrics):
        ax = axes[panel_idx]
        plotted = 0
        ends: List[Tuple[float, float, str]] = []

        for group_idx, group in enumerate(groups):
            result = results[metric][group_idx]
            if result is None:
                continue

            color = colors[group_idx]
            x, mean = result["grid"], result["mean"]
            lo, hi = result["lo"], result["hi"]

            # --xmax means "do not show anything past this step", so truncate
            # the data rather than only zooming the view.
            if args.xmax is not None:
                keep = x <= args.xmax
                x, mean = x[keep], mean[keep]
                if lo is not None:
                    lo, hi = lo[keep], hi[keep]
                if x.size == 0:
                    continue

            if lo is not None:
                ax.fill_between(
                    x,
                    lo,
                    hi,
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                    zorder=2,
                )
            ax.plot(
                x,
                mean,
                color=color,
                linewidth=2.0,
                solid_capstyle="round",
                label=group.label,
                zorder=3,
            )
            plotted += 1

            finite = np.isfinite(mean)
            if finite.any():
                end_idx = int(np.flatnonzero(finite)[-1])
                ends.append((float(x[end_idx]), float(mean[end_idx]), group.label))

        if plotted == 0:
            ax.text(
                0.5,
                0.5,
                "no data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=TEXT_SECONDARY,
            )

        ax.set_title(metric_display_name(metric), loc="left", pad=8)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        if args.logy and plotted:
            ax.set_yscale("log")
        if args.ymin is not None or args.ymax is not None:
            current = ax.get_ylim()
            ax.set_ylim(
                args.ymin if args.ymin is not None else current[0],
                args.ymax if args.ymax is not None else current[1],
            )
        elif plotted and metric.endswith("success") and not args.logy:
            # Success is a rate in [0, 1]; make that range explicit.
            values = np.concatenate(
                [results[metric][i]["mean"] for i in range(len(groups))
                 if results[metric][i] is not None]
            )
            if np.nanmin(values) >= 0.0 and np.nanmax(values) <= 1.0:
                ax.set_ylim(-0.02, 1.02)

        if args.xmax is not None:
            ax.set_xlim(right=args.xmax)

        # Direct labels for a small number of series (identity is never
        # color-alone), with a legend kept alongside.
        want_labels = args.direct_labels == "on" or (
            args.direct_labels == "auto" and 2 <= plotted <= 4
        )
        if want_labels and ends:
            x0, x1 = ax.get_xlim()
            ax.set_xlim(x0, x1 + 0.14 * (x1 - x0))
            y0, y1 = ax.get_ylim()
            spread = _spread_positions([e[1] for e in ends], 0.055 * (y1 - y0))
            ax.set_ylim(y0, max(y1, float(spread.max()) + 0.03 * (y1 - y0)))
            for (x_end, _, label), y_pos in zip(ends, spread):
                ax.annotate(
                    label,
                    xy=(x_end, y_pos),
                    xytext=(6, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8.5,
                    color=TEXT_SECONDARY,
                    zorder=4,
                )

        # Direct-label padding must not override explicit axis limits.
        if args.xmax is not None:
            ax.set_xlim(right=args.xmax)
        if args.ymin is not None or args.ymax is not None:
            bottom, top = ax.get_ylim()
            ax.set_ylim(
                args.ymin if args.ymin is not None else bottom,
                args.ymax if args.ymax is not None else top,
            )

        if label_axis is None and plotted > 0:
            label_axis = ax

    # A legend is present whenever more than one series is drawn.
    if label_axis is not None and len(groups) >= 2:
        label_axis.legend(loc="best")

    axes[-1].set_xlabel(args.xlabel)
    axes[-1].xaxis.set_major_formatter(FuncFormatter(step_formatter))

    title = args.title if args.title is not None else args.env_name
    if len(groups) == 1:
        title = f"{title}  ({groups[0].label})"
    fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    if args.also_pdf:
        pdf_path = os.path.splitext(out_path)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {out_path}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def print_listing(groups: Sequence[RunGroup], offline_root: str) -> None:
    print(f"Discovered {len(groups)} group(s) under {offline_root}\n")
    for group in groups:
        present = [t.seed for t in group.tables]
        missing = [
            _seed_index(os.path.basename(d))
            for d in group.seed_dirs
            if _seed_index(os.path.basename(d)) not in present
        ]
        metrics = available_metrics(group)
        print(f"  {group.agent}  cfg={group.cfg}  dir={os.path.join(offline_root, group.agent, '')}")
        print(f"    seeds with eval.csv: {present or 'none'}")
        if missing:
            print(f"    seeds missing eval.csv: {missing}")
        print(f"    metrics: {', '.join(metrics) if metrics else '(none)'}")
        print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not (0.0 <= args.smooth < 1.0):
        raise SystemExit(f"--smooth must be in [0, 1), got {args.smooth}.")

    groups = build_groups(args)
    for group in groups:
        load_tables(group)

    if args.list:
        offline_root = os.path.join(args.root, args.save_dir, "offline")
        print_listing(groups, offline_root)
        return 0

    # Drop groups with no readable eval.csv and explain why.
    usable_groups: List[RunGroup] = []
    for group in groups:
        if not group.seed_dirs:
            print(f"[warn] {group.label}: no seed* directories under {group.cfg!r}.", file=sys.stderr)
        elif not group.tables:
            print(
                f"[warn] {group.label}: found {len(group.seed_dirs)} seed dir(s) but no non-empty "
                f"eval.csv. main.py writes the first row after eval_interval steps "
                f"(default 100000); the run may still be in its first eval interval.",
                file=sys.stderr,
            )
        else:
            usable_groups.append(group)

    if not usable_groups:
        raise SystemExit("No usable eval.csv files found. Nothing to plot.")
    groups = usable_groups

    if len(groups) > len(PALETTE):
        raise SystemExit(
            f"{len(groups)} series requested, but the categorical palette supports {len(PALETTE)}. "
            "Fold runs into a baseline/'Other' or facet them into separate figures."
        )

    # Resolve which metrics to plot.
    all_columns: List[str] = []
    for group in groups:
        all_columns.extend(available_metrics(group))
    all_columns = sorted(set(all_columns))
    if not all_columns:
        raise SystemExit("No 'evaluation/*' columns found in any eval.csv.")

    if args.metric:
        metrics: List[str] = []
        for requested in args.metric:
            resolved = resolve_metric(all_columns, requested)
            if resolved is None:
                raise SystemExit(
                    f"Metric '{requested}' not found. Available metrics:\n  "
                    + "\n  ".join(all_columns)
                )
            metrics.append(resolved)
    else:
        default = pick_default_metric(all_columns)
        if default is None:
            raise SystemExit(
                "Could not auto-detect a metric. Pass --metric. Available:\n  "
                + "\n  ".join(all_columns)
            )
        metrics = [default]

    # Aggregate every group for every metric.
    results: Dict[str, List[Optional[dict]]] = {}
    for metric in metrics:
        results[metric] = []
        for group in groups:
            result = aggregate_group(group, metric, args.smooth, args.min_seeds, args.band)
            if result is None:
                print(
                    f"[warn] {group.label}: no usable values for '{metric}' "
                    f"in any seed's eval.csv.",
                    file=sys.stderr,
                )
            results[metric].append(result)

    if not any(r is not None for rows in results.values() for r in rows):
        raise SystemExit("All groups were empty for the requested metric(s).")

    # Choose the output path.
    if args.out:
        out_path = args.out
        if not os.path.splitext(out_path)[1]:
            out_path += ".png"
    else:
        suffix = metrics[0] if len(metrics) == 1 else "comparison"
        suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", suffix.split("/")[-1]).strip("_")
        out_path = os.path.join(SCRIPT_DIR, "plots", f"{args.env_name}_{suffix}.png")

    make_figure(args, groups, metrics, results, out_path)

    # Summary table: the "table view" of the same numbers, and a paper-friendly
    # record of final performance.
    summary_path = os.path.splitext(out_path)[0] + "_summary.csv"
    rows = []
    for metric in metrics:
        for group, result in zip(groups, results[metric]):
            if result is None:
                continue
            rows.append(
                {
                    "run": group.label,
                    "agent": group.agent,
                    "cfg": group.cfg_clean,
                    "metric": metric,
                    "n_seeds": len(result["seeds"]),
                    "seeds": " ".join(str(s) for s in result["seeds"]),
                    "last_step": int(result["last_step"]),
                    "final_mean": result["last_mean"],
                    "final_std": result["last_std"],
                    "best_mean": result["best_mean"],
                    "best_step": int(result["best_step"]),
                    "smooth": args.smooth,
                    "band": args.band,
                }
            )

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {summary_path}")

    # Console table.
    print()
    header = f"{'run':<28} {'metric':<34} {'seeds':>5} {'final':>18} {'best':>16}"
    print(header)
    print("-" * len(header))
    for row in rows:
        final = f"{fmt_value(row['final_mean'])} +/- {fmt_value(row['final_std'])}"
        best = f"{fmt_value(row['best_mean'])} @ {int(row['best_step'])}"
        print(
            f"{row['run']:<28} {row['metric']:<34} {row['n_seeds']:>5} "
            f"{final:>18} {best:>16}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
