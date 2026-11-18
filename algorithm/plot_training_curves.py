from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EPISODE_SERIES = (
    ("episode_reward", "Episode reward", "tab:green"),
    ("total_cost_usd", "Resource cost (USD)", "tab:orange"),
    ("p99_ttft_s", "P99 TTFT (s)", "tab:blue"),
)

def _read_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows: List[dict] = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"log is empty: {path}")
    return rows

def _load_columns(
    path: Path,
    x_name: str,
    series: Sequence[Tuple[str, str, str]],
) -> Dict[str, np.ndarray]:
    rows = _read_rows(path)
    required = {x_name, *(name for name, _, _ in series)}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"log is missing columns: {sorted(missing)}")
    result = {x_name: np.asarray([int(row[x_name]) for row in rows])}
    for name, _, _ in series:
        result[name] = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
    return result

def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    weights = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, weights, mode="valid")

def _save_figure(figure, output_dir: Path, stem: str) -> List[Path]:
    outputs = [output_dir / f"{stem}.pdf"]
    for output in outputs:
        figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return outputs

def _plot_episode_curves(
    episodes_csv: Path,
    output_dir: Path,
    rolling_window: int,
    sla_s: float,
    method_label: str,
) -> List[Path]:
    datasets = {
        method_label: _load_columns(episodes_csv, "episode", EPISODE_SERIES)
    }
    episodes = datasets[method_label]["episode"]
    if rolling_window <= 0 or rolling_window > len(episodes):
        raise ValueError("rolling_window must be between 1 and the episode count")

    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    styles = {
        method_label: ("tab:blue", "-"),
    }
    for axis, (name, ylabel, _) in zip(axes, EPISODE_SERIES):
        for label, data in datasets.items():
            color, linestyle = styles[label]
            values = data[name]
            axis.plot(
                episodes,
                values,
                color=color,
                linestyle=linestyle,
                alpha=0.18,
                linewidth=0.7,
            )
            smooth = _moving_average(values, rolling_window)
            smooth_x = episodes[rolling_window - 1 :]
            axis.plot(
                smooth_x,
                smooth,
                color=color,
                linestyle=linestyle,
                linewidth=1.9,
                label=f"{label} ({rolling_window}-episode mean)",
            )
        if name == "p99_ttft_s":
            axis.axhline(sla_s, color="black", linestyle="--", linewidth=1.0, label="SLA")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="best", frameon=False)
    axes[2].legend(loc="best", frameon=False)

    axes[-1].set_xlabel("Training episode")
    title = f"{method_label} training episode metrics"
    figure.suptitle(title)
    figure.tight_layout()
    return _save_figure(figure, output_dir, "training_curves")

def plot_curves(
    episodes_csv: Path,
    output_dir: Path,
    rolling_window: int = 20,
    sla_s: float = 1.0,
    method_label: str = "HAN-DTQN",
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = _plot_episode_curves(
        episodes_csv,
        output_dir,
        rolling_window,
        sla_s,
        method_label,
    )
    return outputs

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--sla-s", type=float, default=1.0)
    parser.add_argument("--method-label", default="HAN-DTQN proposed")
    return parser
