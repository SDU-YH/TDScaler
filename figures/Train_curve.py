import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np
import pandas as pd

from plot_data import PlotData


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("original", "bird"), default="original")
    parser.add_argument("--window", type=int, default=None,
                        help="Moving mean window; default max(1, episodes // 15).")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.window is not None and args.window < 1:
        parser.error("window must be positive")
    source = PlotData(args.workload)
    panels, files = source.training_panels()
    filename = "Train_curve_BIRD.pdf" if source.bird else "Train_curve.pdf"
    args.output = args.output or Path(__file__).resolve().parents[1] / filename
    methods, colors, linestyles = source.methods, source.colors, source.linestyles
    OneDecimalScalarFormatter = source.OneDecimalScalarFormatter

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
        "axes.linewidth": 0.65, "lines.linewidth": 1.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "savefig.bbox": None, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    datasets = [source.load_multi_method_ttft(paths) for paths in files]
    for data in datasets:
        for method, frame in data.items():
            if "total_cost_usd" not in frame:
                raise ValueError(f"{method}: missing total_cost_usd")
            cost = frame["total_cost_usd"].to_numpy(dtype=float)
            if not np.isfinite(cost).all() or (cost < 0).any():
                raise ValueError(f"{method}: invalid cost values")
    all_ttft = np.concatenate([frame["p99_ttft_s"].to_numpy()
                               for data in datasets for frame in data.values()])
    if (all_ttft <= 0).any():
        raise ValueError("Log TTFT axis requires positive data; no points are silently removed")
    ttft_limits = (min(float(all_ttft.min()), 1) / 1.15,
                   max(float(all_ttft.max()), 1) * 1.15)

    # Preserve the former cost-axis dimensions, with TTFT axes 70% as tall.
    # Additional vertical space is solely for complete x axes and four captions.
    cost_height = 96.298 / 72
    ttft_height = .7 * cost_height
    bottom = .60
    gap = .65
    upper_bottom = bottom + cost_height + gap
    # Compact top margin: bring the shared legend closer without resizing axes.
    figure_height = upper_bottom + ttft_height + .36
    fig = plt.figure(figsize=(5, figure_height))
    axes = np.array([[fig.add_axes([left, y / figure_height, 150.1609570312 / 360,
                                    height / figure_height])
                      for left in [31.9577929687 / 360, .56]]
                     for y, height in [(upper_bottom, ttft_height), (bottom, cost_height)]])
    handles, exported = [], []
    for column, data in enumerate(datasets):
        for index, method in enumerate(methods):
            frame = data[method]
            episodes = frame["episode"].to_numpy()
            window = args.window or max(1, len(frame) // 15)
            record = pd.DataFrame({
                "panel": panels[column], "method": method,
                "episode": episodes,
            })
            for row, metric in enumerate(("p99_ttft_s", "total_cost_usd")):
                values = frame[metric].astype(float)
                trend = values.rolling(window, min_periods=1).mean()
                ax = axes[row, column]
                ax.plot(episodes, values, color=colors[index], linewidth=0.4,
                        alpha=0.25, zorder=2)
                line, = ax.plot(episodes, trend, color=colors[index],
                                linestyle=linestyles[index], linewidth=1.1,
                                label=method, zorder=3 + index)
                if row == 0 and column == 0:
                    handles.append(line)
                record[f"raw_{metric}"] = values.to_numpy()
                record[f"mean_{metric}"] = trend.to_numpy()
            record["moving_average_window"] = window
            exported.append(record)

        upper, lower = axes[:, column]
        upper.set_yscale("log")
        upper.set_ylim(*ttft_limits)
        upper.yaxis.set_major_locator(FixedLocator([0.3, 0.5, 1, 2, 5, 10, 30]))
        upper.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        upper.yaxis.set_minor_formatter(NullFormatter())
        upper.axhline(1, color="#555555", linestyle=":", linewidth=0.8, zorder=2)
        upper.text(0.98, 1, "SLA = 1s", transform=upper.get_yaxis_transform(),
                   ha="right", va="bottom", fontsize=6)
        lower.set_ylim(bottom=0)
        formatter = OneDecimalScalarFormatter(useMathText=True, useOffset=False)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        lower.yaxis.set_major_formatter(formatter)
        lower.yaxis.get_offset_text().set_fontsize(7)
        lower.set_xlim(min(frame["episode"].min() for frame in data.values()),
                       max(frame["episode"].max() for frame in data.values()))
        for ax in (upper, lower):
            ax.set_xlabel("Episode", fontsize=7.5, labelpad=3)
            ax.tick_params(axis="x", labelbottom=True, pad=1)
            final_episode = max(frame["episode"].max() for frame in data.values())
            axis_end = max(50, int(np.ceil(final_episode / 50)) * 50)
            ax.set_xlim(0, axis_end)
            ax.set_xticks(np.linspace(0, axis_end, 4))
            ax.set_facecolor("#F9F9F9")
            ax.grid(color="#CCCCCC", linestyle="--", linewidth=0.5, zorder=0)
            for spine in ax.spines.values():
                spine.set_color("#333333")

    axes[0, 0].set_ylabel("P99 TTFT (s)", fontsize=7.5)
    axes[1, 0].set_ylabel("Cost (¥)", fontsize=7.5)
    fig.legend(handles=handles, labels=methods, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
               columnspacing=1.3, handletextpad=0.4, handlelength=1.6)
    captions = [f"(a) {panels[0]} - TTFT", f"(b) {panels[1]} - TTFT",
                f"(c) {panels[0]} - Cost", f"(d) {panels[1]} - Cost"]
    for ax, caption in zip(axes.flat, captions):
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, box.y0 - .44 / figure_height, caption,
                 ha="center", va="bottom", fontsize=8)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
