import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogFormatterMathtext, SymmetricalLogLocator
import numpy as np

from plot_data import PlotData

COLORS = ["#3B5998", "#3D7A44", "#B04A46"]


def boxplot(ax, arrays, labels, colors):
    result = ax.boxplot(arrays, widths=0.45, patch_artist=True, showfliers=True,
        medianprops=dict(color="#111111", linewidth=1, zorder=4),
        whiskerprops=dict(color="#333333", linewidth=0.6),
        capprops=dict(color="#333333", linewidth=0.6),
        flierprops=dict(marker="o", markersize=1.2, markerfacecolor="#444444",
                        markeredgecolor="none", alpha=0.4))
    for box, color in zip(result["boxes"], colors):
        box.set(facecolor=color, edgecolor="#333333", linewidth=0.65, zorder=3)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("original", "bird"), default="original")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bird = args.workload == "bird"
    source = PlotData(args.workload)
    panels = source.panels
    loader = source.load_environment
    draw_reward_panel = source.draw_reward
    display_names = {name: f"SQL-{name}" if bird else name for name in panels}
    filename = "TDScaler_System_plt_BIRD.pdf" if bird else "TDScaler_System_plt.pdf"
    args.output = args.output or Path(__file__).resolve().parents[1] / filename
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
        "mathtext.fontset": "dejavuserif", "axes.linewidth": 0.65,
        "lines.linewidth": 1.0, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "savefig.bbox": None, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    data = [loader(name, path) for name, path in panels.items()]
    row_height = 2.2
    panel_height = 1.276
    original_row_gap = row_height - panel_height
    row_gap = 0.60
    row_pitch = panel_height + row_gap
    figure_height = 2 * row_height - (original_row_gap - row_gap)
    fig = plt.figure(figsize=(5, figure_height))
    # Keep axes and outer margins unchanged; vary only the vertical gap
    # and shorten the canvas by the corresponding amount.
    def panel(left, bottom):
        return fig.add_axes([left, bottom / figure_height, 0.35,
                             panel_height / figure_height])
    throughput_ax = panel(0.10, 0.55)
    ttft_ax = panel(0.59, 0.55)
    reward_ax = panel(0.10, 0.55 + row_pitch)
    reward_export = draw_reward_panel(reward_ax)
    for text in reward_ax.get_legend().get_texts():
        text.set_text(display_names[text.get_text()])
        text.set_fontsize(7)
    instance_ax = panel(0.59, 0.55 + row_pitch)
    for index, (name, (_, steps, _)) in enumerate(zip(panels, data)):
        # Output tokens of requests completed within each executed interval.
        edges = np.r_[steps["start_time_s"], steps["end_time_s"].iloc[-1]] / 3600
        throughput_ax.stairs(steps["output_token_throughput_tps"], edges, baseline=None,
            color=COLORS[index], linestyle=["-", "--", "-."][index],
            linewidth=0.85, label=display_names[name], zorder=3)
        # Interval-end snapshots, not interpolated or treated as start counts.
        instance_ax.scatter(steps["end_time_s"] / 3600, steps["total_live_instances"],
            color=COLORS[index], marker=["o", "x", "^"][index], s=5,
            linewidths=0.45, alpha=0.6, label=display_names[name], zorder=3)
    for ax in (throughput_ax, instance_ax):
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xlabel("Timeline (h)", fontsize=7.5, labelpad=3)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=ax is instance_ax))
    throughput_ax.set_ylabel("Throughput (token/s)", fontsize=7.5, labelpad=2)
    instance_ax.set_ylabel("Instance Count", fontsize=7.5, labelpad=2)
    max_instances = max(float(item[1]["total_live_instances"].max()) for item in data)
    instance_ax.set_ylim(0, max_instances + 0.2)
    tick_step = max(1, int(np.ceil(max_instances / 4)))
    instance_ax.set_yticks(np.arange(0, max_instances + 0.01, tick_step))
    throughput_ax.legend(loc="upper left", frameon=False, fontsize=7,
                          handlelength=1.4, handletextpad=0.35, labelspacing=0.15)
    instance_ax.legend(loc="upper right" if bird else "upper left", frameon=False, fontsize=7,
                       handlelength=1.0, handletextpad=0.2, labelspacing=0.15)

    values = [entry[0] for entry in data]
    boxplot(ttft_ax, values, [display_names[name] for name in panels], COLORS)
    # Compress the low-latency region without hiding any requests: linear below
    # 0.1 s, logarithmic above it. Keep the count annotation's font unchanged.
    ttft_ax.set_yscale("symlog", linthresh=0.1, linscale=0.5)
    ttft_ax.set_ylim(min(x.min() for x in values) / 1.4,
                     max(x.max() for x in values) * 3.0)
    for position, (samples, _, report) in enumerate(data, start=1):
        ttft_ax.text(position, 0.97, f"{report['completed_requests']:,}",
            transform=ttft_ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=6.5, zorder=5)
    ttft_ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ttft_ax.set_yticks([0.01, 0.1, 1.0])
    ttft_ax.yaxis.set_minor_locator(SymmetricalLogLocator(
        linthresh=0.1, base=10, subs=np.arange(2, 10)))
    ttft_ax.axhline(1, color="#555555", linestyle="--", linewidth=0.8)
    ttft_ax.set_ylabel("TTFT (s)", fontsize=7.5, labelpad=2)

    captions = [(reward_ax, "(a) Reward"),
                (instance_ax, "(b) Instance Count"),
                (throughput_ax, "(c) Throughput"),
                (ttft_ax, "(d) TTFT")]
    for ax, caption in captions:
        ax.set_facecolor("#F5F5F5")
        ax.set_axisbelow(True)
        ax.grid(which="major", axis="y" if ax is ttft_ax else "both",
                color="#CCCCCC", linestyle="--", linewidth=0.5)
        if ax is ttft_ax:
            ax.grid(which="minor", axis="y", color="#E4E4E4",
                    linestyle=":", linewidth=0.4)
        ax.tick_params(axis="both", pad=1)
        for spine in ax.spines.values():
            spine.set(linewidth=0.65, edgecolor="#333333")
        bounds = ax.get_position()
        y = bounds.y0 - 0.44 / figure_height
        fig.text(bounds.x0 + bounds.width / 2, y, caption,
                 ha="center", va="bottom", fontsize=8)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
