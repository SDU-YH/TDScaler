import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_data import PlotData

ORIGINAL = PlotData("original")
BIRD = PlotData("bird")
SOURCES = {
    "BrustGPT": ORIGINAL.runs["A.3x"] / "evaluation_original/day01/steps.csv",
    "Azure": ORIGINAL.runs["A.5x"] / "evaluation_original/day02/steps.csv",
    "BrustGPT+BIRD": BIRD.runs["A.3x"] / "evaluation_original/day01/steps.csv",
    "Azure+BIRD": BIRD.runs["A.5x"] / "evaluation_original/day03/steps.csv",
}
COLORS = ["#3B5998", "#3D7A44", "#B04A46", "#6B4C9A"]
LINESTYLES = ["-", "--", "-.", ":"]


def load_intervals(name, path):
    source = path
    frame = pd.read_csv(source).sort_values("start_time_s").reset_index(drop=True)
    starts = frame.start_time_s.to_numpy(float)
    ends = frame.end_time_s.to_numpy(float)
    lengths = ends - starts
    if (not len(frame) or not np.isfinite(np.r_[starts, ends]).all()
        or np.any(lengths <= 0) or not np.isclose(starts[0], 0)
        or not np.isclose(ends[-1], 86400) or not np.allclose(starts[1:], ends[:-1])):
        raise ValueError(f"Expected contiguous full-day decision intervals: {source}")
    return pd.DataFrame({"trace": name, "day": int(path.parent.name.removeprefix("day")),
        "start_time_s": starts, "end_time_s": ends, "interval_s": lengths}), source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "Scaling_Interval.pdf")
    args = parser.parse_args()
    # Match Three_reward.py: fixed canvas, physical axes, fonts and caption gaps.
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
        "mathtext.fontset": "dejavuserif", "axes.linewidth": .65,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "savefig.bbox": None, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(5, 2.2))
    cdf_ax = fig.add_axes([.10, .55 / 2.2, 1.75 / 5, 1.276 / 2.2])
    time_ax = fig.add_axes([.59, .55 / 2.2, 1.75 / 5, 1.276 / 2.2])
    exports, reports = [], []
    loaded = {name: load_intervals(name, path) for name, path in SOURCES.items()}
    limit = max(1000, int(np.ceil(max(frame.interval_s.max() for frame, _ in loaded.values())/400))*400)
    for index, (name, path) in enumerate(SOURCES.items()):
        frame, source = loaded[name]
        lengths = frame.interval_s.to_numpy()
        values, counts = np.unique(lengths, return_counts=True)
        probabilities = counts.cumsum() / len(lengths)
        style = dict(color=COLORS[index], linestyle=LINESTYLES[index],
                     linewidth=1.1, label=name)
        cdf_ax.step(np.r_[0, values, limit], np.r_[0, probabilities, 1],
                    where="post", **style)
        # Hold each decision interval value over its actual execution duration.
        time_ax.step(np.r_[frame.start_time_s.to_numpy(), frame.end_time_s.iloc[-1]] / 3600,
                     np.r_[lengths, lengths[-1]], where="post", **style)
        exports.append(frame)
        reports.append(dict(trace=name, day=int(frame.day.iloc[0]), source=str(source),
            decision_count=len(frame), min_interval_s=float(lengths.min()),
            max_interval_s=float(lengths.max()), duration_s=float(lengths.sum())))
    cdf_ax.set(xlim=(0, limit), ylim=(0, 1.05),
               xticks=np.linspace(0,limit,5), yticks=[0, .25, .5, .75, 1])
    time_ax.set(xlim=(0, 24), ylim=(0, limit),
                xticks=[0, 6, 12, 18, 24], yticks=np.linspace(0,limit,5))
    cdf_ax.set_xlabel("Scaling Interval (s)", fontsize=7.5, labelpad=3)
    cdf_ax.set_ylabel("CDF", fontsize=7.5, labelpad=2)
    time_ax.set_xlabel("Timeline (h)", fontsize=7.5, labelpad=3)
    time_ax.set_ylabel("Scaling Interval (s)", fontsize=7.5, labelpad=2)
    handles, labels = cdf_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .995),
               ncol=4, frameon=False, fontsize=7, handlelength=1.25,
               handletextpad=.3, columnspacing=.65)
    for ax, caption in ((cdf_ax, "(a) Interval Distribution"),
                        (time_ax, "(b) Interval over Time")):
        ax.set_facecolor("#F5F5F5")
        ax.set_axisbelow(True)
        ax.grid(which="major", axis="both", color="#CCCCCC", linestyle="--", linewidth=.5)
        ax.tick_params(axis="x", pad=1)
        ax.tick_params(axis="y", pad=1)
        for spine in ax.spines.values():
            spine.set(linewidth=.65, edgecolor="#333333")
        bounds = ax.get_position()
        fig.text(bounds.x0 + bounds.width / 2, bounds.y0 - .44 / fig.get_figheight(),
                 caption, ha="center", va="bottom", fontsize=8)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
