from pathlib import Path
import json

import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parent.parent


class OneDecimalScalarFormatter(ScalarFormatter):
    def _set_format(self):
        self.format = r"$\mathdefault{%1.1f}$"


class PlotData:
    methods = ["TDScaler", "TDScaler-WoT", "TDScaler-WoHT"]
    colors = ["#3B5998", "#3D7A44", "#B04A46"]
    linestyles = ["-", "--", "-."]
    OneDecimalScalarFormatter = OneDecimalScalarFormatter

    def __init__(self, workload="original"):
        if workload not in ("original", "bird"):
            raise ValueError("workload must be original or bird")
        self.bird = workload == "bird"
        suffix = "-BIRD" if self.bird else ""
        names = {"A.3x": "TDScaler-Llama-BurstGPT",
                 "A.5x": "TDScaler-Llama-Azure", "B.3x": "TDScaler-Qwen"}
        self.runs = {group: ROOT / "train" / (name + suffix)
                     for group, name in names.items()}
        self.panels = {group: folder / "evaluation_original/day01"
                       for group, folder in self.runs.items()}

    def training_panels(self):
        groups = ["A.5x", "B.3x"] if self.bird else ["A.3x", "A.5x"]
        files = []
        for group in groups:
            folder = self.runs[group]
            files.append({method: folder / "train.csv" if method == "TDScaler"
                          else folder / "ablations" / method / "train.csv"
                          for method in self.methods})
        return groups, files

    def load_multi_method_ttft(self, files):
        data, reference = {}, None
        required = ["episode", "p99_ttft_s", "total_cost_usd", "episode_reward"]
        for method in self.methods:
            frame = pd.read_csv(files[method]).sort_values("episode").reset_index(drop=True)
            if (frame.empty or frame.episode.duplicated().any()
                    or not np.isfinite(frame[required].to_numpy(float)).all()
                    or (frame.p99_ttft_s < 0).any()):
                raise ValueError(f"Invalid training data: {files[method]}")
            if reference is not None and not np.array_equal(reference, frame.episode):
                raise ValueError("Training episode ranges differ")
            reference = frame.episode.to_numpy()
            data[method] = frame
        return data

    def load_environment(self, name, directory):
        directory = Path(directory)
        report = json.loads((directory / "report.json").read_text())
        requests = pd.read_csv(directory / "requests.csv")
        steps = pd.read_csv(directory / "steps.csv").sort_values("start_time_s").reset_index(drop=True)
        starts = steps.start_time_s.to_numpy(float)
        ends = steps.end_time_s.to_numpy(float)
        intervals = ends - starts
        if (not len(steps) or np.any(intervals <= 0)
                or not np.isclose(starts[0], 0) or not np.isclose(ends[-1], 86400)
                or not np.allclose(starts[1:], ends[:-1])):
            raise ValueError(f"Expected a contiguous complete daily timeline: {directory}")
        if len(requests) != report["requests"] or requests.request_id.duplicated().any():
            raise ValueError(f"Invalid request count or IDs: {directory}")
        first = pd.to_numeric(requests.first_token_time_s)
        observed = first.notna()
        ttft = (first[observed] - requests.loc[observed, "arrival_time_s"]).to_numpy(float)
        if not len(ttft) or not np.isfinite(ttft).all() or (ttft <= 0).any():
            raise ValueError(f"Invalid observed TTFT: {directory}")
        completion = pd.to_numeric(requests.completion_time_s)
        completed = completion.notna() & completion.between(starts[0], ends[-1])
        bins = np.searchsorted(ends, completion[completed].to_numpy(float), side="left")
        if not np.array_equal(np.bincount(bins, minlength=len(steps)), steps.completed_requests):
            raise ValueError(f"Completion timeline/count mismatch: {directory}")
        output_tokens = requests.loc[completed, "served_output_tokens"].to_numpy(float)
        totals = np.bincount(bins, weights=output_tokens, minlength=len(steps))
        live = steps.filter(regex=r"^live_").to_numpy(float)
        if (not live.shape[1] or not np.isfinite(live).all() or (live < 0).any()
                or not np.isfinite(output_tokens).all() or (output_tokens < 0).any()):
            raise ValueError(f"Invalid instance/token data: {directory}")
        throughput = steps.throughput_rps.to_numpy(float)
        if not np.allclose(throughput * intervals, steps.completed_requests):
            raise ValueError(f"Throughput/count mismatch: {directory}")
        export = pd.DataFrame(dict(environment=name, start_time_s=starts, end_time_s=ends,
            interval_s=intervals, throughput_rps=throughput,
            completed_output_tokens=totals, output_token_throughput_tps=totals / intervals,
            total_live_instances=live.sum(axis=1)))
        report.update(completed_requests=int(completed.sum()))
        return ttft, export, report

    def draw_reward(self, ax):
        exported = []
        for index, (group, folder) in enumerate(self.runs.items()):
            frame = pd.read_csv(folder / "train.csv").sort_values("episode").reset_index(drop=True)
            values = frame.episode_reward.to_numpy(float)
            if not len(values) or not np.isfinite(values).all():
                raise ValueError(f"Invalid reward data: {folder}")
            selected = np.r_[True, values[1:] > np.maximum.accumulate(values)[:-1]]
            peaks = frame.loc[selected]
            style = {} if self.bird else dict(solid_joinstyle="miter", solid_capstyle="butt")
            ax.plot(peaks.episode, peaks.episode_reward, label=group,
                color=self.colors[index], linestyle=self.linestyles[index],
                linewidth=1.7, alpha=0.9, marker=["o", "s", "^"][index],
                markersize=3.5, markerfacecolor="white", markeredgewidth=0.6, **style)
            records = frame[["episode", "episode_reward"]].copy()
            records["source"], records["is_record_high"] = group, selected
            exported.append(records)
        ax.set_xlabel("Episode", fontsize=7.5, labelpad=3)
        ax.set_ylabel("Reward", fontsize=7.5, labelpad=2)
        ax.legend(loc="lower right", fontsize=7 if self.bird else 7.5,
                  frameon=False, handlelength=2, labelspacing=0.3)
        formatter = OneDecimalScalarFormatter(useMathText=True, useOffset=False)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.get_offset_text().set_fontsize(7)
        return pd.concat(exported, ignore_index=True)
