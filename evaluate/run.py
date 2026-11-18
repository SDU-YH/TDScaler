import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from environment.metrics import write_json
from evaluate.core.statistics import evaluate_group

ROOT = Path(__file__).resolve().parents[1]


def main():
    entries = json.loads((ROOT / "evaluate/groups.json").read_text())
    parser = argparse.ArgumentParser(description="Save only group, mean cost, SLA violation percentage and normalized NCE.")
    parser.add_argument("--group", choices=("all", *entries), default="all")
    parser.add_argument("--days", type=int, nargs="+", help="default: every nonempty trace day")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT,
                        help="default: TDScaler-main root directory")
    args = parser.parse_args()
    if args.group == "all" and (args.config or args.checkpoint):
        parser.error("--config/--checkpoint overrides require selecting a single --group")
    if args.config:
        entries[args.group]["config"] = str(args.config.resolve())
    if args.checkpoint:
        entries[args.group]["checkpoint"] = str(args.checkpoint.resolve())
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.days and (any(d < 1 for d in args.days) or len(set(args.days)) != len(args.days)):
        parser.error("--days requires distinct positive day numbers")
    groups = list(entries) if args.group == "all" else [args.group]
    jobs = [(g, entries[g], args.days, args.device) for g in groups]
    if len(jobs) == 1 or args.workers == 1:
        results = [evaluate_group(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            results = list(pool.map(evaluate_group, jobs))
    write_json(args.output_dir.resolve() / "summary.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
