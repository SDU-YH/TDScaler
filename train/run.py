import argparse
from dataclasses import replace
import json
from pathlib import Path

from algorithm.train import train
from environment.config import load_config
from evaluate.core.evaluate import _select_device

ROOT = Path(__file__).resolve().parents[1]


def main():
    groups = json.loads((ROOT / 'evaluate/groups.json').read_text())
    parser = argparse.ArgumentParser(description='Train TDScaler in one of the six experiment environments.')
    parser.add_argument('--group', choices=tuple(groups), default='SQL-B.3x')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--episodes', type=int, default=150)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', type=Path, help='default: train/results/<group>')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--rebuild-cache', action='store_true')
    parser.add_argument('--resume-checkpoint', type=Path)
    args = parser.parse_args()
    output = (args.output_dir or ROOT / 'train/results' / args.group).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be empty; use a new output directory when resuming')
    config = load_config(args.config or ROOT / groups[args.group]['config'])
    seed = args.seed if args.seed is not None else config.seed
    cache = (args.cache or ROOT / 'train/cache' / f'{args.group}_random32_seed{seed}.npz').resolve()
    config = replace(config, seed=seed, output_dir=output,
                     training=replace(config.training, episodes=args.episodes,
                                      replay_prefill_cache_path=cache))
    config.validate()
    summary = train(config, device=_select_device(args.device),
                    resume_checkpoint=args.resume_checkpoint,
                    rebuild_replay_prefill_cache=args.rebuild_cache)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
