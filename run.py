import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['train','evaluate'])
    p.add_argument('--method', choices=['TDScaler'], default='TDScaler')
    p.add_argument('--day', type=int, default=1)
    p.add_argument('--episodes', type=int, default=150)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        p.error('Output directory must be empty')
    if a.phase == 'train':
        cmd = ['-m','algorithms.train','--config',str(ROOT/'configs/train.json'),
               '--episodes',str(a.episodes),'--device',a.device,'--output-dir',str(out)]
    else:
        if a.day not in [1,2,3,4,5,6,8]:
            p.error('Supported source days: 1,2,3,4,5,6,8')
        command = next(j for j in json.loads((ROOT/'configs/evaluation_commands.json').read_text()) if j['method']==a.method)
        cmd = command['argv'][:]
        raw = json.loads((ROOT/f'configs/{a.method}.json').read_text())
        for key in ['trace_path','dataset_path']:
            raw['workload'][key] = str((ROOT/'configs'/raw['workload'][key]).resolve())
        raw['performance_profile_path'] = str((ROOT/'configs'/raw['performance_profile_path']).resolve())
        raw['workload'].update(episode_start_offset_s=(a.day-1)*86400, episode_stride_s=0, episode_cycle_count=1)
        raw['output_dir'] = str(out)
        runtime = out.parent / (out.name + '_config.json')
        runtime.parent.mkdir(parents=True,exist_ok=True)
        runtime.write_text(json.dumps(raw,indent=2))
        cmd[cmd.index('--config')+1]=str(runtime)
        cmd[cmd.index('--output-dir')+1]=str(out)
        for flag in ['--checkpoint','--forecaster-checkpoint','--capacity-cache']:
            if flag in cmd:
                cmd[cmd.index(flag)+1]=str((ROOT/cmd[cmd.index(flag)+1]).resolve()) if flag!='--capacity-cache' else str(out.parent/(out.name+'_capacity.json'))
        if '--day' in cmd:cmd[cmd.index('--day')+1]=str(a.day)
        if '--device' in cmd:cmd[cmd.index('--device')+1]=a.device
    subprocess.run([sys.executable,'-u']+cmd,cwd=ROOT,check=True)

if __name__=='__main__':main()
