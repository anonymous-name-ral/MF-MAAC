"""
Train RWARE models (Table V setting: rware-small-5ag-twostage-v2, 40,000 episodes x 500 steps, beta = 0.1).

  python train.py --model MF-MAAC-P --seed 40
  python train.py --model MF-MAAC-P MAAC --seed 40 41 42 --out results/train
  python train.py --model all --seed 40 --dry_run          # print the commands only

Each run writes <out>/<model>_s<seed>/: config.json, train_log.csv (one row per episode),
best_model.pt (best 100-episode moving average of the training team reward) and model.pt (final).
Runs are executed one after another; start several processes to run them in parallel.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import models as M


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', nargs='+', required=True, help='model names (see models.py) or "all"')
    p.add_argument('--seed', nargs='+', type=int, default=[40])
    p.add_argument('--episodes', type=int, default=M.N_EPISODES)
    p.add_argument('--out', default='results/train', help='parent directory of the run directories')
    p.add_argument('--wandb_mode', default='disabled', choices=['online', 'offline', 'disabled'],
                   help='MAPPO and value-decomposition runs only')
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()
    names = M.MODELS if a.model == ['all'] else a.model
    for m in names:
        if m not in M.MODELS:
            p.error('unknown model %s; choose from %s' % (m, ', '.join(M.MODELS)))
    env = dict(os.environ, PYTHONPATH=M.python_path())
    for m in names:
        for s in a.seed:
            run_dir = (Path(a.out) / ('%s_s%d' % (m, s))).resolve()
            cwd, argv = M.train_command(m, s, run_dir, a.episodes, a.wandb_mode)
            cmd = [sys.executable] + argv
            print('[train] %s seed %d -> %s' % (m, s, run_dir), flush=True)
            if a.dry_run:
                print('  cd %s && PYTHONPATH=%s %s' % (cwd, env['PYTHONPATH'], ' '.join(cmd)))
                continue
            if (run_dir / 'config.json').exists():
                print('  skipped: %s already holds a run' % run_dir)
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / 'train.log', 'w') as log:
                r = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
            print('  exit %d (log: %s)' % (r.returncode, run_dir / 'train.log'), flush=True)
            if r.returncode:
                sys.exit(r.returncode)


if __name__ == '__main__':
    main()
