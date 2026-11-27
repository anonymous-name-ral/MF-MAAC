"""
Train UAV-LDS models (Table V setting: Unity build v2 with 5 UAVs, 50,000 episodes x 1000 steps,
environment reward unchanged, Unity engine at time scale 20 / quality 1 / no graphics).

  python train.py --model MF-MAAC-P --seed 40
  python train.py --model MF-MAAC-P MAAC --seed 40 41 42 --worker 10
  python train.py --model all --seed 40 --dry_run          # print the commands only

Each run writes <out>/<model>_s<seed>/: config.json, episodes.csv (one row per episode), train.log,
best_model.pt (best 100-episode moving average of the episode reward) and final_model.pt.
Every Unity process needs its own --worker id (port offset); runs started by one call use worker, worker + 1, ...
Runs are executed one after another; start several processes with different --worker values to run in parallel.
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
    p.add_argument('--worker', type=int, default=0, help='first Unity worker id (port offset)')
    p.add_argument('--build', default=None, help='Unity executable (default: unity_build/Build_Linux_eval_v2_5uav/'
                                                 'Logistics.x86_64 or $UAVLDS_BUILD)')
    p.add_argument('--out', default='results/train', help='parent directory of the run directories')
    p.add_argument('--wandb_mode', default='disabled', choices=['online', 'offline', 'disabled'],
                   help='MAAC family and value-decomposition runs')
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()
    names = M.MODELS if a.model == ['all'] else a.model
    for m in names:
        if m not in M.MODELS:
            p.error('unknown model %s; choose from %s' % (m, ', '.join(M.MODELS)))
    build = M.build_path(a.build)
    if not build.exists() and not a.dry_run:
        p.error('Unity build not found: %s (see README, "UAV-LDS environment")' % build)
    env = dict(os.environ, PYTHONPATH='', UAVLDS_BUILD=str(build))
    env.pop('DISPLAY', None)          # the build runs headless; a display can crash it on some drivers
    worker = a.worker
    for m in names:
        for s in a.seed:
            run_dir = (Path(a.out) / ('%s_s%d' % (m, s))).resolve()
            cmd = [sys.executable] + M.train_command(m, s, worker, run_dir, a.episodes, a.wandb_mode)
            print('[train] %s seed %d (worker %d) -> %s' % (m, s, worker, run_dir), flush=True)
            if a.dry_run:
                print('  cd %s && UAVLDS_BUILD=%s %s' % (run_dir, build, ' '.join(cmd)))
                worker += 1
                continue
            if (run_dir / 'config.json').exists():
                print('  skipped: %s already holds a run' % run_dir)
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / 'train.log', 'ab') as log:
                r = subprocess.run(cmd, cwd=str(run_dir), env=env, stdout=log, stderr=subprocess.STDOUT)
            print('  exit %d (log: %s)' % (r.returncode, run_dir / 'train.log'), flush=True)
            if r.returncode:
                sys.exit(r.returncode)
            worker += 1


if __name__ == '__main__':
    main()
