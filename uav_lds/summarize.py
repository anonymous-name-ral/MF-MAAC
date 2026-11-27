"""
Build the UAV-LDS result table from the evaluation outputs of evaluate.py.

  python summarize.py                          # reads results/eval/*.json
  python summarize.py --eval_dir results/eval --csv results/table_uav_lds.csv

Per model and training seed one checkpoint is kept: when both best_model.pt and final_model.pt of a run were
evaluated, the one with the higher mean AS is used. Across seeds: mean of the per-seed
means +- sample standard deviation (n - 1).
Columns: AS (mean +- std), mean small deliveries, mean big deliveries, number of seeds.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import models as M


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--eval_dir', default='results/eval')
    p.add_argument('--csv', default=None, help='also write the table as CSV')
    a = p.parse_args()
    per_seed = defaultdict(dict)                      # model -> seed -> summary
    for f in sorted(Path(a.eval_dir).glob('*.json')):
        s = json.load(open(f))['summary']
        if s.get('mode') != M.eval_mode(s['model']):
            continue                                  # keep the paper policy of each model
        key = s.get('train_seed')
        old = per_seed[s['model']].get(key)
        if old is None or s['AS_mean'] > old['AS_mean']:
            per_seed[s['model']][key] = s
    if not per_seed:
        raise SystemExit('no evaluation outputs in %s' % a.eval_dir)

    rows = []
    for m in M.MODELS:
        if m not in per_seed:
            continue
        ss = list(per_seed[m].values())
        x = lambda k: np.array([s[k] for s in ss], dtype=float)
        std = x('AS_mean').std(ddof=1) if len(ss) > 1 else x('AS_std')[0]
        rows.append(dict(model=m, group='Ours' if m in M.OURS else 'Backbones and baselines',
                         AS_mean=x('AS_mean').mean(), AS_std=std, small=x('small_mean').mean(),
                         big=x('big_mean').mean(), n_seeds=len(ss),
                         checkpoints=' '.join('s%s:%s' % (s.get('train_seed'), s['tag']) for s in ss)))
    best = max(r['AS_mean'] for r in rows)
    print('| Model | AS | Small | Big | Seeds |')
    print('|---|---|---|---|---|')
    group = None
    for r in rows:
        if r['group'] != group:
            group = r['group']
            print('| *%s* | | | | |' % group)
        as_ = '%.2f ± %.2f' % (r['AS_mean'], r['AS_std'])
        print('| %s | %s | %.2f | %.2f | %d |' % (r['model'], '**%s**' % as_ if r['AS_mean'] == best else as_,
                                                  r['small'], r['big'], r['n_seeds']))
    print('\nAS: one seed -> mean +- std over the evaluation episodes; several seeds -> mean +- std of the seed means.')
    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print('saved', a.csv)


if __name__ == '__main__':
    main()
