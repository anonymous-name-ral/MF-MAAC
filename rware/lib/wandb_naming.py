"""
wandb run naming: '<model> v<N> | <reward_mode>' (e.g. 'QPLEX v2 | custom').
N = per-model version counter stored in results/wandb_versions.json (file-locked),
incremented only for real (online) runs so smoke tests don't consume versions.
"""
import fcntl
import json
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent / 'results' / 'wandb_versions.json'


def run_name(model, reward_mode, consume=True):
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        txt = f.read()
        reg = json.loads(txt) if txt.strip() else {}
        version = reg.get(model, 0) + 1
        if consume:
            reg[model] = version
            f.seek(0)
            f.truncate()
            json.dump(reg, f, indent=2, ensure_ascii=False)
    return '%s v%d | %s' % (model, version, reward_mode), version
