"""
UAV-LDS model registry: the 20 models of Table V, their training command and their evaluation setting.

kind   : 'maac' (MAAC-family attention critics, lib/MAAC/train_maac.py with --stack false = single-step inputs),
         'mappo' (lib/MAPPO/train_mappo.py), 'pymarl' (value-decomposition baselines: lib/train_pymarl_baseline.py,
         lib/MF-QMIX/train_mf_qmix.py)
mode   : evaluation policy used in the paper (stochastic actors for the actor-critic models, greedy for the
         value-decomposition models)
Hyper-parameters are fixed inside the training scripts; the extra arguments below are the per-model settings
used for the paper runs (see TUNING.md for how the baseline settings were chosen).
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE / 'lib'
PRETRAINED = HERE.parent / 'pretrained' / 'uav_lds'
DEFAULT_BUILD = HERE / 'unity_build' / 'Build_Linux_eval_v2_5uav' / 'Logistics.x86_64'
N_EPISODES = 50000         # training episodes (1000 steps each)

# model -> critic type of lib/MAAC/train_maac.py
MAAC_FAMILY = {
    'MF-MAAC-P': 'mf_p', 'MF-MAAC-T': 'mf_t', 'MSF-MAAC-P': 'msf_p', 'MSF-MAAC-T': 'msf_t',
    'MF-F-MAAC-P': 'fmaac_mf_p', 'MF-F-MAAC-T': 'fmaac_mf_t', 'MSF-F-MAAC-P': 'fmaac_msf_p',
    'MSF-F-MAAC-T': 'fmaac_msf_t',
    'MF-MAHAC-P': 'mahac_mf_p', 'MF-MAHAC-T': 'mahac_mf_t', 'MSF-MAHAC-P': 'mahac_msf_p',
    'MSF-MAHAC-T': 'mahac_msf_t',
    'MAAC': 'plain', 'F-MAAC': 'fmaac', 'MAHAC': 'mahac',
}
QX = {'QMIX': 'qmix', 'QTRAN': 'qtran', 'QPLEX': 'qplex'}
PYMARL = dict(QX, **{'MF-QMIX': 'mfqmix'})
# module directory (under lib/) of the agent network of each value-decomposition model
PYMARL_DIR = {'QMIX': 'QMIX', 'MF-QMIX': 'MF-QMIX', 'QPLEX': 'QPLEX', 'QTRAN': 'QTRAN'}
# Table V order
MODELS = list(MAAC_FAMILY) + ['QMIX', 'MF-QMIX', 'MAPPO', 'QPLEX', 'QTRAN']
OURS = MODELS[:12]


def kind(model):
    if model in MAAC_FAMILY:
        return 'maac'
    if model in PYMARL:
        return 'pymarl'
    if model == 'MAPPO':
        return 'mappo'
    raise KeyError(model)


def eval_mode(model):
    return 'deterministic' if kind(model) == 'pymarl' else 'stochastic'


def train_command(model, seed, worker, run_dir, n_episodes=N_EPISODES, wandb_mode='disabled'):
    """-> argv without the python executable (absolute script paths)"""
    common = ['--n_episodes', str(n_episodes), '--seed', str(seed), '--worker', str(worker), '--run_dir', str(run_dir)]
    if model in MAAC_FAMILY:      # single-step inputs (actor 84, critic 84 + 7)
        return [str(LIB / 'MAAC' / 'train_maac.py'), '--critic_type', MAAC_FAMILY[model], '--stack', 'false',
                '--wandb_mode', wandb_mode] + common
    if model in QX:               # epsilon annealed over 10M steps, normalised global state, RNN 128
        return [str(LIB / 'train_pymarl_baseline.py'), '--algo', QX[model], '--eps_anneal', '10000000',
                '--state_norm', '--wandb_mode', wandb_mode] + common
    if model == 'MF-QMIX':        # epsilon annealed over 5M steps, normalised global state, no reward scaling, RNN 128
        return [str(LIB / 'MF-QMIX' / 'train_mf_qmix.py'), '--eps_anneal', '5000000', '--reward_scale', '1.0',
                '--rnn_hidden_dim', '128', '--state_norm', '--wandb_mode', wandb_mode] + common
    if model == 'MAPPO':
        return [str(LIB / 'MAPPO' / 'train_mappo.py'), '--save_every', '1000000000'] + common
    raise KeyError(model)


def build_path(arg=None):
    """Unity executable; the bundled build is unpacked from unity_build/<build>.zip on first use."""
    path = Path(arg or os.environ.get('UAVLDS_BUILD') or DEFAULT_BUILD).resolve()
    archive = DEFAULT_BUILD.parent.with_suffix('.zip')
    if not path.exists() and path == DEFAULT_BUILD.resolve() and archive.exists():
        import zipfile
        print('unpacking %s' % archive, flush=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(archive.parent)
        path.chmod(0o755)
    return path
