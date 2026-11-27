"""
RWARE model registry: the 20 models of Table V, their training command and their evaluation setting.

kind   : 'maac' (MAAC-family attention critics, lib/MAAC/train_maac.py), 'mappo' (lib/train_mappo.py),
         'pymarl' (value-decomposition baselines, lib/train_pymarl_baseline.py)
mode   : evaluation policy used in the paper (stochastic actors for the actor-critic models, greedy for the
         value-decomposition models)
Hyper-parameters are fixed inside the training scripts; the extra arguments below are the per-model settings
used for the paper runs.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE / 'lib'
PRETRAINED = HERE.parent / 'pretrained' / 'rware'
ENV_ID = 'rware-small-5ag-twostage-v2'
REWARD_BETA = 0.1          # weight of the shared team term in the training reward
N_EPISODES = 40000         # training episodes (500 steps each)

# model -> critic type of lib/MAAC/train_maac.py
MAAC_FAMILY = {
    'MF-MAAC-P': 'mf_p', 'MF-MAAC-T': 'mf_t', 'MSF-MAAC-P': 'msf_p', 'MSF-MAAC-T': 'msf_t',
    'MF-F-MAAC-P': 'fmaac_mf_p', 'MF-F-MAAC-T': 'fmaac_mf_t', 'MSF-F-MAAC-P': 'fmaac_msf_p',
    'MSF-F-MAAC-T': 'fmaac_msf_t',
    'MF-MAHAC-P': 'mahac_mf_p', 'MF-MAHAC-T': 'mahac_mf_t', 'MSF-MAHAC-P': 'mahac_msf_p',
    'MSF-MAHAC-T': 'mahac_msf_t',
    'MAAC': 'plain', 'F-MAAC': 'fmaac', 'MAHAC': 'mahac',
}
# model -> (algo, extra arguments of lib/train_pymarl_baseline.py)
PYMARL = {
    'QMIX': ('qmix', ['--rnn_hidden_dim', '128', '--mixing_embed_dim', '32', '--hypernet_embed', '128',
                      '--epsilon_anneal_time', '800000']),
    'MF-QMIX': ('mfqmix', ['--rnn_hidden_dim', '128', '--mixing_embed_dim', '32', '--hypernet_embed', '128',
                           '--epsilon_anneal_time', '800000']),
    'QPLEX': ('qplex', ['--rnn_hidden_dim', '64', '--mixing_embed_dim', '32', '--hypernet_embed', '64',
                        '--epsilon_anneal_time', '2000000', '--obs_agent_id', 'off', '--greedy_eval_every', '0']),
    'QTRAN': ('qtran', ['--obs_agent_id', 'off', '--greedy_eval_every', '0']),
}
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


def train_command(model, seed, run_dir, n_episodes=N_EPISODES, wandb_mode='disabled'):
    """-> (working directory, argv without the python executable)"""
    k = kind(model)
    if k == 'maac':
        return LIB / 'MAAC', ['train_maac.py', '--critic_type', MAAC_FAMILY[model], '--seed', str(seed),
                              '--reward_beta', str(REWARD_BETA), '--n_episodes', str(n_episodes),
                              '--run_dir', str(run_dir)]
    common = ['--reward_mode', 'custom', '--reward_beta', str(REWARD_BETA), '--seed', str(seed),
              '--n_episodes', str(n_episodes), '--eval_episodes', '0', '--save_best_sr', '0',
              '--wandb_mode', wandb_mode, '--run_dir', str(run_dir)]
    if k == 'mappo':
        return LIB, ['train_mappo.py'] + common
    algo, extra = PYMARL[model]
    return LIB, ['train_pymarl_baseline.py', '--algo', algo] + extra + common


def python_path():
    return '%s:%s' % (LIB / 'MAAC' / 'baselines-master', LIB / 'robotic-warehouse')
