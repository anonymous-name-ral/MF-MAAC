"""
UAV-LDS training of the 15 MAAC-family models (MAAC, F-MAAC, MAHAC backbones and their MF / MSF variants).

Inputs: with --stack false (the paper setting) the actor sees the agent's current observation (84) and the critic
the current observation and action of every agent (84 + 7). --stack true / --plusplus true are input variants that
stack the last 8 steps (not used in the paper).

Setting: reward_scale 100, tau 1e-3, gamma 0.99, lr 1e-3 / 1e-3, 2 updates per 200 environment steps, batch 1024,
buffer 1e6 steps, hidden 128, 4 attention heads, 1000-step episodes, environment reward unchanged,
Unity engine at time scale 20, quality 1, no graphics, map size 13, 3 buildings, parcel limits 100/100.
Critic modules: utils/critics_<model>.py.
  * --unity_seed (default = --seed) is passed to UnityEnvironment(seed=...)
  * episodes.csv: one row per episode (rewards, deliveries, SR = small + 2 * big, losses)
  * best_model.pt (best 100-episode moving average of the episode reward) and final_model.pt

  python train_maac.py --critic_type mf_t --stack false --worker 21 --run_dir results/<run>
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]    # MAAC/ and lib/ (UnityGymWrapperStats)
from utils.phase0_buffer import InputSpec, RolloutHistory, Phase0ReplayBuffer  # noqa: E402
from algorithms import attention_sac_phase0 as asp  # noqa: E402
from UnityGymWrapperStats import GymEnv  # noqa: E402

# Unity build: $UAVLDS_BUILD, default uav_lds/unity_build/Build_Linux_eval_v2_5uav/Logistics.x86_64
BUILD = Path(os.environ.get('UAVLDS_BUILD', HERE.parents[1] / 'unity_build' / 'Build_Linux_eval_v2_5uav' / 'Logistics.x86_64'))
MODELS = {'mf_p': 'MF-MAAC-P', 'mf_t': 'MF-MAAC-T', 'msf_p': 'MSF-MAAC-P', 'msf_t': 'MSF-MAAC-T',
          'mahac_mf_p': 'MF-MAHAC-P', 'mahac_mf_t': 'MF-MAHAC-T', 'mahac_msf_p': 'MSF-MAHAC-P',
          'mahac_msf_t': 'MSF-MAHAC-T', 'fmaac_mf_p': 'MF-F-MAAC-P', 'fmaac_mf_t': 'MF-F-MAAC-T',
          'fmaac_msf_p': 'MSF-F-MAAC-P', 'fmaac_msf_t': 'MSF-F-MAAC-T',
          'fmaac': 'F-MAAC',   # plain F-MAAC (no MF/MSF), added 2026-10-29 for the no-stack check
          'plain': 'MAAC', 'mahac': 'MAHAC'}   # plain MAAC / MAHAC, added 2026-11-11 (multi-seed)
STACK_K = 8


def str2bool(v):
    if v.lower() in ('1', 'true', 'on', 'yes'):
        return True
    if v.lower() in ('0', 'false', 'off', 'no'):
        return False
    raise argparse.ArgumentTypeError(v)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--critic_type', required=True, choices=list(MODELS))
    p.add_argument('--stack', type=str2bool, default=True, help='on = stacked 8-step inputs; off = single-step inputs (paper)')
    p.add_argument('--plusplus', type=str2bool, default=False,
                   help='on = ++ (stack critic + actor 8 x (obs + prev action) + neighbour means); implies --stack on')
    p.add_argument('--n_episodes', type=int, default=50000)
    p.add_argument('--episode_length', type=int, default=1000)
    p.add_argument('--seed', type=int, default=40)
    p.add_argument('--unity_seed', type=int, default=None, help='Unity RNG seed; default = --seed')
    p.add_argument('--save_best_sr', type=str2bool, default=False, help='also keep best_sr_model.pt')
    p.add_argument('--worker', type=int, required=True, help='unique per concurrent Unity process (port offset)')
    p.add_argument('--run_dir', required=True)
    p.add_argument('--buffer_length', type=int, default=int(1e6))
    p.add_argument('--steps_per_update', type=int, default=200)
    p.add_argument('--num_updates', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--attend_heads', type=int, default=4)
    p.add_argument('--pi_lr', type=float, default=1e-3)
    p.add_argument('--q_lr', type=float, default=1e-3)
    p.add_argument('--tau', type=float, default=1e-3)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--reward_scale', type=float, default=100.0)
    p.add_argument('--use_gpu', type=str2bool, default=torch.cuda.is_available())
    p.add_argument('--wandb_mode', default='disabled', choices=['online', 'offline', 'disabled'])
    p.add_argument('--wandb_project', default='MF-MAAC_UAV-LDS')
    p.add_argument('--wandb_entity', default=None)
    p.add_argument('--wandb_name', default=None, help='default: "<model> v1" (no "stack" in the name)')
    return vars(p.parse_args())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def make_spec(od, ad, n, stack, plusplus=False):
    if plusplus:
        return InputSpec(od, ad, n, 0, actor_stack_k=STACK_K, actor_prev_action=True, actor_mf_input=True,
                         critic_stack_k=STACK_K, critic_prev_action=True)
    k = STACK_K if stack else 1
    return InputSpec(od, ad, n, 0, actor_stack_k=k, actor_prev_action=False, actor_mf_input=False,
                     critic_stack_k=k, critic_prev_action=stack)


def run(cfg):
    set_seed(cfg['seed'])
    run_dir = Path(cfg['run_dir']).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / 'config.json').exists():
        raise SystemExit('run_dir already holds a run: %s' % run_dir)
    os.chdir(run_dir)                                   # Unity writes its CSV logs into the working directory
    if cfg['unity_seed'] is None:
        cfg['unity_seed'] = cfg['seed']

    env = GymEnv(name=str(BUILD), mapsize=13, numbuilding=3, max_smallbox=100, max_bigbox=100,
                 width=480, height=270, timescale=20, quality_level=1, no_graphics=True, worker_id=cfg['worker'],
                 seed=cfg['unity_seed'])
    n = env.nagent
    od = int(env.observation_space[0][0])
    ad = int(env.action_space[0])
    if cfg['plusplus']:
        cfg['stack'] = True
    spec = make_spec(od, ad, n, cfg['stack'], cfg['plusplus'])
    model_name = MODELS[cfg['critic_type']] + ('++' if cfg['plusplus'] else '')
    cfg.update(model=model_name, setting='plusplus' if cfg['plusplus'] else ('stack' if cfg['stack'] else 'original'),
               obs_dim=od, act_dim=ad,
               nagents=n, actor_dim=spec.actor_dim, critic_obs_dim=spec.critic_obs_dim,
               critic_module=asp.CRITIC_MODULES[cfg['critic_type']], stack_k=spec.k,
               actor_stack_k=spec.actor_stack_k, actor_prev_action=spec.actor_prev_action,
               actor_mf_input=spec.actor_mf_input,
               critic_stack_k=spec.critic_stack_k, critic_prev_action=spec.critic_prev_action,
               env='UAV-LDS', build=str(BUILD), engine='A (timescale 20, quality 1, no graphics)')
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    import wandb
    name = cfg['wandb_name'] or '%s v1' % model_name
    wandb.init(entity=cfg['wandb_entity'], project=cfg['wandb_project'], name=name, group=model_name,
               tags=[model_name, cfg['setting']], config=cfg, mode=cfg['wandb_mode'], dir=str(run_dir))
    wandb.define_metric('episode')
    wandb.define_metric('*', step_metric='episode')

    model = asp.AttentionSACPhase0.init_from_spec(
        spec, critic_type=cfg['critic_type'], gamma=cfg['gamma'], tau=cfg['tau'], pi_lr=cfg['pi_lr'],
        q_lr=cfg['q_lr'], reward_scale=cfg['reward_scale'], pol_hidden_dim=cfg['hidden'],
        critic_hidden_dim=cfg['hidden'], attend_heads=cfg['attend_heads'])
    buffer = Phase0ReplayBuffer(cfg['buffer_length'], spec, dtype=np.float16)
    hist = RolloutHistory(spec, 1)
    eye = np.eye(ad, dtype=np.float32)
    print('[stack] %s | setting %s | actor_dim %d critic_obs_dim %d | critic %s | buffer %.2f GB | gpu %s'
          % (model_name, cfg['setting'], spec.actor_dim, spec.critic_obs_dim, cfg['critic_module'],
             buffer.nbytes() / 1e9, cfg['use_gpu']), flush=True)

    csv_f = open(run_dir / 'episodes.csv', 'w', newline='')
    writer = None
    win_rew, win_sr, win_small, win_big = (deque(maxlen=100) for _ in range(4))
    best_rew = best_sr = -float('inf')
    t = 0
    t0 = time.time()
    for ep in range(cfg['n_episodes']):
        obs = np.asarray(env.reset(), dtype=np.float32)          # (n, od)
        hist.reset(0, obs, None)
        model.prep_rollouts(device='cpu')
        ep_rew = np.zeros(n)
        info = {}
        upd = []
        for step in range(cfg['episode_length']):
            a_in, _, _ = hist.inputs()
            with torch.no_grad():
                acts = model.step([torch.from_numpy(x) for x in a_in], explore=True)
            acts = np.stack([a.numpy() for a in acts], axis=1)   # (1, n, ad) one-hot
            ai = acts[0].argmax(axis=1)
            nobs, rew, done, info = env.step(eye[ai].astype(np.int32))   # one-hot over the 7 branches (main_db.py)
            nobs = np.asarray(nobs, dtype=np.float32)
            rew = np.asarray(rew, dtype=np.float32)
            buffer.push(obs[None], acts, rew[None], nobs[None], np.zeros(1, np.uint8),   # no terminal in UAV-LDS
                        np.full(1, step), np.full(1, ep))
            hist.push(0, nobs, eye[ai], None)
            obs = nobs
            ep_rew += rew
            t += 1
            if len(buffer) >= cfg['batch_size'] and t % cfg['steps_per_update'] == 0:
                model.prep_training(device='gpu' if cfg['use_gpu'] else 'cpu')
                for _ in range(cfg['num_updates']):
                    sample = buffer.sample(cfg['batch_size'], to_gpu=cfg['use_gpu'])
                    m = model.update_critic(sample)
                    m.update(model.update_policies(sample))
                    model.update_all_targets()
                    upd.append(m)
                model.prep_rollouts(device='cpu')

        small, big = int(info.get('small_delivered', 0)), int(info.get('big_delivered', 0))
        sr = small + 2 * big
        total = float(ep_rew.sum())
        win_rew.append(total); win_sr.append(sr); win_small.append(small); win_big.append(big)
        log = {'episode': ep + 1, 'env_steps': t,
               'Reward/Episode_Raw': total, 'Reward/Episode_Avg_100': float(np.mean(win_rew)),
               'Success/Small': small, 'Success/Big': big, 'Success/SR': sr,
               'Success/Small_MA100': float(np.mean(win_small)), 'Success/Big_MA100': float(np.mean(win_big)),
               'Success/SR_MA100': float(np.mean(win_sr)), 'Collisions': int(info.get('collisions', 0)),
               'Train/updates': model.niter, 'Time/hours': (time.time() - t0) / 3600.0}
        log.update({'Reward/Agent_%d' % (i + 1): float(r) for i, r in enumerate(ep_rew)})
        if upd:
            for key, nm in (('q_loss', 'Loss/Critic_loss'), ('q_grad_norm', 'Grad_Norm/Critic'),
                            ('pol_loss', 'Loss/Mean_Actor_Loss'), ('entropy', 'Metrics/Mean_Entropy')):
                log[nm] = float(np.mean([u[key] for u in upd]))
        wandb.log(log)
        if writer is None:
            keys = list(log) + [k for k in ('Loss/Critic_loss', 'Grad_Norm/Critic', 'Loss/Mean_Actor_Loss',
                                            'Metrics/Mean_Entropy') if k not in log]
            writer = csv.DictWriter(csv_f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
        writer.writerow(log)
        csv_f.flush()

        extra = {'config': cfg}
        if len(win_rew) == win_rew.maxlen:               # same rule as main_db.py: best reward MA100
            if log['Reward/Episode_Avg_100'] > best_rew:
                best_rew = log['Reward/Episode_Avg_100']
                model.save(run_dir / 'best_model.pt', extra)
                model.prep_rollouts(device='cpu')
            if cfg['save_best_sr'] and log['Success/SR_MA100'] > best_sr:
                best_sr = log['Success/SR_MA100']
                model.save(run_dir / 'best_sr_model.pt', extra)
                model.prep_rollouts(device='cpu')
        if ep % 10 == 0 or ep + 1 == cfg['n_episodes']:
            print('ep %6d/%d | R %.1f (MA100 %.1f) | small %d big %d SR %d (MA100 %.2f) | coll %d | upd %d | %.1fh'
                  % (ep + 1, cfg['n_episodes'], total, log['Reward/Episode_Avg_100'], small, big, sr,
                     log['Success/SR_MA100'], log['Collisions'], model.niter, log['Time/hours']), flush=True)

    model.save(run_dir / 'final_model.pt', {'config': cfg})
    csv_f.close()
    env.close()
    wandb.finish()
    print('[stack] done ->', run_dir, flush=True)


if __name__ == '__main__':
    run(parse())
