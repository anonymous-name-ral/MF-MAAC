# MF-MAAC: Mean-Field Enhanced Attention Critics for Multi-Robot Logistics Transportation

Code and pretrained models for the paper *Mean-Field Enhanced Attention Critics for Multi-Robot Logistics
Transportation*, on two benchmarks: **UAV-LDS** (Unity ML-Agents) and **RWARE** (Multi-Robot Warehouse).

Models: MF-MAAC-P/T, MSF-MAAC-P/T, MF-F-MAAC-P/T, MSF-F-MAAC-P/T, MF-MAHAC-P/T, MSF-MAHAC-P/T (ours);
MAAC, F-MAAC, MAHAC (backbones); QMIX, MF-QMIX, MAPPO, QPLEX, QTRAN (baselines).

## Installation

Linux x86_64, Python 3.8, one environment per benchmark. Install PyTorch for your CUDA / CPU setup first.

```bash
# RWARE
pip install torch==1.9.0
pip install -r requirements-rware.txt

# UAV-LDS
pip install torch==2.4.1
pip install -r requirements-uav_lds.txt
```

The UAV-LDS Unity build (Linux) is included as `uav_lds/unity_build/Build_Linux_eval_v2_5uav.zip` and is unpacked
automatically on the first run.

Pretrained models: because of their size, the trained checkpoints will be provided through Google Drive. To keep the
review double-anonymous, the link will be added here after the paper is accepted. Place the downloaded files at
`pretrained/rware/<model>.pt` and `pretrained/uav_lds/<model>.pt`.

## Usage

Run from `rware/` or `uav_lds/`.

```bash
python train.py --model MF-MAAC-P --seed 40          # training (--model all: every model)
python evaluate.py --model all                       # evaluation of the pretrained models (once released)
python evaluate.py --model MF-MAAC-P --checkpoint results/train/MF-MAAC-P_s40/best_model.pt
python summarize.py                                  # result table from the evaluation outputs
```

`python <script>.py -h` lists the options. UAV-LDS: give concurrent runs different `--worker` values (Unity port).

## Code layout

```
rware/                   uav_lds/
├── train.py             ├── train.py
├── evaluate.py          ├── evaluate.py
├── summarize.py         ├── summarize.py
├── models.py            ├── models.py            model list and training arguments
└── lib/                 └── lib/
    ├── MAAC/                ├── MAAC/                MAAC family: one critic per model in utils/critics_<model>.py
    ├── MAPPO/               ├── MAPPO/               MAPPO
    ├── QMIX/                ├── QMIX/ MF-QMIX/       QMIX and MF-QMIX (RWARE: both in QMIX/)
    ├── QPLEX/ QTRAN/        ├── QPLEX/ QTRAN/        QPLEX, QTRAN
    └── robotic-warehouse/   └── UnityGymWrapperStats.py, uav_eval_env.py   environment
```

`models.py` maps each model name to its critic module (`--critic_type`) or baseline script.
`uav_lds/TUNING.md` summarises how the UAV-LDS baselines were tuned.

## Third-party code

`lib/` contains adapted copies of MAAC, OpenAI Baselines (MIT), MAPPO (on-policy), PyMARL-style QMIX / QPLEX / QTRAN
components, robotic-warehouse (MIT) and the UAV-LDS environment wrapper; their licenses apply to those parts.
