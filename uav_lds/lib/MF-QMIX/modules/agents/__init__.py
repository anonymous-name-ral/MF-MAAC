REGISTRY = {}

from .mf_rnn_agent import MeanFieldRNNAgent

REGISTRY["rnn"] = MeanFieldRNNAgent
