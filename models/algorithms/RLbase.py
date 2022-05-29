import random
import json
import random
from os.path import join
from pathlib import Path
from argparse import Namespace

import torch
from torch import nn
import os
import gym

from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import AttrDict
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.algorithms.appo.model_utils import register_custom_encoder
from sample_factory.algorithms.appo.model_utils import get_obs_shape, EncoderBase, ResBlock, nonlinearity
from sample_factory.algorithms.utils.pytorch_utils import calc_num_elements
from sample_factory.algorithms.utils.algo_utils import EXTRA_EPISODIC_STATS_PROCESSING, EXTRA_PER_POLICY_SUMMARIES
from pydantic import Extra, BaseModel, validator
import multiprocessing

from pogema import GridConfig
from pogema.wrappers.multi_time_limit import MultiTimeLimit
from pogema.wrappers.metrics import MetricsWrapper

from algorithms.rele import APPOHolder

class Model:
    def __init__(self):
        self.appo = APPOHolder('./weights/c164')

    def act(self, obs, done, positions_xy, targets_xy):
        actions = self.appo.act(obs)
        return actions

