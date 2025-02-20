# source: https://github.com/young-geng/CQL/tree/934b0e8354ca431d6c083c4e3a29df88d4b0a24d
# https://arxiv.org/pdf/2006.04779.pdf
import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal, TanhTransform, TransformedDistribution
import sys
sys.path.insert(0, '/home/ubuntu/NoisyRewardRL/neorl')
sys.path.append('/home/ubuntu/OODOfflineRL/CORL/algorithms/offline/SCQ/offlinerl/algo')
sys.path.append('/home/ubuntu/OODOfflineRL/CORL/algorithms/offline/SCQ/offlinerl/utils')
import neorl
import copy
from gym.spaces import Box
from sklearn.preprocessing import normalize

# VAE imports
from vae import VAE
from torch.distributions import Normal, kl_divergence
from torch.distributions import Distribution

TensorBatch = List[torch.Tensor]
EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class SimHashLSH:
    def __init__(self, k, d):
        self.k = k  # Dimension of binary codes 50
        self.d = d  # Dimension of original state space
        self.A = np.random.randn(k, d)  # Matrix A with Gaussian random entries
        self.hash_table = {}

    def preprocess(self, state):
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        # Example preprocessing function: normalize the state vector
        return normalize(state.reshape(1, -1), norm='l2')

    def compute_hash(self, state):
        
        g_s = self.preprocess(state)
        hash_code = np.sign(np.dot(self.A, g_s.T)).flatten()
        return hash_code.astype(int)

    def update_hash_table(self, state):
        hash_code = tuple(self.compute_hash(state))
        if hash_code not in self.hash_table:
            self.hash_table[hash_code] = 0
        self.hash_table[hash_code] += 1

    def normalize_counts(self):
        max_count = max(self.hash_table.values())
        min_count = min(self.hash_table.values())
        for hash_code in self.hash_table:
            self.hash_table[hash_code] = (self.hash_table[hash_code] - min_count) / (max_count - min_count)

    def get_normalized_count(self, state):
        hash_code = tuple(self.compute_hash(state))
        if hash_code in self.hash_table:
            return self.hash_table[hash_code]
        else:
            return 0.0

@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "NeoRL-SalesPromotionDataset"  # OpenAI gym environment name
    seed: int = 42  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(1)  # How often (time steps) we evaluate
    n_episodes: int = 3  # How many episodes run during evaluation
    max_timesteps: int = int(200)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None # 'model' # None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load

    # CQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    alpha_multiplier: float = 1.0  # Multiplier for alpha in loss
    use_automatic_entropy_tuning: bool = True  # Tune entropy
    backup_entropy: bool = False  # Use backup entropy
    policy_lr: float = 3e-5  # Policy learning rate
    qf_lr: float = 3e-4  # Critics learning rate
    soft_target_update_rate: float = 5e-3  # Target network update rate
    target_update_period: int = 1  # Frequency of target nets updates
    cql_n_actions: int = 10  # Number of sampled actions
    cql_importance_sample: bool = True  # Use importance sampling
    cql_lagrange: bool = False  # Use Lagrange version of CQL
    cql_target_action_gap: float = -1.0  # Action gap
    cql_temp: float = 1.0  # CQL temperature
    cql_alpha: float = 10.0  # Minimal Q weight
    cql_max_target_backup: bool = False  # Use max target backup
    cql_clip_diff_min: float = -np.inf  # Q-function lower loss clipping
    cql_clip_diff_max: float = np.inf  # Q-function upper loss clipping
    orthogonal_init: bool = True  # Orthogonal initialization
    normalize: bool = False  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    q_n_hidden_layers: int = 3  # Number of hidden layers in Q networks
    reward_scale: float = 1.0  # Reward scale for normalization
    reward_bias: float = 0.0  # Reward bias for normalization

    # AntMaze hacks
    bc_steps: int = int(0)  # Number of BC steps at start
    reward_scale: float = 5.0
    reward_bias: float = -1.0
    policy_log_std_multiplier: float = 1.0

    # Wandb logging
    project: str = "CORL"
    group: str = "A2PR"
    name: str = "A2PR-42"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def remove_traj(dataset):
    action = dataset['action']
    # print unique values in the second dimension of action and there count
    print(np.unique(action[:, 1], return_counts=True))
    # replace all the values in the second dimension of action with 0.95 with equal probabilties from [0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95]
    # remove 70% of the values in data_dict where action[1] = 0.95
    indices = np.where(action[:, 1] == 0.95)[0]
    num_to_remove = int(0.95 * len(indices))
    indices_to_remove = np.random.choice(indices, size=num_to_remove, replace=False)
    # print(indices_to_remove)
    # Remove these indices from data_dict
    for key in dataset.keys():
        dataset[key] = np.delete(dataset[key], indices_to_remove, axis=0)
    action = dataset['action']
    print(np.unique(action[:, 1], return_counts=True))
    print(np.unique(action[:, 0], return_counts=True))

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    # Loads Sales Promotion data in d4rl format, i.e. from Dict[str, np.array].
    def load_finrl_dataset(self, data, state_dim):
        k = 50  # Dimension of binary codes
        d = state_dim # Dimension of original state space
        simhash_lsh = SimHashLSH(k, d)
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["obs"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        for i in range(len(data['obs'])):
            data['obs'][i] = np.clip(np.random.normal(data['obs'][i], 10), 0, np.inf)
            # data['reward'][i] = np.random.normal(data['reward'][i], 10)
            simhash_lsh.update_hash_table(data['obs'][i])
        simhash_lsh.normalize_counts()
        self._states[:n_transitions] = self._to_tensor(data["obs"])
        self._actions[:n_transitions] = self._to_tensor(data["action"])
        self._rewards[:n_transitions] = self._to_tensor(data["reward"])
        self._next_states[:n_transitions] = self._to_tensor(data["next_obs"])
        self._dones[:n_transitions] = self._to_tensor(data["done"])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")
        return simhash_lsh

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(
    dataset: Dict,
    env_name: str,
    max_episode_steps: int = 1000,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def init_module_weights(module: torch.nn.Sequential, orthogonal_init: bool = False):
    # Specific orthgonal initialization for inner layers
    # If orthogonal init is off, we do not change default initialization
    if orthogonal_init:
        for submodule in module[:-1]:
            if isinstance(submodule, nn.Linear):
                nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
                nn.init.constant_(submodule.bias, 0.0)

    # Lasy layers should be initialzied differently as well
    if orthogonal_init:
        nn.init.orthogonal_(module[-1].weight, gain=1e-2)
    else:
        nn.init.xavier_uniform_(module[-1].weight, gain=1e-2)

    nn.init.constant_(module[-1].bias, 0.0)


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(
        self, log_std_min: float = -20.0, log_std_max: float = 2.0, no_tanh: bool = False
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(
        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob




class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        orthogonal_init: bool = False,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2 * action_dim),
        )

        init_module_weights(self.base_network)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state, device=device, dtype=torch.float32) #torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        action, _ = self(state)
        action = action # dist.mean if not self.training else dist.sample()
        action = torch.clamp(action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class FullyConnectedQFunction(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        orthogonal_init: bool = False,
        n_hidden_layers: int = 3,
        dropout_prob: float = 0.5  # Default dropout probability
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init
        self.dropout_prob = dropout_prob  # Dropout probability
        
        layers = [
            nn.Linear(observation_dim + action_dim, 256),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        
        # Adding dropout layer
        layers.append(nn.Linear(256, 1))  # Adding linear layer before dropout
        
        self.network = nn.Sequential(*layers)

        init_module_weights(self.network, orthogonal_init)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, use_dropout: bool = False) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        input_tensor = torch.cat([observations, actions], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        
        # Applying dropout if use_dropout is True
        if use_dropout:
            q_values = F.dropout(q_values, p=self.dropout_prob, training=self.training)
        
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values

class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class ContinuousCQL:
    def __init__(
        self,
        critic_1,
        critic_1_optimizer,
        critic_2,
        critic_2_optimizer,
        actor,
        actor_optimizer,
        target_entropy: float,
        discount: float = 0.99,
        alpha_multiplier: float = 1.0,
        use_automatic_entropy_tuning: bool = True,
        backup_entropy: bool = False,
        policy_lr: bool = 3e-4,
        qf_lr: bool = 3e-4,
        soft_target_update_rate: float = 5e-3,
        bc_steps=100000,
        target_update_period: int = 1,
        cql_n_actions: int = 10,
        cql_importance_sample: bool = True,
        cql_lagrange: bool = False,
        cql_target_action_gap: float = -1.0,
        cql_temp: float = 1.0,
        cql_alpha: float = 5.0,
        cql_max_target_backup: bool = False,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        device: str = "cpu",
        vae: Optional[VAE] = None,
        vae_optim: Optional[torch.optim.Optimizer] = None,
        value_net: Optional[nn.Module] = None,
        value_net_optim: Optional[torch.optim.Optimizer] = None,
        advantage_list: Optional[List] = None,
    ):
        super().__init__()

        self.discount = discount
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.backup_entropy = backup_entropy
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.soft_target_update_rate = soft_target_update_rate
        self.bc_steps = bc_steps
        self.target_update_period = target_update_period
        self.cql_n_actions = cql_n_actions
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_alpha = cql_alpha
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self._device = device

        # ExID
        self.use_teacher: bool = False
        self.simhash_lsh = None
        self.lam = 100
        self.teacher_lr = 0.0001
        self.teacher = deepcopy(actor)
        self.teacher.load_state_dict(torch.load('model/bcsalespenv_149.pt')["actor"])
        self.teacher_optimizer = torch.optim.Adam(self.teacher.parameters(), self.teacher_lr) # actor_optimizer
        self.warm_start = 30
        self.teacher_update = 10

        self.total_it = 0

        self.critic_1 = critic_1
        self.critic_2 = critic_2

        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)

        self.actor = actor

        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer

        # VAE Hyperparameters
        self.vae = vae
        self.vae_optim = vae_optim
        self.num = 10

        self.value_net = value_net
        self.value_net_optim = value_net_optim
        self.advantage_list = advantage_list



        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.policy_lr,
            )
        else:
            self.log_alpha = None

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.qf_lr,
        )

        self.total_it = 0

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic_1, self.critic_1, soft_target_update_rate)
        soft_update(self.target_critic_2, self.critic_2, soft_target_update_rate)

    def _alpha_and_alpha_loss(self, observations: torch.Tensor, log_pi: torch.Tensor):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha() * (log_pi + self.target_entropy).detach()
            ).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    def _policy_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        new_actions: torch.Tensor,
        alpha: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        if self.total_it <= self.bc_steps:
            log_probs = self.actor.log_prob(observations, actions)
            policy_loss = (alpha * log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(
                self.critic_1(observations, new_actions),
                self.critic_2(observations, new_actions),
            )
            policy_loss = (alpha * log_pi - q_new_actions).mean()
        return policy_loss

    def _q_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        new_actions : torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        alpha: torch.Tensor,
        log_dict: Dict,
        idd_action_threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.autograd.set_detect_anomaly(True)

        q1_predicted = self.critic_1(observations, actions)
        q2_predicted = self.critic_2(observations, actions)
        v_pred_1 = self.value_net(observations, actions)
        v_pred_2 = self.value_net(next_observations, new_actions)

        value_loss = F.mse_loss(v_pred_1, v_pred_2)

        extended_obs = torch.cat([observations, next_observations])
        extended_acs = torch.cat([actions, new_actions])

        ## OOD Q1
        q1_ood_pred = self.critic_1(extended_obs, extended_acs)

        ## OOD Q2
        q2_ood_pred = self.critic_2(extended_obs, extended_acs)

        if self.cql_max_target_backup:
            new_next_actions, next_log_pi = self.actor(
                next_observations, repeat=self.cql_n_actions
            )
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_critic_1(next_observations, new_next_actions),
                    self.target_critic_2(next_observations, new_next_actions),
                ),
                dim=-1,
            )
            next_log_pi = torch.gather(
                next_log_pi, -1, max_target_indices.unsqueeze(-1)
            ).squeeze(-1)
        else:
            new_next_actions, next_log_pi = self.actor(next_observations)
            target_q_values = torch.min(
                self.target_critic_1(next_observations, new_next_actions),
                self.target_critic_2(next_observations, new_next_actions),
            )

        if self.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        # sample OOD actions
        curr_loss_coeff = self.get_ood_coeff(observations, actions, idd_action_threshold)
        next_loss_coeff = self.get_ood_coeff(next_observations, new_actions, idd_action_threshold)
        loss_coeff = torch.cat([curr_loss_coeff, next_loss_coeff])

        # update lambda
        # lam, total_lam_error = self.update_lambda(extended_obs, loss_coeff, q1_ood_pred, q2_ood_pred)

        target_q_values = target_q_values.unsqueeze(-1)
        td_target = rewards + (1.0 - dones) * self.discount * target_q_values.detach()
        td_target = td_target.squeeze(-1)
        qf1_bellman_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_bellman_loss = F.mse_loss(q2_predicted, td_target.detach())

        qf1_ood_loss = (loss_coeff*q1_ood_pred).mean()
        qf2_ood_loss = (loss_coeff*q2_ood_pred).mean()
        
        qf1_loss = qf1_bellman_loss + qf1_ood_loss
        qf2_loss = qf2_bellman_loss + qf2_ood_loss

        # CQL
        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]
        cql_random_actions = actions.new_empty(
            (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis = self.actor(
            observations, repeat=self.cql_n_actions
        )
        cql_next_actions, cql_next_log_pis = self.actor(
            next_observations, repeat=self.cql_n_actions
        )
        cql_current_actions, cql_current_log_pis = (
            cql_current_actions.detach(),
            cql_current_log_pis.detach(),
        )
        cql_next_actions, cql_next_log_pis = (
            cql_next_actions.detach(),
            cql_next_log_pis.detach(),
        )

        cql_q1_rand = self.critic_1(observations, cql_random_actions)
        cql_q2_rand = self.critic_2(observations, cql_random_actions)
        cql_q1_current_actions = self.critic_1(observations, cql_current_actions)
        cql_q2_current_actions = self.critic_2(observations, cql_current_actions)
        cql_q1_next_actions = self.critic_1(observations, cql_next_actions)
        cql_q2_next_actions = self.critic_2(observations, cql_next_actions)

        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand,
                torch.unsqueeze(q1_predicted, 1),
                cql_q1_next_actions,
                cql_q1_current_actions,
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand,
                torch.unsqueeze(q2_predicted, 1),
                cql_q2_next_actions,
                cql_q2_current_actions,
            ],
            dim=1,
        )
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5**action_dim)
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_actions - cql_next_log_pis.detach(),
                    cql_q1_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q2_next_actions - cql_next_log_pis.detach(),
                    cql_q2_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q2_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
            )
            cql_min_qf1_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf1_diff - self.cql_target_action_gap)
            )
            cql_min_qf2_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf2_diff - self.cql_target_action_gap)
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
            cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha
            alpha_prime_loss = observations.new_tensor(0.0)
            alpha_prime = observations.new_tensor(0.0)

        shape = actions.shape

        state_cof = 0
        # calculate confidence score for each state
        for s in observations:
            score = self.simhash_lsh.get_normalized_count(s)
            state_cof += (1 - score)
        # print(f'action ============== {actions} ========= {actions.shape}')
        # Create a tensor filled with [5, 0.95]
        # this part was used to train the BC
        '''rule_action = torch.Tensor(shape).to(actions.device)
        rule_action.fill_(5)
        rule_action[:, 1] = 0.95'''
        # get action from the rule-based BC policy
        # rule_action = self.teacher(observations)[0] # self.heuristic_policy(observations)
        # print(f'action ============== {rule_action} =========')
        # q1_rule1 = self.critic_1(observations, rule_action)
        # q1_rule2 = self.critic_2(observations, rule_action)
        if self.use_teacher:
            rule_loss = self.calculate_rule_loss(observations, q1_predicted)
            # rule_loss1 = F.mse_loss(q1_predicted, q1_rule1) # self.calculate_teacher_loss(state, current_Qs, ep, steps)
            # rule_loss2 = F.mse_loss(q2_predicted, q1_rule2)
            qf_loss = qf1_loss + qf2_loss + (1 - state_cof) * 0.5 * rule_loss # self.lam * rule_loss #(rule_loss1 + rule_loss2) 
        else:
            # sferr = self.calculate_safety_loss(observations, q1_predicted)
            qf_loss = qf1_loss + qf2_loss # (1 - state_cof) * sferr # + cql_min_qf1_loss + cql_min_qf2_loss

        log_dict.update(
            dict(
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha=alpha.item(),
                average_qf1=q1_predicted.mean().item(),
                average_qf2=q2_predicted.mean().item(),
                average_target_q=target_q_values.mean().item(),
            )
        )

        log_dict.update(
            dict(
                cql_std_q1=cql_std_q1.mean().item(),
                cql_std_q2=cql_std_q2.mean().item(),
                cql_q1_rand=cql_q1_rand.mean().item(),
                cql_q2_rand=cql_q2_rand.mean().item(),
                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                cql_qf1_diff=cql_qf1_diff.mean().item(),
                cql_qf2_diff=cql_qf2_diff.mean().item(),
                cql_q1_current_actions=cql_q1_current_actions.mean().item(),
                cql_q2_current_actions=cql_q2_current_actions.mean().item(),
                cql_q1_next_actions=cql_q1_next_actions.mean().item(),
                cql_q2_next_actions=cql_q2_next_actions.mean().item(),
                alpha_prime_loss=alpha_prime_loss.item(),
                alpha_prime=alpha_prime.item(),
            )
        )

        return qf_loss, alpha_prime, alpha_prime_loss, value_loss

    def check_condition(self, s):
        # Active users with higher order number and average fee spend more and hence can be given lower discount coupon
        order_number = s[0].cpu().numpy()
        avg_fee = s[1].cpu().numpy()
        # reducing order number as there are very less trajectories that exhibit order number > 60 
        if order_number > 10 and avg_fee > 0.8:
            return True
        else:
            return False
    
    def calculate_uncertainity(self, action_rule, action_pred, states_mismatch):
        num_forward_passes = 10
        predictions_pred = []
        predictions_rule = []
        for i in range(num_forward_passes):
            qval_pred = self.critic_1(states_mismatch, action_pred, True)
            qval_rule = self.critic_1(states_mismatch, action_rule, True)
            predictions_pred.append(torch.mean(qval_pred))
            predictions_rule.append(torch.mean(qval_rule))
        # Covert predictions to tensors
        print(f'predictions_pred =========== {predictions_pred}')
        predictions_pred = torch.tensor(predictions_pred)
        predictions_rule = torch.tensor(predictions_rule)
        # Calculate the variance
        var_student = torch.var(predictions_pred)
        var_teacher = torch.var(predictions_rule)
        return torch.mean(var_student), torch.mean(var_teacher)

    def teacher_learn(self, states):
        logits_teacher = self.teacher(states)[0]
        logits_student = self.actor(states)[0]
        criterion = nn.CrossEntropyLoss()
        teacher_loss = criterion(logits_student, logits_teacher)
        self.teacher_optimizer.zero_grad()
        teacher_loss.backward()
        self.teacher_optimizer.step()
        return teacher_loss

    def calculate_safety_loss(self, observations, q1_predicted):
        action_rule = []
        action_pred = []
        observations_copy = copy.deepcopy(observations)
        for s in observations_copy:
            predicted_action = self.actor(s)[0]
            # get action from the rule-based BC policy
            rule_action = self.teacher(s)[0]
            action_rule.append(rule_action)
            action_pred.append(predicted_action)
        action_rule = torch.stack([torch.tensor(i) for i in action_rule]).to(observations.device)
        action_pred = torch.stack([torch.tensor(i) for i in action_pred]).to(observations.device)
        q1_rule1 = self.critic_1(observations, action_rule)
        q1_pred1 = self.critic_1(observations, action_pred)
        return F.mse_loss(q1_pred1, q1_rule1)
        

        
    # This function calculates the teacher loss for EXID
    def calculate_rule_loss(self, observations, q1_predicted):
        action_rule = []
        action_pred = []
        states_mismatch = []
        observations_copy = copy.deepcopy(observations)
        rule_error = F.mse_loss(q1_predicted, q1_predicted)
        for s in observations_copy:
            condition = self.check_condition(s)
            if condition:
                # get predicted action from the original policy
                predicted_action = self.actor(s)[0]
                # get action from the rule-based BC policy
                rule_action = self.teacher(s)[0]
                if(predicted_action[0]!=rule_action[0] and predicted_action[1]!=rule_action[1]):
                    states_mismatch.append(s.cpu().data.numpy())
                    action_rule.append(rule_action)
                    action_pred.append(predicted_action)

        print(f'len states mismatch ---------- {len(states_mismatch)}')
        if len(states_mismatch) > 0:
            action_rule = torch.stack([torch.tensor(i) for i in action_rule]).to(observations.device)
            action_pred = torch.stack([torch.tensor(i) for i in action_pred]).to(observations.device)
            states_mismatch = torch.stack([torch.tensor(i) for i in states_mismatch]).to(observations.device)
            # get the predicted and the rule loss
            q1_rule1 = self.critic_1(states_mismatch, action_rule)
            q2_rule2 = self.critic_2(states_mismatch, action_rule)
            q1_pred1 = self.critic_1(states_mismatch, action_pred)
            q2_pred2 = self.critic_2(states_mismatch, action_pred)

            # Code block for warm start and uncertainity update
            if self.total_it > self.warm_start:
                rule_error = F.mse_loss(q1_pred1, q1_rule1) + F.mse_loss(q2_pred2, q2_rule2)
            else:
                if torch.mean(q1_rule1) < torch.mean(q1_pred1):
                    rule_error = F.mse_loss(q1_predicted, q1_predicted) # F.mse_loss(q1_pred1, q1_rule1) + F.mse_loss(q2_pred2, q2_rule2)
                    if self.total_it % self.teacher_update ==0:
                        var_student, var_teacher = self.calculate_uncertainity(action_rule,action_pred,states_mismatch)
                        if var_student < var_teacher:
                            teacher_loss = self.teacher_learn(states_mismatch)
                            print(f'--------Training teacher-------------{teacher_loss}----------{var_student}------- {var_teacher}')
                            rule_error = F.mse_loss(q1_rule1, q1_rule1)
                        else:
                            rule_error = F.mse_loss(q1_pred1, q1_rule1) + F.mse_loss(q2_pred2, q2_rule2)
                else:
                    rule_error = F.mse_loss(q1_pred1, q1_rule1) + F.mse_loss(q2_pred2, q2_rule2)

        return rule_error

    # This is to calculate the actions for the rule-based policy to be replaced by BC network
    def heuristic_policy(self, state: np.ndarray) -> np.ndarray:
         action_list = []
         action_space = Box(low=np.array([0.0, 0.6]), high=np.array([5., 1.0]), shape=(2,), dtype=np.float32)
         for i in range(state.shape[0]):
            order_number = state[i,0].cpu().numpy()
            avg_fee = state[i,1].cpu().numpy()
            # Active users with higher order number and average fee spend more and hence can be given lower discount coupon
            if order_number > 60 and avg_fee > 0.8:
                action = [5,0.95]
            else:
                # take a random action 
                action = action_space.sample()
            action_list.append(action)
         action_list = [torch.tensor(i) for i in action_list]  # Convert each array to a tensor
         action_tensor = torch.stack(action_list).to(state.device)  # Stack them together
         return action_tensor

    def update_vae(self, obs, actions):
        # train vae
        vae_dist, _action = self.vae(obs, actions)
        kl_loss = kl_divergence(vae_dist, Normal(0, 1)).sum(dim=-1).mean()
        recon_loss = ((actions - _action) ** 2).sum(dim=-1).mean()
        vae_loss = 0.5 * kl_loss + recon_loss

        self.vae_optim.zero_grad()
        vae_loss.backward()
        self.vae_optim.step()

    def get_ood_coeff(self, obs, actions, idd_action_threshold):
        with torch.no_grad():
            ood_action_dist = self.vae.calc_dist(obs, actions) # calculate distance from the dataset
            ood_idx = torch.where(ood_action_dist>idd_action_threshold)[0]
            coeff = torch.zeros_like(ood_action_dist, requires_grad=False)
            coeff[ood_idx] = 1.0

        return coeff


    

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        self.total_it += 1

        self.update_vae(observations, actions)
        with torch.no_grad():
            idd_action_threshold = self.vae.calc_dist(observations, actions).mean()

        new_actions, log_pi = self.actor(observations.clone())

        alpha, alpha_loss = self._alpha_and_alpha_loss(observations, log_pi)

        """ Policy loss """
        policy_loss = self._policy_loss(
            observations, actions, new_actions, alpha, log_pi
        )

        log_dict = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
        )

        """ Q function loss """
        qf_loss, alpha_prime, alpha_prime_loss, value_loss = self._q_loss(
            observations, actions, new_actions.detach(), next_observations, rewards, dones, alpha, log_dict, idd_action_threshold
        )

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        self.value_net_optim.zero_grad()
        value_loss.backward()
        self.value_net_optim.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.critic_1.load_state_dict(state_dict=state_dict["critic1"])
        self.critic_2.load_state_dict(state_dict=state_dict["critic2"])

        self.target_critic_1.load_state_dict(state_dict=state_dict["critic1_target"])
        self.target_critic_2.load_state_dict(state_dict=state_dict["critic2_target"])

        self.critic_1_optimizer.load_state_dict(
            state_dict=state_dict["critic_1_optimizer"]
        )
        self.critic_2_optimizer.load_state_dict(
            state_dict=state_dict["critic_2_optimizer"]
        )
        self.actor_optimizer.load_state_dict(state_dict=state_dict["actor_optim"])

        self.log_alpha = state_dict["sac_log_alpha"]
        self.alpha_optimizer.load_state_dict(
            state_dict=state_dict["sac_log_alpha_optim"]
        )

        self.log_alpha_prime = state_dict["cql_log_alpha"]
        self.alpha_prime_optimizer.load_state_dict(
            state_dict=state_dict["cql_log_alpha_optim"]
        )
        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    env = neorl.make("sp")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # dataset = d4rl.qlearning_dataset(env)

    if config.normalize_reward:
        modify_reward(
            dataset,
            config.env,
            reward_scale=config.reward_scale,
            reward_bias=config.reward_bias,
        )

    dataset = np.load('/home/ubuntu/OODOfflineRL/CORL/algorithms/offline/data/sp-v0-10000-train.npz')
    # take first three users for testing
    data_dict = {name: dataset[name][:1000] for name in dataset.files} 

    # remove 70% trajectory from the dataset with 0.95 cupon to reduce bias
    remove_traj(data_dict)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    '''dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )'''
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    simhashls = replay_buffer.load_finrl_dataset(data_dict, state_dim)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    critic_1 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)
    critic_2 = FullyConnectedQFunction(state_dim, action_dim, config.orthogonal_init).to(
        config.device
    )
    critic_1_optimizer = torch.optim.Adam(list(critic_1.parameters()), config.qf_lr)
    critic_2_optimizer = torch.optim.Adam(list(critic_2.parameters()), config.qf_lr)
    vae = VAE(state_dim, action_dim, 750, 1, max_action).to(config.device)
    vae_optim = torch.optim.Adam(vae.parameters(), lr=0.003)

    # A2PR hyperparameters
    value_net = FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)

    value_net_optimizer = torch.optim.Adam(value_net.parameters(), lr=3e-4)
    advantage_list = []


    actor = TanhGaussianPolicy(
        state_dim,
        action_dim,
        max_action,
        log_std_multiplier=config.policy_log_std_multiplier,
        orthogonal_init=config.orthogonal_init,
    ).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), config.policy_lr)

    kwargs = {
        "critic_1": critic_1,
        "critic_2": critic_2,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2_optimizer": critic_2_optimizer,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        # CQL
        "target_entropy": -np.prod(env.action_space.shape).item(),
        "alpha_multiplier": config.alpha_multiplier,
        "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
        "backup_entropy": config.backup_entropy,
        "policy_lr": config.policy_lr,
        "qf_lr": config.qf_lr,
        "bc_steps": config.bc_steps,
        "target_update_period": config.target_update_period,
        "cql_n_actions": config.cql_n_actions,
        "cql_importance_sample": config.cql_importance_sample,
        "cql_lagrange": config.cql_lagrange,
        "cql_target_action_gap": config.cql_target_action_gap,
        "cql_temp": config.cql_temp,
        "cql_alpha": config.cql_alpha,
        "cql_max_target_backup": config.cql_max_target_backup,
        "cql_clip_diff_min": config.cql_clip_diff_min,
        "cql_clip_diff_max": config.cql_clip_diff_max,
        "vae" : vae,
        "vae_optim" : vae_optim,
        "value_net" : value_net,
        "value_net_optim" : value_net_optimizer,
        "advantage_list" : advantage_list
    }

    print("---------------------------------------")
    print(f"Training CQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ContinuousCQL(**kwargs)
    trainer.simhash_lsh = simhashls

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            # normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            # evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {eval_score:.3f}"
            )
            print("---------------------------------------")
            if config.checkpoints_path:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"bcsalespenv_{t}.pt"),
                )
            wandb.log(
                 {"slaespromotion_score": eval_score}, step=trainer.total_it
            )


if __name__ == "__main__":
    train()
