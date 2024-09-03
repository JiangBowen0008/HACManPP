import numpy as np
import torch as th
import torch.jit as jit
from torch.nn import functional as F
from stable_baselines3.td3.td3 import TD3
from stable_baselines3.common.preprocessing import preprocess_obs as pobs
from stable_baselines3.common.preprocessing import get_action_dim
from typing import Any, Dict, List, Optional, Type, Union, Tuple
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv
import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import ContinuousCritic, BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.td3.policies import TD3Policy, Actor


class MultiActor(BasePolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        num_primitives: int = None,
        squash_output: bool = True
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=squash_output,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        
        action_dim = get_action_dim(self.action_space)
        self.num_primitives = num_primitives
        
        # Assume action_dim is divisible by 2 for simplicity
        mus = []
        for i in range(num_primitives):
            mus.append(nn.Sequential(*create_mlp(
                features_dim, action_dim, net_arch, activation_fn, squash_output=squash_output)))
        self.mus = nn.ModuleList(mus)
    
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        prim_idx = obs['prim_idx'].flatten().long()
        bs = prim_idx.shape[0]

        # Fork the computation for parallel processing
        futures = [jit.fork(mu, features) for mu in self.mus]
        outputs = [jit.wait(future) for future in futures]
        outputs = th.stack(outputs, dim=1)
        outputs = outputs[th.arange(bs), prim_idx]

        # Mask out the actions that are not used
        prim_param_dims = obs['prim_param_dims']
        prim_param_dims = prim_param_dims[th.arange(bs), prim_idx]

        # Vectorized mask creation
        max_dim = outputs.shape[-1]
        cumulative_dims = th.arange(max_dim).unsqueeze(0).repeat(bs, 1).to(outputs.device)
        mask = cumulative_dims < prim_param_dims.unsqueeze(1)
        outputs = outputs * mask.float()

        return outputs

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(observation)

class MultiContinuousCritic(ContinuousCritic):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        num_primitives: int = None,
    ):
        self.num_primitives = num_primitives
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )


    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        prim_idx = obs['prim_idx'].flatten().long().detach()
        bs = prim_idx.shape[0]
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        
        qvalue_input = th.cat([features, actions], dim=1)
        futures = [jit.fork(q_net, qvalue_input) for q_net in self.q_networks]
        outputs = [jit.wait(future) for future in futures]
        outputs = th.stack(outputs, dim=1)                      # bs, n * n_critics
        
        all_outputs = []     # all_outputs collect the outputs from different primitives
        n_critics_per_primitive = self.n_critics // self.num_primitives
        for i in range(n_critics_per_primitive):
            all_outputs.append(
                outputs[th.arange(bs), prim_idx + i * self.num_primitives])
        return tuple(all_outputs)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        prim_idx = obs['prim_idx'].flatten().long().detach()
        bs = prim_idx.shape[0]
        with th.no_grad():
            features = self.extract_features(obs)
        
        qvalue_input = th.cat([features, actions], dim=1)
        futures = [jit.fork(q_net, qvalue_input) for q_net in self.q_networks[:self.num_primitives]]
        outputs = [jit.wait(future) for future in futures]
        outputs = th.stack(outputs, dim=1)                      # bs, n
        outputs = outputs[th.arange(bs), prim_idx]

        return outputs

class MultiTD3Policy(TD3Policy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        num_primitives: int = None,
    ):
        self.num_primitives = num_primitives
        super().__init__(
            observation_space,
            action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=num_primitives*n_critics, # n_critics,
            share_features_extractor=share_features_extractor,         
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> MultiActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor) 
        return MultiActor(num_primitives=self.num_primitives, **actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> MultiContinuousCritic:        
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return MultiContinuousCritic(num_primitives=self.num_primitives, **critic_kwargs).to(self.device)

class MultiTD3(TD3):
  
    def get_next_q_values(self, replay_data):
        obs_tensor = replay_data.next_observations
        prim_idx = obs_tensor['prim_idx']

        # Select action according to policy and add clipped noise
        noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
        noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)

        next_actions = self.actor_target(replay_data.next_observations)
        next_actions = (next_actions + noise).clamp(-1, 1)

        # Compute the next Q-values: min over all critics targets
        next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
        if self.mean_q:
            next_q_values = th.mean(next_q_values, dim=1, keepdim=True)
        else:
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
        
        return next_q_values
    
    def get_actor_loss(self, replay_data):
        actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
        return actor_loss
    

    
