import numpy as np
import torch
from stable_baselines3.common.preprocessing import preprocess_obs as pobs

from hacman.utils.primitive_utils import GroundingTypes, sample_prim_idx
from hacman.algos.utils import top_k_top_p_filtering

class LocationPolicy(object):
    def __init__(self) -> None:
        pass
        
    def load_model(self, model):
        pass
    
    def get_action(self, obs):
        pass

def get_model_output(model, obs):
    model_type = model.__class__.__name__
    feature_extractor_type = model.actor.features_extractor.__class__.__name__
    if model_type == 'MultiTD3':
        # Extract features based on the feature extractor type
        if feature_extractor_type == 'PointCloudExtractor':
            actor_features = model.actor.features_extractor(obs, per_point_output=True)     # bs (* 800), 128
            critic_features = model.critic.features_extractor(obs, per_point_output=True)   # bs (* 800), 128
        elif feature_extractor_type == 'PointCloudGlobalExtractor':
            actor_features = model.actor.features_extractor(obs, per_primitive_output=False)     # bs, 1024
            critic_features = model.critic.features_extractor(obs, per_primitive_output=False)   # bs, 1024
        elif feature_extractor_type == 'StatesExtractor':
            actor_features = model.actor.features_extractor(obs)     # bs, 1024
            critic_features = model.critic.features_extractor(obs)   # bs, 1024
        else:
            raise NotImplementedError
        
        num_prims = len(model.actor.mus)
        out, all_actions = [], []   # all_actions collect actions across different primitives
        for i in range(num_prims):
            actions = model.actor.mus[i](actor_features)                    # bs (* 800), 3
            all_actions.append(actions)
            critic_input = torch.cat([critic_features, actions], dim=-1)    # bs (* 800), 131 or 1027
            out.append(model.critic.q_networks[i](critic_input))            # bs (* 800), 1
        out = torch.concat(out, dim=-1)                                     # bs (* 800), 2
        actions = torch.stack(all_actions, dim=1)                           # bs (* 800), 2, 3

    else:
        if feature_extractor_type == 'PointCloudExtractor':
            actor_features = model.actor.features_extractor(obs, per_point_output=True)     # bs * 800 * 2, 128
            critic_features = model.critic.features_extractor(obs, per_point_output=True)   # bs * 800 * 2, 128
            if hasattr(model.actor, "latent_pi"):
                actor_features = model.actor.latent_pi(actor_features)                      # For SAC policy
            actions = model.actor.mu(actor_features)                                        # bs * 800 * 2, 3
        elif feature_extractor_type == 'PointCloudGlobalExtractor':
            actor_features = model.actor.features_extractor(obs, per_primitive_output=True)     # bs * 2, 1024
            critic_features = model.critic.features_extractor(obs, per_primitive_output=True)   # bs * 2, 1024
            if hasattr(model.actor, "latent_pi"):
                actor_features = model.actor.latent_pi(actor_features)                      # For SAC policy
            actions = model.actor.mu(actor_features)                                            # bs * 2, 3
        elif feature_extractor_type == "StatesExtractor":
            actions = model.actor(obs)                                                      # bs * 2, 3
            critic_features = model.critic.features_extractor(obs)                          # bs * 2, 1024

        # Concatenate with per point critic feature and get per point value
        critic_input = torch.cat([critic_features, actions], dim=-1)                        # bs (* 800) * 2, 131 or 1027
        out = model.critic.q_networks[0](critic_input)                                      # bs (* 800) * 2, 1
            
    return actions, out

class LocationPolicyWithArgmaxQ(LocationPolicy):
    def __init__(self, model=None, temperature=1., deterministic=False, vis_only=False, egreedy=0., sampling_strategy="softmax_all") -> None:
        """
        Args:
            model: the model to be used for inference
            temperature: the softmax temperature
            deterministic: whether to use argmax or sample from the softmax
            vis_only: whether to use the scores for visualization only
            egreedy: the probability of using a random action
            sampling_strategy: the sampling strategy for the primitive grounding
                "softmax_all": sample from the softmax of all potential actions
                "max_primitive": take the argmax of the primitive and then sample from softmax of the locations
                "separate": sample from the softmax of the primitive and then sample from softmax of the locations
        """
        self.model = model
        self.temperature = temperature
        self.egreedy = egreedy
        self.deterministic = deterministic
        self.sampling_strategy = sampling_strategy
        self.vis_only = vis_only # only use the scores for visualization
        return
    
    def load_model(self, model):
        self.model = model
        return
    
    def get_action(self, obs):  
        assert self.model, "LocationPolicyWithArgmaxQ: needs to load a model."
        
        # Determine if the obs is already batched
        batched = obs['object_pcd_points'].ndim == 3
        bs = obs['object_pcd_points'].shape[0] if batched else 1

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        obs_tensor = {}
        for key in self.model.actor.observation_space.spaces.keys():
            try:
                if key not in {'action_location_score', 'action_params'}:
                    obs_tensor[key] = torch.tensor(obs[key]).to(device)
                    if not batched:
                        obs_tensor[key] = obs_tensor[key].unsqueeze(dim=0)
            except:
                print(f'obs_tensor[key] = torch.tensor(obs[key]).to(device).unsqueeze(dim=0)')
                print(f'key:{key}')
        
        # Need to use eval. Otherwise this will affect batch norm.
        was_training = self.model.policy.training
        self.model.policy.set_training_mode(False)
        with torch.no_grad():
            preprocessed_obs = pobs(obs_tensor, self.model.actor.observation_space, normalize_images=self.model.actor.normalize_images)
            actions, out = get_model_output(self.model, preprocessed_obs)

            self.model.policy.set_training_mode(was_training)
            
            # Mask out values based on the primitive grounding
            num_obj_points = obs['object_pcd_points'].shape[-2]     # 400
            num_bg_points = obs['background_pcd_points'].shape[-2]  # 400
            num_points = num_obj_points + num_bg_points             # 800
            num_prims = int(obs['available_prims'].reshape(-1)[0]) # 2, assuming all steps have the same available prims
            prim_groundings = preprocessed_obs['prim_groundings'].reshape(bs, -1).long()    # BS, 2
            prim_groundings = prim_groundings[:, :num_prims]   # BS, 10 -> BS, 2
            out = out.reshape(bs, num_points, num_prims) # BS * 800 * 2 -> BS, 800, 2
            
            masks = torch.ones((len(GroundingTypes), num_points), device=device)
            masks[GroundingTypes.OBJECT_ONLY, num_obj_points:] = 0
            masks[GroundingTypes.BACKGROUND_ONLY, :num_obj_points] = 0
            masks[GroundingTypes.NONE, :] = 0

            prim_mask = masks[prim_groundings]              # BS, 2 -> BS, 2, 800
            prim_mask = torch.transpose(prim_mask, 1, 2)    # BS, 2, 800 -> BS, 800, 2
            out = out + (prim_mask - 1) * 1e7       # BS, 800, 2

            # Compute the scores of all potential actions (note that some sampling strategt does not use this)
            action_score_all = out.reshape(bs, num_points * num_prims)                              # BS, 800 * 2
            action_score_all = torch.nn.Softmax(dim=-1)(action_score_all/self.temperature)          # BS, 800 * 2
            
            action_score = None # score for the selected primitive
            
            if self.vis_only:
                prim_idx, poke_idx = sample_prim_idx(prim_groundings, num_obj_points, num_bg_points)
                
            elif self.deterministic:
                max_idx = torch.argmax(action_score_all, dim=-1)
            else:
                # Uniformly random before learning_starts
                try:
                    if self.model.num_timesteps < self.model.learning_starts or np.random.rand() < self.egreedy:
                        max_idx = torch.multinomial(prim_mask.reshape(bs, -1), num_samples=1).squeeze()
                    else:
                        if self.sampling_strategy == "softmax_all":
                            # Sample from the the softmax of all actions
                            max_idx = torch.multinomial(action_score_all, num_samples=1).squeeze()
                        elif self.sampling_strategy == "max_primitive_softmax_loc":
                            # Calculate the max primitive 
                            prim_score = torch.max(out, dim=-2)[0]
                            prim_idx = torch.argmax(prim_score, dim=-1)

                            action_score = out[torch.arange(bs), :, prim_idx]     # BS, 800, 2 -> BS, 800
                            action_score = torch.nn.Softmax(dim=-1)(action_score/self.temperature)
                            max_idx = torch.multinomial(action_score, num_samples=1).squeeze() * num_prims + prim_idx
                        elif self.sampling_strategy == "separate_softmax":
                            # Sample from the softmax of the primitive and then sample from the softmax of the locations
                            prim_score = torch.max(out, dim=-2)[0]
                            prim_score = torch.nn.Softmax(dim=-1)(prim_score/self.temperature)
                            prim_idx = torch.multinomial(prim_score, num_samples=1).squeeze()

                            action_score = out[torch.arange(bs), :, prim_idx]     # BS, 800, 2 -> BS, 800
                            action_score = torch.nn.Softmax(dim=-1)(action_score/self.temperature)
                            max_idx = torch.multinomial(action_score, num_samples=1).squeeze() * num_prims + prim_idx
                        elif self.sampling_strategy.startswith("top_p"):
                            top_p_value = float(self.sampling_strategy.split("_")[-1])
                            action_logits = out.reshape(bs, num_points * num_prims) / self.temperature
                            max_idx = top_k_top_p_filtering(action_logits, top_p=top_p_value)
                        else:
                            print(f"Sampling strategy {self.sampling_strategy} not implemented.")
                            raise NotImplementedError
                except:
                    max_idx = torch.argmax(action_score_all, dim=-1)
            
            poke_idx = torch.div(max_idx, num_prims).reshape(bs, 1)
            prim_idx = torch.remainder(max_idx, num_prims).reshape(bs, 1)

            # Reformat the return values
            batch_indices = torch.arange(bs, device=device)
            action_score_all = action_score_all.reshape(bs, num_points, num_prims)
            action_params = actions.reshape(bs, num_points, num_prims, -1)
            action_params = action_params[batch_indices, :, prim_idx.flatten()]  # BS, 800, 2, 3 -> BS, 800, 3

            # Reduce action location info for visualization
            if action_score is None:
                action_score = action_score_all[batch_indices, :, prim_idx.flatten()]   # BS, 800, 2 -> BS, 800 
            
            action = {'action_location_score':action_score.cpu().detach().numpy(),
                      'action_location_score_all': action_score_all.cpu().detach().numpy(),
                    'action_params': action_params[..., :3].cpu().detach().numpy(),
                    'poke_idx': poke_idx.cpu().detach().numpy(),
                    'prim_idx': prim_idx.cpu().detach().numpy()}
        return action

class PrimitivePolicyWithArgmaxQ(LocationPolicyWithArgmaxQ):
    def get_action(self, obs):  
        assert self.model, "LocationPolicyWithArgmaxQ: needs to load a model."
        
        # Determine if the obs is already batched
        batched = obs['object_pcd_points'].ndim == 3
        bs = obs['object_pcd_points'].shape[0] if batched else 1

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        obs_tensor = {}
        for key in self.model.actor.observation_space.spaces.keys():
            try:
                if key not in {'action_location_score', 'action_params'}:
                    obs_tensor[key] = torch.tensor(obs[key]).to(device)
                    if not batched:
                        obs_tensor[key] = obs_tensor[key].unsqueeze(dim=0)
            except:
                print(f'obs_tensor[key] = torch.tensor(obs[key]).to(device).unsqueeze(dim=0)')
                print(f'key:{key}')
        
        # Need to use eval. Otherwise this will affect batch norm.
        was_training = self.model.policy.training
        self.model.policy.set_training_mode(False)
        with torch.no_grad():
            preprocessed_obs = pobs(obs_tensor, self.model.actor.observation_space, normalize_images=self.model.actor.normalize_images)
            actions, out = get_model_output(self.model, preprocessed_obs)   # [bs, 2, 3], [bs, 2]

            num_prims = int(obs['available_prims'].reshape(-1)[0]) # 2, assuming all steps have the same available prims
            prim_groundings = preprocessed_obs['prim_groundings'].reshape(bs, -1).long()    # BS, 2
            prim_groundings = prim_groundings[:, :num_prims]   # BS, 10 -> BS, 2
            out = out.reshape(bs, num_prims)                   
            
            prim_mask = torch.ones_like(out, device=device)
            prim_mask[prim_groundings == GroundingTypes.NONE] = 0

            out = out + (prim_mask - 1) * 1e7

            # Softmax
            action_score = out.reshape(bs, num_prims)
            action_score = torch.nn.Softmax(dim=-1)(action_score/self.temperature)
            
            if self.vis_only:
                prim_idx, poke_idx = sample_prim_idx(prim_groundings, 1, 1)
                
            elif self.deterministic:
                prim_idx = torch.argmax(action_score, dim=-1)
            else:
                # Uniformly random before learning_starts
                try:
                    if self.model.num_timesteps < self.model.learning_starts or np.random.rand() < self.egreedy:
                        prim_idx = torch.multinomial(prim_mask.reshape(bs, -1), num_samples=1).squeeze()
                    else:
                        prim_idx = torch.multinomial(action_score, num_samples=1).squeeze()
                except:
                    prim_idx = torch.argmax(action_score, dim=-1)
            
            prim_idx = prim_idx.reshape(bs, 1)
            action = {'prim_idx': prim_idx.cpu().detach().numpy()}
        self.model.policy.set_training_mode(was_training)
        return action


class RandomLocation(LocationPolicy):
    def __init__(self) -> None:
        return
    
    def get_action(self, obs):
        # Choose an index from the observation point cloud
        points = obs['object_pcd_points']
        num_obj_points = obs['object_pcd_points'].shape[-2]
        num_bg_points = obs['background_pcd_points'].shape[-2]
        num_points = num_obj_points + num_bg_points
        prim_groundings = obs['prim_groundings']
        num_prims = prim_groundings.shape[-1]
        if points.ndim == 2:
            prim_idx, poke_idx = sample_prim_idx(prim_groundings, num_obj_points, num_bg_points)
            return {'poke_idx': np.expand_dims(poke_idx, axis=0),
                    'prim_idx': np.expand_dims(prim_idx, axis=0),
                    'action_params': np.zeros((num_points, 3)),
                    'action_location_score': np.zeros(num_points)}
        elif points.ndim == 3: # batched
            bs, _, _ = points.shape
            prim_idxs, poke_idxs = sample_prim_idx(prim_groundings, num_obj_points, num_bg_points)
            prim_idxs = np.array(prim_idxs)[:, np.newaxis]
            poke_idxs = np.array(poke_idxs)[:, np.newaxis]
            return {'poke_idx': poke_idxs,
                    'prim_idx': prim_idxs,
                    'action_params': np.zeros((bs, num_points, 3)),
                    'action_location_score': np.zeros((bs, num_points))}
