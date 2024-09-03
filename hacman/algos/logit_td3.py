from stable_baselines3.td3.policies import TD3Policy, Actor
import torch as th
import torch.nn.functional as F

class LogitTD3Policy(TD3Policy):
    """
    TD3 Policy class used for logit-based primitive selection
    """
    def __init__(self, *args, num_primitives=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_primitives = num_primitives

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # action consists of primitive params and primitive logits
        # When deterministic, we replace the primitive logits with one hot of the highest logit
        # When not deterministic, we sample from the primitive logits
        action = self.actor(observation)
        logits = action[:, -self.num_primitives:]
        if deterministic:
            index = th.argmax(logits, dim=-1)
            one_hot = F.one_hot(index, num_classes=self.num_primitives).float()
        else:
            one_hot = F.gumbel_softmax(logits, hard=True)

        action[:, -self.num_primitives:] = one_hot

        return action