import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from rob831.infrastructure import pytorch_util as ptu
from rob831.policies.base_policy import BasePolicy

from rob831.infrastructure.utils import normalize

class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers, size=self.size)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            if len(obs.shape) > 1:
                observation = obs
            else:
                observation = obs[None]

            # return the action that the policy prescribes
            input = ptu.from_numpy(observation)
            output_dist = self.forward(input)
            action = ptu.to_numpy(output_dist.sample())
            return action

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        input = ptu.from_numpy(observations)
        pred_action_distribution = self.forward(input)
        target_action = ptu.from_numpy(actions)
        loss = -1 * pred_action_distribution.log_prob(target_action).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if self.discrete:
            logits = self.logits_na(observation)
            probs = torch.softmax(logits, dim=-1)
            return distributions.Categorical(probs)
        else:
            batch_mean = self.mean_net(observation)
            scale_tril = torch.diag(torch.exp(self.logstd))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.MultivariateNormal(
                batch_mean,
                scale_tril=batch_scale_tril,
            )
            return action_distribution

#####################################################

class MLPPolicyPG(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):

        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.baseline_loss = nn.MSELoss()

    def update(self, observations, actions, advantages, q_values=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        distribution = self.forward(observations)  # returns distribution
        log_prob = distribution.log_prob(actions)
        loss = - torch.sum(log_prob * advantages)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        policy_loss = loss.detach().clone()

        # If we are using a baseline, we must update it with MSE to predict Q-values
        if self.nn_baseline:
            assert q_values is not None, (
                "q_values must be provided if using baseline."
            )
            targets = normalize(q_values, np.mean(q_values), np.std(q_values))
            targets = ptu.from_numpy(targets)

            baseline_predictions = self.baseline(observations).squeeze(-1)
            baseline_loss = self.baseline_loss(baseline_predictions, targets)

            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()

        train_log = {
            'Training Loss': ptu.to_numpy(policy_loss),
        }
        return train_log

    def run_baseline_prediction(self, observations):
        """
            Helper function that converts `observations` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `observations`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        observations = ptu.from_numpy(observations)
        pred = self.baseline(observations)
        return ptu.to_numpy(pred.squeeze())