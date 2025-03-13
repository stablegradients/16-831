from collections import OrderedDict

from rob831.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from rob831.infrastructure.replay_buffer import ReplayBuffer
from rob831.infrastructure.utils import *
from rob831.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent
from rob831.infrastructure import pytorch_util as ptu


class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # advantage = estimate_advantage(...)

        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        loss = OrderedDict()
        
        # Update the critic
        critic_loss = 0
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)
        loss['Loss_Critic'] = critic_loss
        
        # Estimate advantage
        advantage = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)
        
        # Update the actor
        actor_loss = 0
        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            actor_loss = self.actor.update(ob_no, ac_na, advantage)
        loss['Loss_Actor'] = actor_loss
        
        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        # TODO Implement the following pseudocode:
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
        
        # Convert numpy arrays to torch tensors before passing to the critic
        ob_no_tensor = ptu.from_numpy(ob_no)
        next_ob_no_tensor = ptu.from_numpy(next_ob_no)
        re_n_tensor = ptu.from_numpy(re_n)
        terminal_n_tensor = ptu.from_numpy(terminal_n)
        
        # 1) Get V(s) by querying critic with current states
        v_s = self.critic.forward(ob_no_tensor)
        
        # 2) Get V(s') by querying critic with next states
        v_sp = self.critic.forward(next_ob_no_tensor)
        
        # 3) For terminal states, set V(s') to 0 to cut off future rewards
        # Convert terminal_n to boolean array for proper indexing
        terminal_mask = terminal_n_tensor.bool()
        v_sp[terminal_mask] = 0
        
        # Calculate Q(s,a) = r(s,a) + gamma*V(s')
        # For terminal states, this becomes just r(s,a) since V(s') is 0
        q_values = re_n_tensor + self.gamma * v_sp
        
        # 4) Calculate advantage A(s,a) = Q(s,a) - V(s)
        # This gives us: A(s,a) = r(s,a) + gamma*V(s') - V(s) for non-terminal states
        # And A(s,a) = r(s,a) - V(s) for terminal states
        adv_n = q_values - v_s
        
        # Convert back to numpy for the rest of the pipeline
        adv_n = ptu.to_numpy(adv_n)
        
        # Optionally standardize advantages to stabilize training
        if self.standardize_advantages:
            adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
            
        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
