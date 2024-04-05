# PPO implementation adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py 

import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer

class ValueNetwork:
    def __init__(self, policy_network):
      self.policy_network = policy_network

    def infer(self, state):
      return self.policy_network.get_value(state).cpu().detach().numpy()

    def learn(self, buffer):
      pass # Warning: Learning handled in PolicyNetwork.

class PolicyNetwork(nn.Module):
    def __init__(self, 
                 n_states, 
                 n_actions, 
                 n_steps, 
                 n_steps_per_learn, 
                 reward_fn=None,
                 clip_coef=0.2,
                 norm_adv=True,
                 clip_vloss=True,
                 ent_coef=0.0,
                 vf_coef=0.5,
                 gamma=0.99,
                 gae_lambda=0.95,
                 max_grad_norm=0.5,
                 learning_rate=2e-4,
                 update_epochs=10,
                 n_minibatches=32,
                 hidden_size=64):
      super(PolicyNetwork, self).__init__() 
      # Parameters.
      self.n_steps = n_steps
      self.n_episodes = n_steps_per_learn // n_steps
      self.n_states = n_states
      self.n_actions = n_actions
      self.reward_fn = reward_fn

      self.clip_coef = clip_coef
      self.norm_adv = norm_adv
      self.clip_vloss = clip_vloss
      self.ent_coef = ent_coef
      self.vf_coef = vf_coef
      self.gamma = gamma
      self.gae_lambda = gae_lambda
      self.max_grad_norm = max_grad_norm
      self.learning_rate = learning_rate
      self.update_epochs = update_epochs
      self.n_minibatches = n_minibatches
      self.hidden_size = hidden_size
      self.batch_size = self.n_steps * self.n_episodes
      self.minibatch_size = self.batch_size // self.n_minibatches
      self.cuda = True 
      # Build network.
      self.critic = nn.Sequential(
          layer_init(nn.Linear(n_states, self.hidden_size)),
          nn.Tanh(),
          layer_init(nn.Linear(self.hidden_size, self.hidden_size)),
          nn.Tanh(),
          layer_init(nn.Linear(self.hidden_size, 1), std=1.0),
          nn.Tanh(),
      )
      self.actor_mean = nn.Sequential(
          layer_init(nn.Linear(n_states, self.hidden_size)),
          nn.Tanh(),
          layer_init(nn.Linear(self.hidden_size, self.hidden_size)),
          nn.Tanh(),
          layer_init(nn.Linear(self.hidden_size, n_actions), std=0.01),
          nn.Tanh(),
      )
      self.actor_logstd = torch.Tensor(torch.zeros(1, n_actions))
      # Create optimizer and storage.
      self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)
      self.device = "cuda" if torch.cuda.is_available() and self.cuda else "cpu"

      self.obs = torch.zeros((self.n_steps, self.n_episodes, n_states)).to(self.device)
      self.actions = torch.zeros((self.n_steps, self.n_episodes, n_actions)).to(self.device)
      self.logprobs = torch.zeros((self.n_steps, self.n_episodes)).to(self.device)
      self.rewards = torch.zeros((self.n_steps, self.n_episodes)).to(self.device)
      self.dones = torch.zeros((self.n_steps, self.n_episodes)).to(self.device)
      self.values = torch.zeros((self.n_steps, self.n_episodes)).to(self.device)
    
    def __call__(self, state):
      with torch.no_grad():
        action, logprob, _, value = self.get_action_and_value(torch.Tensor(state))
      return action
 
    def get_log_prob(self, action):
      action_logstd = self.actor_logstd 
      action_std = torch.exp(action_logstd)
      probs = Normal(action, action_std)
      return probs.log_prob(action).sum(1)

    def get_value(self, x):
      return self.critic(x)

    def get_action_and_value(self, x, action=None):
      action_mean = self.actor_mean(x)
      action_logstd = self.actor_logstd 
      action_std = torch.exp(action_logstd)
      probs = Normal(action_mean, action_std)
      if action is None:
          action = probs.sample()
      return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def infer(self, state, noise=True):
      with torch.no_grad():
        state = torch.Tensor(state)
        if noise:
          action, _, _, _ = self.get_action_and_value(state)
          action = action.squeeze(0) # TODO. Is squeeze dimension correct?
        else:
          action = self.actor_mean(state)
      return action.cpu().numpy()

    def learn(self, buffer):
      # Prepare batch and compute returns with variance reduction.
      states = torch.from_numpy(buffer.states)
      actions = torch.from_numpy(buffer.actions)
      rewards = buffer.rewards
      assert states.shape[0] == actions.shape[0] == self.n_episodes
    #   assert states.shape[1] == actions.shape[1] + 1 == self.n_steps + 1
      advantages = torch.zeros_like(self.rewards).to(self.device)
      for episode in range(self.n_episodes):
        for step in range(self.n_steps):
          obs = states[episode, step]
          action = actions[episode, step]
          with torch.no_grad():
            logprob = self.get_log_prob(action)
            value = self.get_value(obs) 
            reward = rewards[episode, step]
          done = step == self.n_steps - 1
          self.obs[step, episode] = obs
          self.actions[step, episode] = action
          self.logprobs[step, episode] = logprob
          self.rewards[step, episode] = reward
          self.values[step, episode] = value
          self.dones[step, episode] = done
        # Advantages. 
        with torch.no_grad():
          next_obs = states[-1, episode]
          next_value = self.get_value(next_obs).reshape(1, -1)
          next_done = True # For now, assume that episode is done on last step. 
          lastgaelam = 0
          for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
              nextnonterminal = 1.0 - next_done
              nextvalues = next_value
            else:
              nextnonterminal = 1.0 - self.dones[t + 1, episode]
              nextvalues = self.values[t + 1, episode]
            delta = self.rewards[t, episode] + self.gamma * nextvalues * nextnonterminal - self.values[t, episode]
            advantages[t, episode] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
      # Returns.
      with torch.no_grad():
        returns = advantages + self.values
      # Flatten batch.
      b_obs = self.obs.reshape((-1, self.n_states))
      b_logprobs = self.logprobs.reshape(-1)
      b_actions = self.actions.reshape((-1, self.n_actions))
      b_advantages = advantages.reshape(-1)
      b_returns = returns.reshape(-1)
      b_values = self.values.reshape(-1)
      # Optimize the policy and value networks.
      b_inds = np.arange(self.batch_size)
      clipfracs = []
      for epoch in range(self.update_epochs):
          np.random.shuffle(b_inds)
          for start in range(0, self.batch_size, self.minibatch_size):
              end = start + self.minibatch_size
              mb_inds = b_inds[start:end]

              _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
              logratio = newlogprob - b_logprobs[mb_inds]
              ratio = logratio.exp()

              with torch.no_grad():
                  # calculate approx_kl http://joschu.net/blog/kl-approx.html
                  old_approx_kl = (-logratio).mean()
                  approx_kl = ((ratio - 1) - logratio).mean()
                  clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

              mb_advantages = b_advantages[mb_inds]
              if self.norm_adv:
                  std = mb_advantages.std()
                  std = 0 if torch.isnan(std) else std # Handle NaN.
                  mb_advantages = (mb_advantages - mb_advantages.mean()) / (std + 1e-8)

              # Policy loss.
              pg_loss1 = -mb_advantages * ratio
              pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
              pg_loss = torch.max(pg_loss1, pg_loss2).mean()

              # Value loss.
              newvalue = newvalue.view(-1)
              if self.clip_vloss:
                  v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                  v_clipped = b_values[mb_inds] + torch.clamp(
                      newvalue - b_values[mb_inds],
                      -self.clip_coef,
                      self.clip_coef,
                  )
                  v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                  v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                  v_loss = 0.5 * v_loss_max.mean()
              else:
                  v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

              # Optimize.
              entropy_loss = entropy.mean()
              loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
              self.optimizer.zero_grad()
              loss.backward()
              nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
              self.optimizer.step()
      #print('Mean reward: {}'.format(self.rewards.mean().item()))
