import numpy as np

class Buffer:
  def __init__(self, state_space, action_space, n_episodes, n_steps, log_map=None):
    self.state_space = state_space
    self.action_space = action_space
    self.n_episodes = n_episodes
    self.n_steps = n_steps
    self.log_map = log_map 
    self.reset()

  def ready(self):
    return self.episode_index == self.n_episodes

  def reset(self):
    self.states = np.zeros((self.n_episodes, self.n_steps, self.state_space if self.state_space is not None else 0), dtype=np.float32)
    self.actions = np.zeros((self.n_episodes, self.n_steps, self.action_space if self.action_space is not None else 0), dtype=np.float32)
    self.rewards = np.zeros((self.n_episodes, self.n_steps), dtype=np.float64)
    self.episode_index = 0

  def add(self, trajectory):
    # assert len(trajectory) == self.n_steps + 1
    # assert not self.ready()
    for index in range(self.n_steps):
      if self.state_space is not None:
        self.states[self.episode_index, index] = trajectory.data[index][0]
      if self.action_space is not None:
        self.actions[self.episode_index, index] = trajectory.data[index][1]
      self.rewards[self.episode_index, index] = trajectory.data[index][2] # TODO. Technically we don't always need this...
    # if self.state_space is not None:
    #   self.states[self.episode_index, self.n_steps] = trajectory.data[self.n_steps][0]
    self.episode_index += 1

class Trajectory:
  def __init__(self):
    self.data = []
    self.size = 0

  def __len__(self):
    return len(self.data)

  def add(self, state, action, reward):
    self.data.append((state, action, reward)) 

  def reset(self):
    self.data = []
  
  def to_numpy(self):
    # Note that goals are not returned.
    # Trajectories are episodic, but we support multiple episodes for legacy reasons.
    states = []
    actions = []
    rewards = []
    episode_states = []
    episode_actions = []
    episode_rewards = []
    for state, action, reward in self.data:
      episode_states.append(state)
      if action is None: # None action marks end of episode.
        states.append(episode_states)
        actions.append(episode_actions)
        rewards.append(episode_rewards)
        episode_states = []
        episode_actions = []
        episode_rewards = []
      else:
        episode_actions.append(action)
        episode_rewards.append(reward)
    if len(states) > 0:
      return np.array(states), np.array(actions), np.array(rewards)
    else:
      return np.array([episode_states]), np.array([episode_actions], np.array([episode_rewards]))
