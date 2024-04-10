from env import simGym
from ppo_network import PolicyNetwork
import numpy as np
from data import Trajectory, Buffer
import pickle
from datetime import datetime

n_actions = 1
n_states = 2
n_steps = 200  # steps for per episode
n_episodes = 10
n_steps_per_learn = n_episodes * n_steps
# pn = Agent(n_states, 
#                  n_actions, 
#                  n_steps, 
#                  n_episodes)
pn = PolicyNetwork(n_states, 
                 n_actions, 
                 n_steps, 
                 n_steps_per_learn)
t = Trajectory()
buffer = Buffer(n_states, n_actions, n_episodes, n_steps)

sim = simGym()
observation, info = sim.reset()
training_times = 5001
reward_list = []

for times in range(training_times):
    if pn.current_training_time%500 == 0: # auto save model
        current_date_and_time = datetime.now()
        name = "./model/" + str(pn.current_training_time) + "-" + str(current_date_and_time) + ".model"
        with open(name, "wb") as f: 
            pickle.dump(pn, f)
    buffer.reset()
    for i in range(n_episodes):
        t.reset()
        for _ in range(n_steps):
            actions = pn.infer(state=observation, noise=True)
            # actions = pn.get_action(observation)
            observation, reward, terminated, truncated, info = sim.step(actions)
            if terminated or truncated:
                observation, info = sim.reset()
            t.add(observation, actions, reward)
        observation, info = sim.reset()
        buffer.add(t)
    print(times, np.mean(buffer.rewards))
    reward_list.append(np.mean(buffer.rewards))
    pn.learn(buffer)

# show results
sim = simGym(render_mode=True)
observation, info = sim.reset()
for _ in range(300):
            actions = pn.infer(state=observation, noise=None)
            observation, reward, terminated, truncated, info = sim.step(actions)
            if terminated or truncated:
                observation, info = sim.reset()
                break