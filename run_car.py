from env import simGym
from ppo_network import PolicyNetwork
import numpy as np
from data import Trajectory, Buffer
import pickle
from datetime import datetime
import time

n_actions = 1
n_states = 2
n_steps = 200  # steps for per episode
n_episodes = 10
n_steps_per_learn = n_episodes * n_steps

import tkinter as tk
from tkinter import filedialog

# 创建一个Tkinter根窗口并隐藏它
root = tk.Tk()
root.withdraw()

# 打开文件选择对话框并获取选择的文件路径
file_path = filedialog.askopenfilename()  # 这会打开一个对话框，并让用户选择一个文件，返回文件的完整路径

# 检查是否选择了文件（点击“取消”返回的是空字符串）
if file_path:
    print("The path:", file_path)
else:
    print("No choosen file...")
    exit

with open(file_path, "rb") as f: # "rb" because we want to read in binary 
    pn = pickle.load(f)

# show results
sim = simGym(pn.env_name,render_mode=True)
observation, info = sim.reset()
for _ in range(10000):
            actions = pn.infer(state=observation, noise=False)
            observation, reward, terminated, truncated, info = sim.step(actions)
            print(reward, actions)
            time.sleep(0.1)
            # if terminated or truncated:
            #     observation, info = sim.reset()
            #     break