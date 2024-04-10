import serial
import time
import gymnasium as gym
import numpy as np
from gymnasium import spaces

RECORDNUM = 10

def read_force():
    return 0

class kunEnv():
    def __init__(self):
        self.ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=0.2)
        self.ser.flush()
        self.curved_angle = 0
        self.force = 0
        self.freq = 10
        self.states = [0, 0]
        self.force_record = [0 for i in range(RECORDNUM)]
        self.force_record_index = 0
        self.previous_actions = 0
        self.actions = 0
        
    def step(self,actions): # actions: lens = 1, actions = pressure
        self.previous_actions = self.actions
        self.ser.write(f"{actions}\n".encode('utf-8'))
        line = self.ser.readline().decode('utf-8').rstrip()  # 尝试读取一行数据
        if line:  # 检查是否接收到数据
            if ',' in line:  # 检查数据是否为两个值
                analog_values = line.split(',')  # 使用逗号分隔字符串
                if len(analog_values) == 2:
                    a1_value, a2_value = analog_values
                    print(f"A1 Value: {a1_value}, A2 Value: {a2_value}")
        self.force = read_force()
        self.force_record[self.force_record_index] = self.force
        self.force_record_index += 1
        if self.force_record_index>=RECORDNUM:
            self.force_record_index = 0
        self.states = np.array([a1_value,a2_value])
        self.actions = actions
        action_speed_weight = -0.01
        normal_bias = -10
        self.reward = sum(self.force_record) + action_speed_weight*(self.actions-self.previous_actions)**2 - 10
        terminated, truncated, info = 0
        return self.states, self.reward, terminated, truncated, info

    def reset(self):
        self.previous_actions = 0
        self.ser.write(f"{0.00}\n".encode('utf-8'))
        time.sleep(0.5)
        self.actions = 0
        self.force_record = [0 for i in range(RECORDNUM)]
        self.force_record_index = 0
        self.curved_angle = 0
        self.force = 0

class simGym():
    def __init__(self, render_mode=None):
        if not render_mode:
            self.env = gym.make('MountainCarContinuous-v0')
        else:
            self.env = gym.make('MountainCarContinuous-v0', render_mode="human")
        # self.env = gym.make('CartPole-v1')

    def reset(self):
        observation, info = self.env.reset()
        return observation, info 

    def step(self,actions):
        observation, reward, terminated, truncated, info = self.env.step(actions)
        return observation, reward, terminated, truncated, info
        