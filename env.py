import serial
import time
import gymnasium as gym

RECORDNUM = 10

def read_force():
    return 0

class kunEnv():
    def __init__(self):
        ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
        ser.flush()
        self.curved_angle = 0
        self.force = 0
        self.freq = 10
        self.env = [0, 0]
        self.force_record = [0 for i in range(RECORDNUM)]
        self.force_record_index = 0
        self.previous_actions = 0
        self.actions = 0
        
    def step(self,actions): # actions: lens = 1, actions = pressure
        self.previous_actions = self.actions
        self.ser.write(f"{actions}\n".encode('utf-8'))
        time.sleep(0.001)
        for i in range(10): # try communication times
            if self.ser.in_waiting > 0:
                self.curved_angle = self.ser.readline().decode('utf-8').rstrip() # Curved angle
                break
        self.force = read_force()
        self.force_record[self.force_record_index] = self.force
        self.force_record_index += 1
        if self.force_record_index>=RECORDNUM:
            self.force_record_index = 0
        self.env = [self.curved_angle]
        self.actions = actions
        
        
    def reward(self):
        action_speed_weight = 0.0001
        normal_bias = -10
        self.reward = sum(self.force_record) + action_speed_weight*(self.actions-self.previous_actions)**2 + normal_bias
        
    def reset(self):
        self.previous_actions = 0
        self.ser.write(f"{0.00}\n".encode('utf-8'))
        time.sleep(0.1)
        self.actions = 0
        self.force_record = [0 for i in range(RECORDNUM)]
        self.force_record_index = 0
        self.curved_angle = 0
        self.force = 0

class simGym():
    def __init__(self, render_mode=None):
        if not render_mode:
            self.env = gym.make('Pendulum-v1', g=9.81)
        else:
            self.env = gym.make('Pendulum-v1', g=9.81, render_mode="human")
        # self.env = gym.make('CartPole-v1')

    def reset(self):
        observation, info = self.env.reset()
        return observation, info 

    def step(self,actions):
        observation, reward, terminated, truncated, info = self.env.step(actions)
        return observation, reward, terminated, truncated, info
        