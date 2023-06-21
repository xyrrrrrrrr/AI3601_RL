from collections import deque
import math
import random
import time
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.distributions import Normal

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        '''
        Args: state (ndarray (3,)), action (int), reward (float), next_state (ndarray (3,)), done (bool)

        No return
        '''
        """ ------------- Programming 5: implement the push operation of one sample ------------- """
        """ YOUR CODE HERE """
        # 将状态、动作、奖励、下一个状态、是否结束标志放入一个元组并放入buffer中
        self.buffer.append((state, action, reward, next_state, done))
        """ ------------- Programming 5 ------------- """

    def sample(self, batch_size):
        '''
        Args: batch_size (int)

        Required return: a batch of states (ndarray, (batch_size, state_dimension)), a batch of actions (list or tuple, length=batch_size),
        a batch of rewards (list or tuple, length=batch_size), a batch of next-states (ndarray, (batch_size, state_dimension)),
        a batch of done flags (list or tuple, length=batch_size)
        '''
        """ ------------- Programming 6: implement the sample operation of a batch of samples (note that to you need to satisfy 
        the format of the return as stated above to make the replay buffer compatible with other components in main.py) ------------- """
        """ YOUR CODE HERE """
        # 初始化状态、动作、奖励、下一个状态、是否结束标志
        batch_state = np.zeros((batch_size, 3))
        batch_action = []
        batch_reward = []
        batch_next_state = np.zeros((batch_size, 3))
        batch_done = []
        # 按照batch_size的大小从buffer中随机抽取样本
        Sample = random.sample(self.buffer, batch_size)
        # 将抽取的样本分别放入对应的列表中
        for i in range(batch_size):
            batch_state[i] = Sample[i][0]
            batch_action.append(Sample[i][1])
            batch_reward.append(Sample[i][2])
            batch_next_state[i] = Sample[i][3]
            batch_done.append(Sample[i][4])

        return batch_state, batch_action, batch_reward, batch_next_state, batch_done
        """ ------------- Programming 6 ------------- """

    def __len__(self):
        return len(self.buffer)