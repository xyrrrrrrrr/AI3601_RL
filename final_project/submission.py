'''
This file is used to generate the submission file for the competition.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import copy
from pathlib import Path

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, phi=0.05):
		super(Actor, self).__init__()
		self.l1 = layer_init(nn.Linear(state_dim + action_dim, 400))
		self.l2 = layer_init(nn.Linear(400, 300))
		self.l3 = layer_init(nn.Linear(300, action_dim), std=0.01)
		
		self.max_action = max_action
		self.phi = phi


	def forward(self, state, action):
		a = F.relu(self.l1(torch.cat([state, action], 1)))
		# a = F.tanh(self.l1(torch.cat([state, action], 1)))
		a = F.relu(self.l2(a))
		# a = F.tanh(self.l2(a))
		a = self.phi * self.max_action * torch.tanh(self.l3(a))
		return (a + action).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		self.l1 = layer_init(nn.Linear(state_dim + action_dim, 400))
		self.l2 = layer_init(nn.Linear(400, 300))
		self.l3 = layer_init(nn.Linear(300, 1), std=0.003)

		self.l4 = layer_init(nn.Linear(state_dim + action_dim, 400))
		self.l5 = layer_init(nn.Linear(400, 300))
		self.l6 = layer_init(nn.Linear(300, 1), std=0.003)


	def forward(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(torch.cat([state, action], 1)))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2
	
	def q1(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

class Actor_2(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, phi=0.05):
		super(Actor_2, self).__init__()
		self.l1 = layer_init(nn.Linear(state_dim, 256))
		self.l2 = layer_init(nn.Linear(256, 128))
		self.l3 = layer_init(nn.Linear(128, action_dim), std=0.4, bias_const=0.1)

		# self.l4 = layer_init(nn.Linear(128, action_dim), std=0.01)
		
		self.max_action = max_action
		self.phi = phi


	def forward(self, state):
		a = F.relu(self.l1(state))
		# a = F.tanh(self.l1(state))
		a = F.relu(self.l2(a))
		# a = F.tanh(self.l2(a))
		mean = self.l3(a)
		# logstd = torch.zeros_like(mean) #
		logstd = torch.ones_like(mean) * 0.01
		# calculate logstd from a using 
		# logstd = self.l4(a).clamp(-4, 4)
		dist = torch.distributions.Normal(mean, logstd)
		return dist

class Critic_2(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic_2, self).__init__()
		self.l1 = layer_init(nn.Linear(state_dim , 256))
		self.l2 = layer_init(nn.Linear(256, 128))
		self.l3 = layer_init(nn.Linear(128, 1), std=0.003)

	def forward(self, state):
		q1 = F.relu(self.l1(state))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
		super(VAE, self).__init__()
		self.e1 = layer_init(nn.Linear(state_dim + action_dim, 750))
		self.e2 = layer_init(nn.Linear(750, 750))

		self.mean = layer_init(nn.Linear(750, latent_dim), std=0.1)
		self.log_std = layer_init(nn.Linear(750, latent_dim), std=0.003)

		self.d1 = layer_init(nn.Linear(state_dim + latent_dim, 750))
		self.d2 = layer_init(nn.Linear(750, 750))
		self.d3 = layer_init(nn.Linear(750, action_dim), std=0.01)

		self.max_action = max_action
		self.latent_dim = latent_dim
		self.device = device


	def forward(self, state, action):
		z = F.relu(self.e1(torch.cat([state, action], 1)))
		z = F.relu(self.e2(z))

		mean = self.mean(z)
		# Clamped for numerical stability 
		log_std = self.log_std(z).clamp(-4, 15)
		std = torch.exp(log_std)
		z = mean + std * torch.randn_like(std)
		
		u = self.decode(state, z)

		return u, mean, std


	def decode(self, state, z=None):
		# When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
		if z is None:
			z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		return self.max_action * torch.tanh(self.d3(a))
		


class BCQ(object):
	def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
		latent_dim = action_dim * 2

		self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

		self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
		self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-4) 

		self.max_action = max_action
		self.action_dim = action_dim
		self.discount = discount
        
		self.tau = tau
		self.lmbda = lmbda
		self.device = device


	def select_action(self, state):		
		with torch.no_grad():
			state = torch.FloatTensor(state).repeat(100, 1).to(self.device)
			action = self.actor(state, self.vae.decode(state))
			q1 = self.critic.q1(state, action)
			ind = q1.argmax(0)
		return action[ind].cpu().data.numpy().flatten()
	
	def load_pth(self):
		actor_path = os.path.dirname(os.path.abspath(__file__)) + "/BCQ_actor_Hopper-v2_0_0.99_0.005_0.75_0.05_3250.pth"
		critic_path = os.path.dirname(os.path.abspath(__file__)) + "/BCQ_critic_Hopper-v2_0_0.99_0.005_0.75_0.05_3250.pth"
		vae_path = os.path.dirname(os.path.abspath(__file__)) + "/BCQ_vae_Hopper-v2_0_0.99_0.005_0.75_0.05_3250.pth"
		self.actor.load_state_dict(torch.load(actor_path, map_location=torch.device('cpu')))
		# self.actor_target.load_state_dict(torch.load(dir + 'actor.pth'))
		self.critic.load_state_dict(torch.load(critic_path, map_location=torch.device('cpu')))
		# self.critic_target.load_state_dict(torch.load(dir + 'critic_target.pth'))
		self.vae.load_state_dict(torch.load(vae_path, map_location=torch.device('cpu')))

	def set_not_grad(self):
		self.actor.eval()
		self.critic.eval()
		self.vae.eval()


class AWR(object):
	def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
		self.actor = Actor_2(state_dim, action_dim, max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
		# self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=2.5e-5, momentum=0.9)

		self.critic = Critic_2(state_dim, action_dim).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-2)
		# self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=1e-2, momentum=0.9)

		self.max_action = max_action
		self.action_dim = action_dim
		self.discount = discount
		
		self.tau = tau
		self.lmbda = lmbda
		self.device = device

		self.actor_update_iter = 1000
		self.critic_update_iter = 200

		self.state_mean = torch.tensor([0] * 11)
		self.state_std = torch.tensor([10] * 11)
		self.action_mean = torch.tensor([0] * 3)
		self.action_std = torch.tensor([1] * 3)

		self.current_id = 0

	def select_action(self, state):		
		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device)
			state = (state - self.state_mean) / self.state_std
			dist = self.actor(state)
			action = dist.sample()
			action = action * self.action_std + self.action_mean
			action = action.clamp(-self.max_action, self.max_action)
			# print(action)
		return action.cpu().data.numpy().flatten()
	
	def load_pth(self):
		actor_path = os.path.dirname(os.path.abspath(__file__)) + "/AWR_2500_actor.pth"
		critic_path = os.path.dirname(os.path.abspath(__file__)) + "/AWR_2500_critic.pth"
		self.actor.load_state_dict(torch.load(actor_path))
		self.critic.load_state_dict(torch.load(critic_path))

	def set_not_grad(self):
		self.actor.eval()
		self.critic.eval()

	def normalizer(self, state, action):
		# normalize state and action
		# state_max = [np.inf] * 11
		# state_min = [-np.inf] * 11
		# action_max = [1.0] * 3
		# action_min = [-1.0] * 3

		state = (state - self.state_mean) / self.state_std
		action = (action - self.action_mean) / self.action_std

		return state, action

def sample(dataset, batch_size=100)-> list:
    '''
    This function is used to sample the dataset.
    :param dataset: the dataset
    :param length: the length of the sample
    :return: the sample
    '''
    # generate the index not repeat
    index = np.random.choice(len(dataset[0]), batch_size, replace=False)
    # sample the dataset
    b_state = dataset[0][index]
    b_action = dataset[1][index]
    b_reward = dataset[2][index]
    b_next_state = dataset[3][index]
    b_not_done = dataset[4][index]

    return [b_state, b_action, b_reward, b_next_state, b_not_done]

def layer_init(layer, std=np.sqrt(2), bias_const=0.0)-> torch.nn.Module:
    '''
    This function is used to initialize the layer.
    '''
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    
    return layer

def set_seed(seed)->None:
    '''
    This function is used to set the random seed.
    :param seed: the seed.
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(0)
policy = BCQ(state_dim=11, action_dim=3, max_action=1.0, device='cpu')
# policy = AWR(state_dim=11, action_dim=3, max_action=1.0, device='cpu')
policy.load_pth()
policy.set_not_grad()

def my_controller(observation, action_space, is_act_continuous=False)->list:
    '''
    This function is used to generate the action.
    :param observation: the observation
    :param action_space: the action space
    :param is_act_continuous: whether the action is continuous
    :return: the action
    '''
    if type(observation) is dict:
        observation = observation['obs']
    # convert the observation to tensor
    # observation = torch.tensor(observation, dtype=torch.float32, device='cpu')
    # get the action
    action = policy.select_action(observation)
    # return the action
    return [list(action)]

if __name__ == '__main__':
    observation = [0] * 11
    action_space = [0] * 3
    is_act_continuous = False
    action = my_controller(observation, action_space, is_act_continuous)
    print(action)