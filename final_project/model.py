'''
This file is used to define the model.
'''
import os
import copy
import numpy as np
import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from utils import layer_init, sample, sample_whole_process

FixedNormal = Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)
entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)
FixedNormal.mode = lambda self: self.mean

class GuaussianAction(nn.Module):

    def __init__(self, size_in, size_out):
        super().__init__()
        self.fc_mean = nn.Linear(size_in, size_out)

        # ====== INITIALIZATION ======
        self.fc_mean.weight.data.mul_(0.1)
        self.fc_mean.bias.data.mul_(0.0)

        self.fc_std = nn.Linear(size_in, size_out)
        self.fc_std.weight.data.mul_(0.1)
        self.fc_std.bias.data.mul_(0.01)

    def forward(self, x):
        action_mean = self.fc_mean(x)
        action_std = F.relu(self.fc_std(x)) + 1e-5
        # print(action_mean.shape, self.logstd.shape)
        return FixedNormal(action_mean, action_std.exp())


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
		# a = F.tanh(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		# a = F.tanh(self.d2(a))
		return self.max_action * torch.tanh(self.d3(a))
		



class BCQ(object):
	def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
		latent_dim = action_dim * 2

		self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-8)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-8)

		self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
		self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-8) 

		self.max_action = max_action
		self.action_dim = action_dim
		self.discount = discount
        
		self.tau = tau
		self.lmbda = lmbda
		self.device = device


	def select_action(self, state):		
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
			action = self.actor(state, self.vae.decode(state))
			q1 = self.critic.q1(state, action)
			ind = q1.argmax(0)
		return action[ind].cpu().data.numpy().flatten()
	
	def load_pth(self):
		# actor_path = os.path.dirname(os.path.abspath(__file__)) + "/models/BCQ_actor_Hopper-v2_0_0.99_0.005_0.75_0.05_3000.pth"
		# critic_path = os.path.dirname(os.path.abspath(__file__)) + "/models/BCQ_critic_Hopper-v2_0_0.99_0.005_0.75_0.05_3000.pth"
		# vae_path = os.path.dirname(os.path.abspath(__file__)) + "/models/BCQ_vae_Hopper-v2_0_0.99_0.005_0.75_0.05_3000.pth"
		actor_path = os.path.dirname(os.path.abspath(__file__)) + "/models/BCQ_actor_Hopper-v2_0_0.99_0.005_0.75_0.05_3250.pth"
		critic_path = os.path.dirname(os.path.abspath(__file__)) + "/models/BCQ_critic_Hopper-v2_0_0.99_0.005_0.75_0.05_3250.pth"
		vae_path = os.path.dirname(os.path.abspath(__file__)) + "/models/BCQ_vae_Hopper-v2_0_0.99_0.005_0.75_0.05_3250.pth"
		self.actor.load_state_dict(torch.load(actor_path, map_location=torch.device('cpu')))
		# self.actor_target.load_state_dict(torch.load(dir + 'actor.pth'))
		self.critic.load_state_dict(torch.load(critic_path, map_location=torch.device('cpu')))
		# self.critic_target.load_state_dict(torch.load(dir + 'critic_target.pth'))
		self.vae.load_state_dict(torch.load(vae_path, map_location=torch.device('cpu')))

	def set_not_grad(self):
		self.actor.eval()
		self.critic.eval()
		self.vae.eval()

	def train(self, dataset, iterations, batch_size=100):
		print("BCQ training")
		for it in tqdm.tqdm(range(iterations)):
			# Sample replay buffer / batch
			state, action, reward, next_state, not_done = sample(dataset, batch_size)

			# Variational Auto-Encoder Training
			recon, mean, std = self.vae(state, action)
			recon_loss = F.mse_loss(recon, action)
			KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			vae_loss = recon_loss + 0.5 * KL_loss

			self.vae_optimizer.zero_grad()
			vae_loss.backward()
			self.vae_optimizer.step()


			# Critic Training
			with torch.no_grad():
				# Duplicate next state 10 times
				next_state = torch.repeat_interleave(next_state, 10, 0)

				# Compute value of perturbed actions sampled from the VAE
				target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state, self.vae.decode(next_state)))

				# Soft Clipped Double Q-learning 
				target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
				# Take max over each action sampled from the VAE
				target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)
				
				target_Q = reward + not_done * self.discount * target_Q

			current_Q1, current_Q2 = self.critic(state, action)
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()


			# Pertubation Model / Action Training
			sampled_actions = self.vae.decode(state)
			perturbed_actions = self.actor(state, sampled_actions)

			# Update through DPG
			actor_loss = -self.critic.q1(state, perturbed_actions).mean()
		 	 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()


			# Update Target Networks 
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



	
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

		# self.current_id = 0
		# self.current_reward = 0

	def select_action(self, state):		
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
			state = (state - self.state_mean) / self.state_std
			dist = self.actor(state)
			action = dist.sample()
			action = action * self.action_std + self.action_mean
			action = action.clamp(-self.max_action, self.max_action)
			# print(action)
		return action.cpu().data.numpy().flatten()
	
	def load_pth(self):
		actor_path = os.path.dirname(os.path.abspath(__file__)) + "/models/AWR_2500_actor.pth"
		critic_path = os.path.dirname(os.path.abspath(__file__)) + "/models/AWR_2500_critic.pth"
		self.actor.load_state_dict(torch.load(actor_path))
		self.critic.load_state_dict(torch.load(critic_path))
		# self.best_actor = copy.deepcopy(self.actor)
		# self.best_critic = copy.deepcopy(self.critic)

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

	def set_current_reward(self, reward):
		self.current_reward = reward

	def train(self, dataset, iterations, batch_size = 100):
		print('AWR training...')
		# print('current reward: ', self.current_reward)
		# if self.current_reward == 0 or self.current_reward < 2600:
		# 	print('use best model')
		# 	self.actor = copy.deepcopy(self.best_actor)
		# 	self.critic = copy.deepcopy(self.best_critic)
		# set grad to true
		self.actor.train()
		self.critic.train()
		state_, action_, reward_, next_state_, not_done_ = sample_whole_process(dataset, batch_size)
		for it in tqdm.tqdm(range(iterations)):
			# idx = self.current_id
			# self.current_id = (self.current_id + 1) % len(state_)
			idx = np.random.randint(0, len(state_))
			# idx = 24
			state, action, reward, next_state, not_done = state_[idx], action_[idx], reward_[idx], next_state_[idx], not_done_[idx]
			state, action = self.normalizer(state, action)
			# calculate discounted reward using monte carlo method
			discounted_reward = torch.zeros_like(reward)
			cur_value = self.critic(state).detach()
			# print(cur_value)
			discounted_reward[-1] = reward[-1] + self.discount * cur_value[-1]
			for t in reversed(range(0, len(reward)-1)):
				cur_reward = reward[t]
				next_return = discounted_reward[t+1]
				discounted_reward[t] = cur_reward + self.discount * ((0.5) * cur_value[t + 1] + 0.5 * next_return)
			steps_per_shuffle = int(np.ceil(len(state) / batch_size))
			# update critic
			sample_idx = np.array(range(len(state)))
			num_idx = len(state)
			for b in range(self.critic_update_iter):
				if b % steps_per_shuffle == 0:
					np.random.shuffle(sample_idx)

				batch_idx_beg = b * batch_size
				batch_idx_end = min((b + 1) * batch_size, len(state))
				batch_idx = np.array(range(batch_idx_beg, batch_idx_end), dtype=np.int32)
				batch_idx = np.mod(batch_idx, num_idx)
				critic_batch_idx = sample_idx[batch_idx]
				cur_value = self.critic(state[critic_batch_idx])			
				critic_loss = F.mse_loss(cur_value, discounted_reward[critic_batch_idx])
				self.critic_optimizer.zero_grad()
				critic_loss.backward()
				self.critic_optimizer.step()

			# calculate discounted reward using monte carlo method
			discounted_reward = torch.zeros_like(reward)
			cur_value = self.critic(state).detach()
			discounted_reward[-1] = reward[-1] + self.discount * cur_value[-1]
			for t in reversed(range(0, len(reward)-1)):
				cur_reward = reward[t]
				next_return = discounted_reward[t+1]
				discounted_reward[t] = cur_reward + self.discount * (0.05 * cur_value[t + 1] + 0.95 * next_return)
			advantage = discounted_reward - cur_value
			advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

			# update actor
			for b in range(self.actor_update_iter):
				if b % steps_per_shuffle == 0:
					np.random.shuffle(sample_idx)
				batch_idx_beg = b * batch_size
				batch_idx_end = min((b + 1) * batch_size, len(state))
				batch_idx = np.array(range(batch_idx_beg, batch_idx_end), dtype=np.int32)
				batch_idx = np.mod(batch_idx, num_idx)
				actor_batch_idx = sample_idx[batch_idx]
				weight = torch.min(torch.exp(advantage[actor_batch_idx]), 20 * torch.ones_like(advantage[actor_batch_idx])).float().reshape(-1, 1)
				cur_policy = self.actor(state[actor_batch_idx])
				log_prob = cur_policy.log_prob(action[actor_batch_idx])
				actor_loss = - torch.mean(weight * log_prob)
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()

