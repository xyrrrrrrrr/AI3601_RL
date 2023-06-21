import torch
import torch.nn.functional as F


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, beta, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda  ###### The parameter used to compute advantage function
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.device = device
        self.beta = beta
        self.action_dim = action_dim

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        """ ------------- Programming 4: Compute Advantage Function ------------- """
        """ YOUR CODE HERE """
        # compute td delta by using Bellman equation
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        # compute advantage
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta)
        # normalize advantage
        # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-12)
        # initalize old_log_probs for computing ratio
        old_probs = self.actor(states)
        dist = torch.distributions.Categorical(old_probs)
        old_log_probs = dist.log_prob(actions.squeeze(1)).detach()
        old_probs = old_probs.view((-1, self.action_dim)).detach()
        """ ------------- Programming 4 ------------- """

        for _ in range(self.epochs):
            """ ------------- Programming 5: Update the parameter of actor and critic (you may refer to original PPO paper) ------------- """
            """ YOUR CODE HERE """
            from torch.distributions import Categorical
            from torch.nn.functional import kl_div
            # compute new_log_probs
            new_probs = self.actor(states)
            dist = Categorical(new_probs)
            new_log_probs = dist.log_prob(actions.squeeze(1))
            # compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            # compute surrogate loss
            surrogate_loss = ratio * advantage
            # compute kl divergence(penalty)
            penalty = kl_div(input = new_probs.log(), target = old_probs.detach(), reduction='batchmean')
            # update actor
            self.actor_optimizer.zero_grad()
            actor_loss = - (surrogate_loss - self.beta * penalty).mean()
            actor_loss.backward()
            self.actor_optimizer.step()
            # update critic
            self.critic_optimizer.zero_grad()
            critic_loss = F.mse_loss(self.critic(states), td_target.detach())
            critic_loss.backward()
            self.critic_optimizer.step()
        # I want to update beta but it's not required.
        # dl = kl_div(input=self.actor(states).log(),
        #             target=old_probs.detach(), reduction='sum')
        # # print(dl)
        # if abs(dl) >= 1.5 * 1e-2:
        #     self.beta *= 2
        # if abs(dl) <= 1e-2 / 1.5:
        #     self.beta *= 0.5
        # print(self.beta)
        """ ------------- Programming 5 ------------- """





