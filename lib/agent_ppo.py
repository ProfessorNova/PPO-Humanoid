import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class PPOAgent(nn.Module):
    def __init__(self, num_inputs: int, num_actions: int):
        super(PPOAgent, self).__init__()

        # Actor Network for mu
        actor_hid1_size = num_inputs * 20
        actor_hid3_size = num_actions * 10
        actor_hid2_size = int(np.sqrt(actor_hid1_size * actor_hid3_size))
        self.actor_mu = nn.Sequential(
            nn.Linear(num_inputs, actor_hid1_size),
            nn.Tanh(),
            nn.Linear(actor_hid1_size, actor_hid2_size),
            nn.Tanh(),
            nn.Linear(actor_hid2_size, actor_hid3_size),
            nn.Tanh(),
            nn.Linear(actor_hid3_size, num_actions),
            nn.Tanh()  # [-1, 1]
        )

        # Diagonal covariance matrix variables are separately trained
        self.actor_logstd = nn.Parameter(torch.ones(1, num_actions) * -0.5)

        # Critic Network
        critic_hid1_size = num_inputs * 20
        critic_hid3_size = 10
        critic_hid2_size = int(np.sqrt(critic_hid1_size * critic_hid3_size))
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, critic_hid1_size),
            nn.Tanh(),
            nn.Linear(critic_hid1_size, critic_hid2_size),
            nn.Tanh(),
            nn.Linear(critic_hid2_size, critic_hid3_size),
            nn.Tanh(),
            nn.Linear(critic_hid3_size, 1)
        )

    def forward(self, x):
        mu = self.actor_mu(x)
        std = torch.exp(self.actor_logstd).expand_as(mu)
        return mu, std

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mu, std = self.forward(x)
        dist = Normal(mu, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, self.get_value(x)
