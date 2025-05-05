import torch
import torch.nn as nn
from torch.distributions import Normal


class PPOAgent(nn.Module):
    def __init__(self, num_inputs: int, num_actions: int):
        super(PPOAgent, self).__init__()

        # Actor Network for mu
        self.actor_mu = nn.Sequential(
            nn.Linear(num_inputs, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, num_actions),
            nn.Tanh()  # [-1, 1]
        )

        # Diagonal covariance matrix variables are separately trained
        self.actor_logstd = nn.Parameter(torch.ones(1, num_actions) * -0.5)

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
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
            action = dist.rsample()  # reparameterization trick because the action is continuous
        log_prob = dist.log_prob(action).sum(-1)  # sum log prob of each action dimension
        entropy = dist.entropy().mean(-1)  # average entropy per action dimension
        return action, log_prob, entropy, self.get_value(x)
