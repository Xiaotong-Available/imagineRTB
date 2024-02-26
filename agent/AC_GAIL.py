import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from agent.network import Actor, Discriminator, RTB_ExpertTraj


class AC_GAIL:
    def __init__(self, camp_id, state_dim, action_dim, action_value, hidden_dim, lr, device, exper_traj_path):
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        self.discriminator = Discriminator(state_dim, action_value, hidden_dim).to(device)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        
        self.expert = RTB_ExpertTraj(camp_id, exper_traj_path)
        self.loss_fn = nn.BCELoss() 
        self.device = device

    def select_action(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        action_prob = self.actor(state)
        action = Categorical(action_prob).sample()
        return action.item()
        
    def update(self, n_iter, batch_size=100):
        for i in range(n_iter):
            # sample expert transitions
            exp_state, exp_action = self.expert.sample(batch_size)
            exp_state = torch.tensor(exp_state, device=self.device, dtype=torch.float32)
            exp_action = torch.tensor(exp_action, device=self.device, dtype=torch.float32)
            
            # sample expert states for actor
            state, _ = self.expert.sample(batch_size)
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            action = Categorical(self.actor(state)).sample()
            
            # update discriminator
            self.optim_discriminator.zero_grad()
            
            # label tensors
            exp_label = torch.full((batch_size,1), 1.0, device=self.device)
            policy_label = torch.full((batch_size,1), 0.0, device=self.device)
            # with expert transitions
            prob_exp = self.discriminator(exp_state, exp_action)
            loss = self.loss_fn(prob_exp, exp_label)
            # with policy transitions
            prob_policy = self.discriminator(state, action.detach())
            loss += self.loss_fn(prob_policy, policy_label)
            loss.backward()
            self.optim_discriminator.step()
            
            # update policy
            self.optim_actor.zero_grad()
            loss_actor = -self.discriminator(state, action)
            loss_actor.mean().backward()
            self.optim_actor.step()
            
        