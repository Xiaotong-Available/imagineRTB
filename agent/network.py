import torch
import torch.nn as nn
import pickle
import os
import numpy as np
from torch.distributions.categorical import Categorical


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state):
        dist = self.actor(state)
        return dist


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        value = self.critic(x)
        return value


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        if len(action.shape) == 1:
            action = torch.unsqueeze(action, dim=1)
        state_action = torch.cat([state, action], 1)
        x = torch.sigmoid(self.fc(state_action))
        return x


class Q_network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Q_network, self).__init__()
        self.Q_FC = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        value = self.Q_FC(state)
        return value


class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ICM, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.inverse_net = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.forward_net = nn.Sequential(
            nn.Linear(action_dim+hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, state, next_state, action):
        # encode state
        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)
        # get pred next state
        pred_next_state_feature = self.forward_net(torch.cat((encode_state, action.squeeze()), 1))
        real_next_state_feature = encode_next_state
        return pred_next_state_feature, real_next_state_feature, pred_action


class RTB_ExpertTraj:
    def __init__(self, camp_id, expert_traj_path):
        self.camp_path = os.path.join(expert_traj_path, "expert_" + camp_id +".pkl")
        self.expert = pickle.load(open(self.camp_path, "rb"))
        self.n_transitions = len(self.expert)
    
    def sample(self, batch_size):
        indexes = np.random.randint(0, self.n_transitions, size=batch_size)
        state, action = [], []
        for i in indexes:
            s = self.expert[i][0]
            a = self.expert[i][1]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
        return np.array(state), np.array(action)
    

class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=20, output_size=1):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(1,0)
        out, hidden = self.rnn(x, None)
        out = self.fc(out[:, -1]) 
        return out