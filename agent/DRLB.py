import random
import torch
import torch.nn as nn
import math 

from agent.network import Q_network

class DRLB_agent:
    def __init__(self, state_dim, action_dim, hidden_dim, memory_size, lr, gamma, batch_size,
                 epsilon_start, epsilon_end, epsilon_decay, device):
        self.state_size = state_dim
        self.action_size = action_dim
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.device = device
        # e-greedy
        self.frame_idx = 0  
        self.epsilon = lambda frame_idx: epsilon_end + \
            (epsilon_start - epsilon_end) * \
            math.exp(-1. * frame_idx / epsilon_decay)
    
        self.memory_buffer = Memory_Buffer(memory_size)
        self.q_network = Q_network(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_network = Q_network(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def select_action(self, state):
        self.frame_idx += 1
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if random.random() <= self.epsilon(self.frame_idx):
            return random.randrange(self.action_size)
        else:
            return torch.argmax(self.q_network(state)).item()

    def update(self):
        if self.memory_buffer.size() < self.batch_size:
            return 0
        # 从memory中随机采样transition
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory_buffer.sample(
            self.batch_size)
        '''转为torch.tensor
        '''
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)  # [batch_size, action_dim]
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)  
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(done_batch, device=self.device, dtype=torch.float)

        '''计算当前(s_t,a)对应的Q(s_t, a)'''
        q_values = self.q_network(state_batch).gather(dim=1, index=action_batch)  
        # 计算所有next states的V(s_{t+1})，即通过target_net中选取reward最大的对应states
        next_q_values = self.target_q_network(next_state_batch).max(1)[0].detach()  # 比如tensor([ 0.0060, -0.0171,...,])
        
        # 对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + self.gamma * next_q_values * (1-done_batch)
        loss = self.loss_func(q_values.squeeze(), expected_q_values)  # 计算 均方误差loss
       
        # update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()  
        return loss.item()

    def target_network_update(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())


class Memory_Buffer(object):
    def __init__(self, memory_size=100000):
        self.buffer = []
        self.memory_size = memory_size
        self.next_idx = 0
        
    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) <= self.memory_size: # buffer not full
            self.buffer.append(data)
        else: # buffer is full
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            
            idx = random.randint(0, self.size() - 1)
            data = self.buffer[idx]
            state, action, reward, next_state, done= data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
        return states, actions, rewards, next_states, dones
    
    def size(self):
        return len(self.buffer)