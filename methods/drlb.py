import os
import numpy as np
import pandas as pd
import torch
from agent.DRLB import DRLB_agent
from rtb_environment import RTB_environment
from globals import common_param, train_param
from globals import common_param


def DRLB_test(test_file_dict, budget, initial_Lambda, agent, episode_length, step_length):
    test_environment = RTB_environment(test_file_dict, episode_length, step_length)
    budget_list = []
    Lambda_list = []
    episode_budget = 0

    while test_environment.data_count > 0:
        episode_budget = min(episode_length * step_length, test_environment.data_count)\
                         / test_file_dict['imp'] * budget + episode_budget
        state, reward, termination = test_environment.reset(episode_budget, initial_Lambda)
        while not termination:
            action = agent.select_action(state)
            next_state, reward, termination = test_environment.step(action)
            state = next_state

            budget_list.append(test_environment.budget)
            Lambda_list.append(test_environment.Lambda)
        episode_budget = test_environment.budget
    impressions, click, cost, win_rate, ecpc, ecpi, cer, wrc = test_environment.result()

    return impressions, click, cost, win_rate, ecpc, ecpi, cer, wrc,\
           [np.array(budget_list).tolist(), np.array(Lambda_list).tolist()]


def DRLB_train(train_file_dict, test_file_dict, budget_scaling):
    seed = common_param["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    initial_Lambda = common_param["initial_Lambda"]
    budget_init_var = common_param["budget_init_var"]

    episode_length = common_param["episode_length"]
    step_length = common_param["step_length"]

    state_dim = train_param["state_dim"]
    action_dim = train_param["action_dim"]
    hidden_dim = train_param["hidden_dim"]
    memory_size = train_param["memory_size"]
    lr = train_param["lr"]
    gamma = train_param["gamma"]
    batch_size = train_param["batch_size"]
    epsilon_start = train_param["epsilon_start"]
    epsilon_end = train_param["epsilon_end"]
    epsilon_decay = train_param["epsilon_decay"]
    device = train_param["device"]
    update_freq = train_param["update_freq"]


    total_budget = 0
    total_impressions = 0
    global_step_counter = 0


    total_budget += train_file_dict["budget"]
    total_impressions += train_file_dict["imp"]

    env = RTB_environment(train_file_dict, episode_length, step_length)
    agent = DRLB_agent(state_dim, action_dim, hidden_dim, memory_size, lr, gamma, batch_size,
                        epsilon_start, epsilon_end, epsilon_decay, device)
    
    while env.data_count > 0:
        # get train budget
        episode_size = min(episode_length * step_length, env.data_count)
        train_budget = train_file_dict['budget'] * episode_size \
                    / train_file_dict['imp'] * budget_scaling
        train_budget = np.random.normal(train_budget, budget_init_var)
        # reset environment
        state, reward, termination = env.reset(train_budget, initial_Lambda)

        while not termination:
            action = agent.select_action(state)
            next_state, reward, termination = env.step(action)
            agent.memory_buffer.push(state, action, reward, next_state, termination)
            agent.update()
            if global_step_counter % update_freq == 0:
                agent.target_network_update()

            state = next_state
            global_step_counter += 1
            
                
    test_budget = total_budget / total_impressions * test_file_dict['imp'] * budget_scaling
    imp, click, cost, win_rate, ecpc, ecpi, cer, wrc, camp_info = \
        DRLB_test(test_file_dict, test_budget, initial_Lambda, agent, episode_length, step_length)
    

    # print("testing DRLB {}--- click: {}, win_rate: {}, ecpc: {}, cer: {}, wrc:{}".format(list(train_file_dict.keys())[0], click, win_rate, ecpc, cer, wrc))
    return imp, click, cost, win_rate, ecpc, ecpi, cer, wrc


