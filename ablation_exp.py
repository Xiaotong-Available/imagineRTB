import os
import pickle
import pandas as pd
import numpy as np
import torch
from collections import namedtuple

from globals import train_param, common_param
from rtb_environment import RTB_environment
from agent.AC_GAIL import AC_GAIL
from agent.ICM_GAIL import ICM_GAIL
from agent.PPO_GAIL import PPO_GAIL
from agent.PPO import PPO
from rtb_environment import drl_test


def ablation_exp(agent_name, train_file_dict, test_file_dict, dataset, budget_scaling):
    """
    This function trains the agent for RTB environment.
    :param budget_scaling: a list of budgets for training and testing
    :param train_file_dict: a dictionary of training data
    :param test_file_dict: a dictionary of testing data
    :param agent_name: the name of the agent
    :return: a list of results for training and testing
    """   
    Transition = namedtuple("Transition", ["state", "action", "reward", "a_log_prob", "next_state"])

    camp_n = common_param["{}_ids".format(dataset)]
    seed = common_param["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    initial_Lambda = common_param["initial_Lambda"]
    budget_init_var = common_param["budget_init_var"]
    episode_length = common_param["episode_length"]
    step_length = common_param["step_length"]

    state_dim = train_param["state_dim"]
    action_dim = train_param["action_dim"]
    hidden_dim = train_param["hidden_dim"]
    lr = train_param["lr"]
    batch_size = train_param["batch_size"]
    device = train_param["device"]
    n_iter = train_param["n_iter"]
    expert_traj_path = train_param["expert_traj_path"]
    action_value = train_param["action_value"]
    update_frequency = train_param["update_frequency"]

    result_table = pd.DataFrame(columns=['impressions', 'clicks', 'cost', 'win_rate', 'ecpc', 'ecpi', 'cer', 'wrc']
                            ,index=camp_n)
    total_budget = 0
    total_impressions = 0
    global_step_counter = 0

    for camp_id in camp_n:
        # init agent
        if agent_name == 'AC_GAIL':
            rtb_agent = AC_GAIL(camp_id, state_dim, action_dim, action_value, 
                        hidden_dim, lr, device, expert_traj_path)
        elif agent_name == 'IIBidder':
            rtb_agent = ICM_GAIL(camp_id, state_dim, action_dim, action_value, 
                        hidden_dim, lr, device, expert_traj_path)
        elif agent_name == 'PPO_GAIL':
            rtb_agent = PPO_GAIL(camp_id, state_dim, action_dim, action_value, 
                        hidden_dim, lr, device, expert_traj_path)
        elif agent_name == 'PPO':
            rtb_agent = PPO(camp_id, state_dim, action_dim, action_value, 
                        hidden_dim, lr, device, expert_traj_path)


        # init environment
        rtb_environment = RTB_environment(train_file_dict[camp_id], episode_length, step_length)
        total_budget += train_file_dict[camp_id]['budget']
        total_impressions += train_file_dict[camp_id]['imp']

        while rtb_environment.data_count > 0:
            episode_size = min(episode_length * step_length, rtb_environment.data_count)
            budget = train_file_dict[camp_id]['budget'] * min(rtb_environment.data_count, episode_size) \
                     / train_file_dict[camp_id]['imp'] * budget_scaling
            budget = np.random.normal(budget, budget_init_var)
            # init state
            state, reward, termination = rtb_environment.reset(budget, initial_Lambda)
            while not termination:
                # get action
                action = rtb_agent.select_action(state)
                if isinstance(action, tuple):
                    action, action_prob = action
                    next_state, reward, termination = rtb_environment.step(action)
                    transition = Transition(state, action, reward, action_prob, next_state)
                    rtb_agent.store_transition(transition)

                else:
                    action = action
                    next_state, reward, termination = rtb_environment.step(action)
                state = next_state

                global_step_counter += 1
                if global_step_counter % update_frequency == 0:
                    rtb_agent.update(n_iter, batch_size)
                
        # test
        budget = total_budget / total_impressions * test_file_dict[camp_id]['imp'] * budget_scaling
        imp, click, cost, win_rate, ecpc, ecpi, cer, wrc, camp_info = drl_test(test_file_dict[camp_id], budget, 
                                        initial_Lambda, rtb_agent, episode_length, step_length)
        
        temp_dict = {"impressions": imp, "clicks": click, "cost": cost, 
                     "win_rate": win_rate, "ecpc": ecpc, "ecpi": ecpi,
                     "cer": cer, 'wrc': wrc}

        result_table.loc[camp_id] = [temp_dict[key] for key in result_table.columns]
        # print("testing {} in {}--- click: {}, win_rate: {}, ecpc: {}, cer:{}, wrc:{}".format(agent_name, camp_id, click, win_rate, ecpc, cer, wrc))
    return result_table

    