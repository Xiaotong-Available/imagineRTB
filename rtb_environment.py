
import numpy as np


class RTB_environment:
    """
    This class will construct and manage the environments which the
    agent will interact with. The distinction between training and
    testing environment is primarily in the episode length.
    """
    def __init__(self, camp_dict, episode_length, step_length):
        """
        We need to initialize all of the data, which we fetch from the
        campaign-specific dictionary. We also specify the number of possible
        actions, the state, the amount of data which has been trained on,
        and so on.
        :param camp_dict: a dictionary containing data on winning bids, ctr
        estimations, clicks, budget, and so on. We copy the data on bids, ctr
        estimations and clicks; then, we delete the rest of the dictionary.
        :param episode_length: specifies the number of steps in an episode
        :param step_length: specifies the number of auctions per step.
        """
        self.camp_dict = camp_dict
        self.data_count = camp_dict['imp']

        self.result_dict = {'auctions':0, 'impressions':0, 'click':0, 'cost':0, 'win-rate':0, 'eCPC':0, 'eCPI':0, 'CER':0, 'WRC':0}

        self.actions = [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08]

        self.step_length = step_length
        self.episode_length = episode_length

        self.Lambda = 1
        self.time_step = 0
        self.budget = 1
        self.init_budget = 1
        self.n_regulations = 0
        self.budget_consumption_rate = 0
        self.winning_rate = 0
        self.cost = 0
        self.ctr_value = 0
        self.click = 0
        self.impressions = 0
        self.termination = True

        self.state = [self.budget / self.init_budget, self.n_regulations,
                      self.budget_consumption_rate,
                      self.winning_rate, self.ctr_value]

    def update_camp_data(self):
        """
        This function updates the data variables which are then accessible
        to the step-function. This function also deletes data that has already
        been used and, hence, tries to free up space.
        :return: updated data variables (i.e. bids, ctr estimations and clicks)
        """
        if self.data_count < self.step_length:
            ctr_estimations = np.array(self.camp_dict['data'].iloc[:self.data_count, :]['pctr'])
            winning_bids = np.array(self.camp_dict['data'].iloc[:self.data_count, :]['winprice'])
            clicks = list(self.camp_dict['data'].iloc[:self.data_count, :]['click'])

            self.data_count = 0
            return ctr_estimations, winning_bids, clicks
        else:
            ctr_estimations = np.array(
                self.camp_dict['data'].iloc[self.data_count - self.step_length:self.data_count, :]['pctr'])
            winning_bids = np.array(
                self.camp_dict['data'].iloc[self.data_count - self.step_length:self.data_count, :]['winprice'])
            clicks = list(
                self.camp_dict['data'].iloc[self.data_count - self.step_length:self.data_count, :]['click'])

            self.data_count -= self.step_length
            return ctr_estimations, winning_bids, clicks

    def reset(self, budget, initial_Lambda):
        """
        This function is called whenever a new episode is initiated
        and resets the budget, the Lambda, the time-step, the termination
        bool, and so on.
        :param budget: the amount of money the bidding agent can spend during
        the period
        :param initial_Lambda: the initial scaling of ctr-estimations to form bids
        :return: initial state, zero reward and a false termination bool
        """
        self.n_regulations = min(self.episode_length, self.data_count / self.step_length)
        self.budget = budget
        self.init_budget = budget * self.episode_length / self.n_regulations
        self.Lambda = initial_Lambda
        self.time_step = 0

        ctr_estimations, winning_bids, clicks = self.update_camp_data()
        bids = [int(i * (1 / self.Lambda)) for i in ctr_estimations]
        budget = self.budget
        self.budget_consumption_rate = 0
        self.winning_rate = 0
        self.ctr_value = 0
        self.click = 0
        self.impressions = 0

        for i in range(min(self.data_count, self.step_length)):
            if bids[i] > winning_bids[i] and budget > bids[i]:
                budget -= winning_bids[i]
                self.impressions += 1
                self.cost += winning_bids[i]
                self.click += clicks[i]
                self.ctr_value += ctr_estimations[i]
                self.winning_rate += 1 / min(self.data_count, self.step_length)
            else:
                continue

        self.state = [self.budget / self.init_budget, self.n_regulations,
                      self.budget_consumption_rate,
                      self.winning_rate, self.ctr_value]

        self.budget_consumption_rate = (self.budget - budget) / self.budget
        self.budget = budget
        self.n_regulations -= 1
        self.time_step += 1

        reward = self.ctr_value
        self.termination = False

        return self.state, reward, self.termination

    def step(self, action_index):
        """
        This function takes an action from the bidding agent (i.e.
        a change in the ctr-estimation scaling, and uses it to compute
        the agent's bids, s.t. it can compare it to the "winning bids".
        If one of the agent's bids exceed a winning bid, it will subtract
        the cost of the impression from the agent's budget, etc, given that
        the budget is not already depleted.
        :param action_index: an index for the list of allowed actions
        :return: a new state, reward and termination bool (if time_step = 96)
        """
        action = self.actions[action_index]
        self.Lambda = self.Lambda*(1 + action)
        ctr_estimations, winning_bids, clicks = self.update_camp_data()

        bids = [int(i*(1/self.Lambda)) for i in ctr_estimations]
        budget = self.budget
        self.click = 0
        self.cost = 0
        self.ctr_value = 0
        self.winning_rate = 0
        self.impressions = 0

        for i in range(min(self.data_count, self.step_length)):
            if bids[i] > winning_bids[i] and budget > bids[i]:
                budget -= winning_bids[i]
                self.impressions += 1
                self.cost += winning_bids[i]
                self.click += clicks[i]
                self.ctr_value += ctr_estimations[i]
                self.winning_rate += 1 / min(self.data_count, self.step_length)
            else:
                continue

        self.result_dict['impressions'] += self.impressions
        self.result_dict['click'] += self.click
        self.result_dict['cost'] += self.cost
        self.result_dict['win-rate'] += self.winning_rate * min(self.data_count, self.step_length) / self.camp_dict['imp']

        self.budget_consumption_rate = (self.budget - budget) / self.budget
        self.budget = budget
        self.n_regulations -= 1
        self.time_step += 1

        if self.time_step == self.episode_length or self.data_count == 0:
            self.termination = True

        self.state = [self.budget / self.init_budget, self.n_regulations,
                      self.budget_consumption_rate,
                      self.winning_rate, self.ctr_value]

        reward = self.ctr_value

        return self.state, reward, self.termination

    def result(self):
        """
        This function returns some statistics from the episode or test
        :return: number of impressions won, number of
        actual clicks, winning rate, effective cost per click,
        and effective cost per impression.
        """
        if self.result_dict['click'] == 0:
            self.result_dict['eCPC'] = 0
        else:
            self.result_dict['eCPC'] = self.result_dict['cost'] / self.result_dict['click']
            self.result_dict['eCPI'] = self.result_dict['cost'] / self.result_dict['impressions'],
            self.result_dict['CER'] = self.result_dict['click'] **2 / self.result_dict['cost'],
            self.result_dict['WRC'] = self.result_dict['win-rate'] / self.result_dict['cost'],

        return self.result_dict['impressions'], self.result_dict['click'], self.result_dict['cost'], \
               self.result_dict['win-rate'], self.result_dict['eCPC'], self.result_dict['eCPI'], \
                self.result_dict['CER'], self.result_dict['WRC']
    
def drl_test(test_file_dict, budget, initial_Lambda, agent, episode_length, step_length):
    test_environment = RTB_environment(test_file_dict, episode_length, step_length)
    budget_list = []
    Lambda_list = []
    episode_budget = 0

    while test_environment.data_count > 0:
        episode_budget = min(episode_length * step_length, test_environment.data_count)\
                        / test_file_dict['imp'] * budget + episode_budget
        state, reward, termination = test_environment.reset(episode_budget, initial_Lambda)
        while not termination:
            action_or = agent.select_action(state)
            if isinstance(action_or, tuple):
                action, _ = action_or
            else:
                action = action_or
            next_state, reward, termination = test_environment.step(action)
            state = next_state

            budget_list.append(test_environment.budget)
            Lambda_list.append(test_environment.Lambda)
        episode_budget = test_environment.budget
    impressions, click, cost, win_rate, ecpc, ecpi, cer, wrc = test_environment.result()

    return impressions, click, cost, win_rate, ecpc, ecpi, cer, wrc, \
        [np.array(budget_list).tolist(), np.array(Lambda_list).tolist()]


if __name__ == '__main__':
    rtn_env = RTB_environment()