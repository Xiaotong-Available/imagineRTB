import pandas as pd
import numpy as np


def lin_test(train_file_dict, test_file_dict, budget, type_of_average="base"):
    click = list(test_file_dict['data']['click'])
    winning_bids = list(test_file_dict['data']['winprice'])
    ctr_estimations = list(test_file_dict['data']['pctr'])
    impressions = 0
    clicks = 0
    cost = 0
    win_rate = 0
    ecpc = 0
    ecpi = 0

    if type_of_average == 'historical':
        historical_bid_average = np.array(train_file_dict['data']['winprice']).mean()
        historical_ctr_average = np.array(train_file_dict['data']['pctr']).mean()

        for i in range(test_file_dict['imp']):
            bid = historical_bid_average * ctr_estimations[i] / historical_ctr_average
            if bid > winning_bids[i] and bid < budget:
                impressions += 1
                budget -= winning_bids[i]
                clicks += click[i]
                cost += winning_bids[i]
                win_rate += 1 / test_file_dict['imp']
            else:
                continue

    elif type_of_average == 'base':
        base_bid = budget / train_file_dict['imp']
        historical_ctr_average = np.array(train_file_dict['data']['pctr']).mean()

        for i in range(test_file_dict['imp']):
            bid = base_bid * ctr_estimations[i] / historical_ctr_average
            if bid > winning_bids[i] and bid < budget:
                impressions += 1
                budget -= winning_bids[i]
                clicks += click[i]
                cost += winning_bids[i]
                win_rate += 1 / test_file_dict['imp']
            else:
                continue
            
    if clicks > 0:
        ecpc = cost / clicks
        cer = clicks**2 / cost
        wrc = win_rate / cost
    else:
        cer = 0
        wrc = 0
    if impressions > 0:
        ecpi = cost / impressions
  
    print("testing Lin--- click: {}, win_rate: {}, ecpc: {}, cer:{}, wrc:{}".format(clicks, win_rate, ecpc, cer, wrc))
        
    return impressions, clicks, cost, win_rate, ecpc, ecpi, cer, wrc
