import os
import pickle
import pandas as pd
import numpy as np
from globals import data_path, common_param


# simulate the missing data 
def mask_processing(data, threshold):
    mask = np.random.random(size=data.shape)
    mask = (mask > threshold).astype(int)
    masked_data = mask * data
    return masked_data

def get_camp_info(data_path, camp):
    camp_info = pickle.load(open(data_path + os.sep + 'info_'+camp+'.txt', "rb"))
    test_budget = camp_info['cost_test']
    train_budget = camp_info['cost_train']
    test_imp = camp_info['imp_test']
    train_imp = camp_info['imp_train']
    return test_budget, train_budget, test_imp, train_imp


def read_data(camp):
    if camp in common_param['yoyi_ids']:
        read_test = pd.read_csv(data_path + os.sep + 'test.theta_'+ camp +'.txt',
                                    header=None, index_col=False, sep='\t',names=['click', 'winprice', 'pctr'])
        read_train = pd.read_csv(data_path + os.sep + 'train.theta_'+camp+'.txt',
                                    header=None, index_col=False, sep='\t', names=['click', 'winprice', 'pctr'])
    elif camp in common_param['ipinyou_ids']:
        read_test = pd.read_csv(data_path + os.sep + 'test.theta_'+ camp +'.txt',
                                    header=None, index_col=False, sep=' ',names=['click', 'winprice', 'pctr'])
        read_train = pd.read_csv(data_path + os.sep + 'train.theta_'+camp+'.txt',
                                    header=None, index_col=False, sep=' ', names=['click', 'winprice', 'pctr'])
    else:
        print('{} not in the dataset'.format(camp))
    return read_train, read_test


def data_preprocessed(camp_n, mask):
    train_file_dict = {}
    test_file_dict = {}
    total_budget = 0
    total_impressions = 0    
    for camp in camp_n:
        read_train, read_test = read_data(camp)
        masked_train = mask_processing(read_train, mask)
        masked_test = mask_processing(read_test, mask)        
        test_budget, train_budget, test_imp, train_imp = get_camp_info(data_path, camp)
        train = {'imp':train_imp, 'budget':train_budget, 'data':masked_train}
        test = {'imp':test_imp, 'budget':test_budget, 'data':masked_test}

        total_budget += train_budget
        total_impressions += train_imp
        train_file_dict[camp] = train
        test_file_dict[camp] = test
    
    return train_file_dict, test_file_dict


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
