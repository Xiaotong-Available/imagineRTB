import pandas as pd
from datetime import datetime
import os
import time
from multiprocessing import Process
from preprocess import data_preprocessed, mkdir
from ablation_exp import ablation_exp
from globals import common_param
from compared_exp import compared_exp

def process_experiment(method, dataset, mask, budget_scaling, train_file_dict, test_file_dict, camp_n):
    start = time.time()
    if method in common_param['ablation_list']:
        result_table = ablation_exp(method, train_file_dict, test_file_dict, dataset, budget_scaling)
    elif method in common_param['compared_list']:
        result_table = compared_exp(method, train_file_dict, test_file_dict, camp_n, budget_scaling)

    imp = result_table["impressions"].mean()
    click = result_table["clicks"].mean()
    win_rate = result_table["win_rate"].mean()
    ecpc = result_table["ecpc"].mean()
    cost = result_table['cost'].mean()
    cer = click **2 / cost
    wrc = win_rate / cost
    time_cost = time.time() - start
    print("testing {} // mask {} // budget {} //---click: {}, win_rate: {}, ecpc: {}, cer: {}, wrc: {}".
            format(method, mask, budget_scaling, click, win_rate, ecpc, cer, wrc))
    
    return {
        'method': method,
        'dataset': dataset,
        'mask': mask,
        'budget': budget_scaling,
        'impresions': imp,
        'click': click,
        'win_rate': win_rate,
        'ecpc': ecpc,
        'cer': cer,
        'wrc': wrc,
        'cost_time': time_cost / 60,
    }

def run():
    # Select the list of models for experimentation, which consists of ablation models and comparison models
    method_list = common_param['ablation_list'] + common_param['compared_list']
    budget_scalings = common_param["budget_scalings"]
    mask_list = common_param['mask_list']
    dataset_list = common_param['dataset_list']
    results = []

    # Explore the bidding performance under different scenarios of 
    # data missing, budget levels, datasets, and methods
    for mask in mask_list:
        print("the mask is {}".format(mask))
        for budget_scaling in budget_scalings:
            print("the budget_scaling is {}".format(budget_scaling))
            for dataset in dataset_list:
                print("dataset:", dataset)
                train_file_dict, test_file_dict = data_preprocessed(common_param["{}_ids".format(dataset)], mask)
                for method in method_list:
                    camp_n = common_param["{}_ids".format(dataset)]
                    result_dict = process_experiment(method, dataset, mask, budget_scaling, train_file_dict, test_file_dict, camp_n)
                    results.append(result_dict)
    result_df = pd.DataFrame(results)
                
    # define the output path
    dir_path = os.path.join(common_param["path"])  # , "exp_num{}".format(num)
    mkdir(dir_path)
    localtime = time.strftime("%m%d%H%M", time.localtime()) 
    result_df.to_csv(os.path.join(dir_path, "results{}.csv".format(localtime)), index=False)
    print("saving result to {}".format(os.path.join(dir_path, "results{}.csv".format(localtime))))


if __name__ == '__main__':
    run()






    