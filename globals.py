import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
data_path = os.path.join(os.getcwd(), 'datasets')



common_param = {
    "dataset_list": ['ipinyou'],  # 数据集类型有ipinyou和yoyi两种，仅提供ipinyou
    # "ipinyou_ids": ['1458', '2259', '2261', '2821', '2997', '3358',  '3386', '3427', '3476'], 
    'ipinyou_ids': ['2997'], # ipinyou的子数据集
    "yoyi_ids": [], # 'yoyi'子数据集为'yoyi'
    "budget_scalings": [1/32, 1/16, 1/8, 1/4, 1/2], # 预算水平
    "mask_list": [0, 0.1, 0.2, 0.3, 0.4, 0.5], # 数据缺失率
    "ablation_list": ['IIBidder', 'AC_GAIL', 'PPO_GAIL', 'PPO'], # 消融实验对象
    "compared_list": ['DRLB', 'Lin', 'Gamma', 'Uniform', 'Normal'], # 对比实验对象
    "initial_Lambda": 0.0001,  # 初始参数调整预算值
    "budget_init_var": 2500,  # 预算初始设置
    "step_length": 500, # 每个周期长度
    "seed": 1,
    "episode_length": 96, # 每个步骤长度
    "path": os.path.join(os.getcwd(), "result"),
}


train_param = {
    "state_dim": 5,
    "action_dim": 7,
    "hidden_dim": 100,
    "memory_size": 100000,
    "lr": 0.001, # ② 0.01
    "gamma": 0.99,
    "batch_size": 32,
    "epsilon_start": 0.9,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.99,
    "device": device,
    "update_freq": 10, #50

    # GAIL
    "action_value": 1, 
    "n_iter": 200, # ③ 200
    "expert_traj_path": os.path.join(os.getcwd(), "expert_traj"),
    "update_frequency": 128
}

# post
post_param = {
    'id_vars': ['method', 'dataset', 'mask', 'budget'],
    'best_dict': {'click': 'max', 'win_rate': 'max', 'ecpc': 'min', 'cer': 'max', 'wrc': 'max'},
    'worst_dict': {'click': 'min', 'win_rate': 'min', 'ecpc': 'max', 'cer': 'min', 'wrc': 'min'},
    'output_file' : 'final.xlsx'
}

# plot
plot_param = {
    'id_vars': ['method', 'dataset', 'mask', 'budget'], # 聚合对象
    'value_vars': ['cer', 'wrc'], # 聚合后的值被操作的对象
    'save_type' : ".pdf",
    'columns' : ['method','dataset','metric','hp1','hp2','value'],
    'dis_pal' : "Set2", # 离散调色板
    'con_pal' : "Reds", # 连续调色板
    'tabs' : "tables/",
    'figs' : "figures/",
    'save_it' : True,
    'compared' : ['IIBidder','DRLB','Lin', 'Normal', 'Uniform', 'Gamma'], # 绘制对象
    'ablation' : ['IIBidder','PPO_GAIL','AC_GAIL', 'PPO']

}