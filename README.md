# Cost-effective bidding under partially observable prices landscapes

Real-time bidding has become a major mean for online advertisement exchange. The goal of a real-time bidding strategy is to maximize the benefits for stakeholders, e.g. click-through rates or conversion rates. 
However, in practise, the optimal bidding strategy for real-time bidding is constrained by three aspects: cost-effectiveness, the dynamic nature of market prices, and the issue of information scarcity in bidding values. 
To address these challenges, we propose Imagine and Imitate Bidding (IIBidder), which includes Strategy Imitation and Imagination modules, to generate cost-effective bidding strategies under partially observable prices landscapes.  
Experimental results on the iPinYou and YOYI datasets demonstrate that IIBidder reduces investment costs, optimizes bidding strategies, and improves future market price predictions.

codes are to be released soon


## Usage

1. Install Python 3.9. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can download the original datasets from [IPINYOU](https://contest.ipinyou.com/) and [YOYI](https://apex.sjtu.edu.cn/datasets/7),
After downloading, preprocess the datasets as follows: 
(1) Group the datasets by advertiser objects, resulting in nine sub-datasets. Split the datasets into training and testing sets in a 2:1 ratio. 
(2) Retain the original columns 'click' and 'bid_price', convert the discrete numerical values of 'click' into continuous values and use regression models to learn other feature information to fit 'click', generating a third column 'ctr' representing click-through rate
(3) Save the processed data as a .txt file format. 
Partially processed datasets will be provided as examples for this project.

3. Train and evaluate model. You can adjust parameters in global.py and reproduce the experiment results as the following examples:

```
python3 main.py
```

4. Check the results
-  results are in .csv format at folder 
- figures are at folder
- latex tables are at folder


## Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{luo2024cost,
  title={Cost-effective bidding under partially observable prices landscapes},
  author={Xiaotong Luo, Yin Tang},
  journal={},
  year={2024},
}
```

## Contact

If you have any questions or suggestions, feel free to contact:

- Xiaotong Luo()
- Yin Tang (ytang@jnu.edu.cn)


Or describe it in Issues.

## Acknowledgement

This work is supported by National Natural Science Foundation of China (62272198)


