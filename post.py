import os
import pandas as pd
from globals import common_param, post_param


def concatenate_xlsx_files():
    folder_path = common_param['path']
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
    all_data = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
    all_data = all_data.dropna()
    return all_data


def find_want(df, ours):
    our_method = df[df['method'] == ours]
    best = our_method.groupby(post_param['id_vars']).agg(post_param['best_dict'])
    best = best.reset_index()
    other_method = df[df['method'] != ours]
    worst = other_method.groupby(post_param['id_vars']).agg(post_param['worst_dict'])
    worst = worst.reset_index()
    df_wanted = pd.concat([best, worst], ignore_index=True)
    return df_wanted



def post(ours='IIBidder'):
    data = concatenate_xlsx_files()
    result_df = find_want(data, ours)
    output_file=post_param['output_file']
    result_df.to_excel(output_file, index=False)
    print("saved to", output_file)
    return result_df

if __name__ == '__main__':
    post()