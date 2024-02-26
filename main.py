from run import run
from post import post
from plot import plot_and_tex
import pandas as pd

def main(df):
    # 输出中间结果到 result/
    run()
    # 输出单独final.csv
    post()
    # 输出图和表到 figures
    plot_and_tex(df)

if __name__ == '__main__':
    df = pd.read_excel('final.xlsx')
    main(df)
