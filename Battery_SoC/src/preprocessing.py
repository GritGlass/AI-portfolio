import glob
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat
import argparse 
from tqdm import tqdm
import time
from utils import re_sampling, find_minimum_cycle
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Preprocessing the dataset")

    parser.add_argument("--input", type=str, required=True, help="Input CSV folder path")
    parser.add_argument("--output", type=str, required=True, help="Save path")

    args = parser.parse_args()

    battery_list=glob.glob(args.input+'/*.csv')
    print('start preprocessing')
    prep_df=[pd.read_csv(file) for file in battery_list]
    min_row=find_minimum_cycle(prep_df)
    print(min_row)
    for df,file_path in tqdm(zip(prep_df,battery_list)):
        time.sleep(0.05)
        file_name=os.path.basename(file_path)
        save_path=os.path.join(args.output,file_name)
        clean_df=re_sampling(df,min_row)
        clean_df.to_csv(save_path,index=False)
    print('preprocess done!')

if __name__=="__main__":
    main()