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
from utils import drop_dupllicate, status_cycle, find_min_row_by_cycle, re_sampling, find_minimum_cycle
import warnings
warnings.filterwarnings("ignore")

def mat_to_csv(file_path):
    data = loadmat(file_path)
    columns=list(data['data'][0].dtype.fields.keys())
    columns.pop()

    battery=pd.DataFrame({col:data['data'][0][0][idx].flatten() for idx,col in enumerate(columns)})
    battery['cycle']=0

    for c in range(1,len(data['data'][0])):
        battery_C=pd.DataFrame({col:data['data'][0][c][idx].flatten() for idx,col in enumerate(columns)})
        battery_C['cycle']=c
        battery=pd.concat([battery,battery_C])
    
    return battery


def preprocessing(df,save_path):
    
    # 충전, 방전 flag 추가
    df['status']=np.where(df["current_A"] >= 0, "Charge", "Discharge")
    df["sign"] = df["status"].map({"Charge": 1, "Discharge": -1})
    df=df[(df['current_A']!=0)&(df['capacity_Ah']!=0)]

    # 1) 사이클별 cum_capa (현재 + 직전 값의 합: rolling(2))
    df["cum_capa"] = (
        (df["capacity_Ah"] * df["sign"])
        .groupby(df["cycle"])
        .rolling(2).sum()
        .reset_index(level=0, drop=True)
    )

    # 2) 사이클별 최대값
    max_cum = df.groupby("cycle")["cum_capa"].transform("max")

    # 3) SOC 계산 (Charge: x/max*100, Discharge: (1 + x/max)*100)
    ratio = np.divide(df["cum_capa"], max_cum, out=np.zeros_like(df["cum_capa"]), where=max_cum.ne(0))
    df["SOC"] = np.where(
        df["status"].eq("Charge"),
        ratio * 100.0,
        (1.0 + ratio) * 100.0
    )

    df["SOC"] = df["SOC"].clip(upper=100).fillna(0)
    df.drop(['sign','cum_capa'],axis=1,inplace=True)

    try:
        df.to_csv(save_path,index=False)
        print(f'saved : {save_path}')
    except:
        print(f'save fail')

    return df
   

def main():
    parser = argparse.ArgumentParser(description="Convert mat file to CSV file")

    parser.add_argument("--input", type=str, required=True, help="Input mat folder path")
    parser.add_argument("--output", type=str, required=True, help="Save path")

    args = parser.parse_args()

    battery_list=glob.glob(args.input+'/*.mat')
    print('Start convert')
    for file_path in tqdm(battery_list):
        save_path=os.path.join(args.output,os.path.basename(file_path).replace('mat','csv'))
        time.sleep(0.05)
        df=mat_to_csv(file_path) 
        preprocessing(df,save_path)
    
    print('Convert mat file to csv done!')

if __name__=="__main__":
    main()