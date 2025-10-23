import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

def drop_dupllicate(battery_df):
    '''
    SOC를 기준으로 중복 된 row 제거
    '''
    df_clean = battery_df.groupby("cycle", group_keys=False).apply(
                                   lambda g: g.drop_duplicates(subset=["SOC"], keep="first"))
    return df_clean

def status_cycle(battery_df,stauts,max_cycle):
    '''
    status : Charge, Discharge
    max_cycle : cycle은 1이상 max_cycle 미만인 데이터만 사용
    '''
    df=battery_df[(battery_df['status']==stauts)&(battery_df['cycle']>0)&(battery_df['cycle']<=max_cycle)]
    return df

def find_min_row_by_cycle(charge_batteries):
    '''
    min_rows : dataset 별로 cycle별 최소 row 값 list
    min_num  : cycle별 최소 row 값, 모든 dataframe의 cycle별 row는 해당 값으로 resample 되어야 함
    '''
    min_rows=[]
    for dd in range(len(charge_batteries)):
        row_min=float('inf')
        for i in charge_batteries[dd].cycle.unique():
            if row_min>len(charge_batteries[dd][charge_batteries[dd]['cycle']==i]):
                row_min=len(charge_batteries[dd][charge_batteries[dd]['cycle']==i])
        min_rows.append(row_min)
    return min(min_rows)

def re_sampling(df,n_sample):
    '''
    cycle별 데이터 개수를 min_num에 맞춰서 등간격으로 resample 수행
    '''
    one_df_index=[]
    for cy in df.cycle.unique():
        cycle_df=df[df['cycle']==cy]

        n_total = len(cycle_df)
        idx = np.linspace(0, n_total - 1, n_sample).astype(int)
        iid=cycle_df.iloc[idx].index.tolist()
        one_df_index += iid

    re_df=df.loc[one_df_index]
    return re_df

def find_minimum_cycle(battery_df):
    '''
    Read CSV files and split 'train_df' into 'test_df'
    example 
        test_df_id : 7    
        train_df   : 0~6 battery files
        test_df    : 7 battery files
    '''
    clean_batteries=[drop_dupllicate(df) for df in battery_df]
    charge_batteries=[status_cycle(df,'Charge',300) for df in clean_batteries]
    min_row_cycle_=find_min_row_by_cycle(charge_batteries)
    return min_row_cycle_ 
    
    
def train_test_split(charge_batteries_sampled,test_df_id):
    test_df=charge_batteries_sampled.pop(test_df_id)
    train_df_list=charge_batteries_sampled
    train_df=pd.concat(train_df_list)
    return train_df,test_df
    

def df_to_dataloader(train_df,test_df,cycle_id=10):
    fine_tuning=test_df[test_df['cycle']<=cycle_id]
    test_df=test_df[test_df['cycle']>cycle_id]

    #train data
    train_x_df=train_df[['relative_time_min', 'voltage_V', 'current_A' ,'temperature_C']].values.astype(np.float32)
    train_y=train_df['SOC'].values.astype(np.float32)

    #fine-tune data
    fine_tuning_x_df=fine_tuning[['relative_time_min', 'voltage_V', 'current_A' ,'temperature_C']].values.astype(np.float32)
    fine_tuning_y=fine_tuning['SOC'].values.astype(np.float32)
    
    #test data
    test_x_df=test_df[['relative_time_min', 'voltage_V', 'current_A' ,'temperature_C']].values.astype(np.float32)
    test_y=test_df['SOC'].values.astype(np.float32)

    #scaler
    scaler = StandardScaler().fit(train_x_df)
    train_x = scaler.transform(train_x_df)
    fine_tuning_x = scaler.transform(fine_tuning_x_df)
    test_x = scaler.transform(test_x_df)


    train_dataset=dataset(train_x,train_y)
    train_loader = DataLoader(train_dataset, batch_size=561600, shuffle=False) # 7개 dataset, 각 datatset에는 37440개 row가 있음

    fine_tuning_dataset=dataset(fine_tuning_x,fine_tuning_y)
    fine_tuning_loader = DataLoader(fine_tuning_dataset, batch_size=1872, shuffle=False) # 1개 dataset, 각 dataset에는 524160개 row가 있음

    test_dataset=dataset(test_x,test_y)
    test_dataloader = DataLoader(test_dataset, batch_size=1872, shuffle=False) # 1개 dataset, 각 dataset에는 524160개 row가 있음

    return train_loader,fine_tuning_loader,test_dataloader, test_df

class dataset(Dataset):
    def __init__(self,x,y):
        self.x=torch.tensor(x, dtype=torch.float32)
        self.y=torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
