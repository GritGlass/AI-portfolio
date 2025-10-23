import os
import glob
import pandas as pd
import numpy as np
import argparse
import io
from typing import Tuple, Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import lightning as L
from torchmetrics.functional import accuracy
from torchmetrics.regression import MeanSquaredError, R2Score
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from utils import df_to_dataloader, train_test_split
from utils import dataset
from lightning.pytorch.loggers import CSVLogger
import sys
import json
import warnings
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from model.DNN import DNN
warnings.filterwarnings("ignore")

def train(model,train_loader,test_loader,save_path,try_id,epochs=1000,early_stop=30,wandblogger=False,resume=None,):

    save_root=os.path.join(save_path,f'{try_id}')
    os.makedirs(save_root,exist_ok=True)

    early_stop_cb = EarlyStopping(
        monitor="valid_loss",   
        min_delta=0.0,       
        patience=early_stop,         
        mode="min",          
        verbose=True,
        check_on_train_epoch_end=False, 
    )

    ck_filenm=f'{try_id}_DNN_TL_BASE'
    ckpt_cb = ModelCheckpoint(
        dirpath=save_root, 
        filename=ck_filenm+"_ep{epoch:02d}-val{val_loss:.4f}", 
        monitor="valid_loss",
        mode="min",
        save_top_k=1,         
        save_last=True,      
        auto_insert_metric_name=False,   
        every_n_epochs=1
    )

    if wandblogger:
        wandb_logger = WandbLogger(log_model="all",project='Battery',name=f'DNN_TL_{try_id}',save_dir=save_root)
        trainer = Trainer(
            max_epochs=epochs,
            accelerator="auto",   
            devices="auto",
            precision="16-mixed",       
            callbacks=[early_stop_cb, ckpt_cb],
            log_every_n_steps=10,
            enable_checkpointing=True,
            logger=wandb_logger)
    else:
        csv_logger = CSVLogger(
            save_dir=save_root        
        )

        trainer = Trainer(
            max_epochs=epochs,
            accelerator="auto",   
            devices="auto",
            precision="16-mixed",       
            callbacks=[early_stop_cb, ckpt_cb],
            log_every_n_steps=10,
            enable_checkpointing=True,
            logger=csv_logger)

    if resume:
        trainer.fit(model, train_loader, test_loader,ckpt_path=resume)
    else:
        trainer.fit(model, train_loader, test_loader)

def finetuning(model,fine_tuning_loader, test_dataloader,try_id,cycle,save_path,epochs=1000,early_stop=30,wandblogger=False,resume=None):

    save_root=os.path.join(save_path,f'{try_id}')
    os.makedirs(save_root,exist_ok=True)

    #freeze
    for name, param in model.named_parameters():
        if "net.0" in name:  
            param.requires_grad = False


    early_stop_cb = EarlyStopping(
        monitor="valid_loss",   
        min_delta=0.0,        
        patience=early_stop,         
        mode="min",          
        verbose=True,
        check_on_train_epoch_end=False,  
    )

    ckpt_filenm=f'{try_id}_DNN_TL_AD_cy{cycle}_'
    ckpt_cb = ModelCheckpoint(
        dirpath=save_root,   
        filename=ckpt_filenm+"ep{epoch:02d}-val{val_loss:.4f}",  
        monitor="valid_loss",
        mode="min",
        save_top_k=1,        
        save_last=True,     
        auto_insert_metric_name=False,  
        every_n_epochs=1
    )

    if wandblogger:
        wandb_logger = WandbLogger(log_model="all",project='Battery',name='DNN_TL_AD',save_dir=save_root)
        trainer = Trainer(
            max_epochs=epochs,
            logger=wandb_logger,
            callbacks=[early_stop_cb, ckpt_cb],
        )
    else:
        csv_logger = CSVLogger(
            save_dir=save_root
        )
         
        trainer = Trainer(
            max_epochs=epochs,
            logger=csv_logger,
            callbacks=[early_stop_cb, ckpt_cb],
        )

    if resume:
        trainer.fit(model, fine_tuning_loader, test_dataloader,ckpt_path=resume)
    else:
        trainer.fit(model, fine_tuning_loader, test_dataloader)

def test(model, test_df, test_dataloader, save_path,try_id):

    save_root=os.path.join(save_path,f'{try_id}')
    os.makedirs(save_root,exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    print('Start inference')
    torch.set_grad_enabled(False)
    model.eval()
    all_preds = []
    MSE = []
    R2 = []
    for batch_idx, batch in enumerate(test_dataloader):
        batch = [b.to(device) for b in batch] # Move batch to the device
        metrics = model.test_step(batch)
        pred = model.predict_step(batch)
        all_preds+=list(pred.detach().cpu().flatten().numpy())
        MSE.append(metrics['test_mse'].item())
        R2.append(metrics['test_r2'].item())

    # Save mse, r2 result
    results={'MSE':np.mean(MSE), 'R2':np.mean(R2)}
    with open( os.path.join(save_root,f"{try_id}_test_results.json"), "w") as json_file:
        json.dump(results, json_file)

    print('Save error well')

    fig_path=os.path.join(save_root,f'figures')
    os.makedirs(fig_path, exist_ok=True)

    test_df['Pred_SOC']=all_preds


    for i in test_df.cycle.unique():
        plt.figure()
        sns.lineplot(data=test_df[test_df['cycle']==i],x='relative_time_min',y='Pred_SOC',color='blue',label='Pred')
        sns.lineplot(data=test_df[test_df['cycle']==i],x='relative_time_min',y='SOC',color='red',label='GT')
        plt.legend()
        plt.title(f'cycle : {i}')
        plt.savefig(f'{fig_path}/cycle_{i}.png',bbox_inches='tight')
        plt.close()

    print('Save figure well')
    
def main():
    parser = argparse.ArgumentParser(description="Run the code")

    parser.add_argument("--input", type=str, required=True, help="Data folder path for training")
    parser.add_argument("--test_id", type=int, required=True, help="Select test file")
    parser.add_argument("--epoch", type=int, required=False, help="Set train and finetuning epoch")
    parser.add_argument("--early_stop", type=int, required=False, help="Set early_stop")
    parser.add_argument("--test_cycle", type=int, required=True, help="Set the test cycle: use the data before this cycle for fine-tuning, and the data after this cycle for testing.")
    parser.add_argument("--run_type", type=str, required=True, help="Select operation type, one of {train, finetune, test}")
    parser.add_argument("--model_weights", type=str, required=False, help="Ckpt path for finetuning and test operation")
    parser.add_argument("--output", type=str, required=True, help="Root path for saving everything")
    parser.add_argument("--try_id", type=int, required=True, help="Try_id")
    parser.add_argument("--resume", type=str, required=False, help="Ckpt file path")

    args = parser.parse_args()

    if args.epoch==False:
        epoch=1000
    else:
        epoch=args.epoch  

    if args.early_stop==False:
        early_stop=1000
    else:
        early_stop=args.early_stop  


    running_log={
        'dataset_path': args.input,
        'testset_id'  : args.test_id,
        'test_cycle'  : args.test_cycle,
        'epoch'       : args.epoch,
        'early_stop'  : early_stop,
        'run_type'    : epoch,
        'model'       : args.model_weights,
        'output_path' : args.output,
        'try_id'      : args.try_id,
        'resume'      : args.resume
    }
    

    battery_list=glob.glob(args.input+'/*.csv')
    battery_df=[pd.read_csv(file) for file in battery_list]
    print(f'the number of files : {len(battery_df)}')

    train_df,test_df=train_test_split(battery_df,args.test_id)
    train_loader,fine_tuning_loader,test_loader, test_dataset=df_to_dataloader(train_df,test_df,cycle_id=args.test_cycle)

    model=DNN(4)

    if args.run_type=='train':
        save_path=os.path.join(args.output,'train')
        os.makedirs(save_path,exist_ok=True)
        #실험 코드
        with open( os.path.join(save_path,f"{args.try_id}_train_running_log.json"), "w") as json_file:
            json.dump(running_log, json_file)

        if args.resume:
            train(model,train_loader,test_loader,save_path,args.try_id,epochs=epoch,early_stop=early_stop,wandblogger=False,resume=args.resume)
        else:
            train(model,train_loader,test_loader,save_path,args.try_id,epochs=epoch,early_stop=early_stop,wandblogger=False,resume=None)

    elif args.run_type=='finetune':
        save_path=os.path.join(args.output,'finetune')
        os.makedirs(save_path,exist_ok=True)
        #실험 코드
        with open( os.path.join(save_path,f"{args.try_id}_finetune_running_log.json"), "w") as json_file:
            json.dump(running_log, json_file)


        ckpt = torch.load(args.model_weights)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)

        if args.resume:
            finetuning(model,fine_tuning_loader, test_loader,args.try_id,args.test_cycle,save_path,epochs=epoch,early_stop=early_stop, wandblogger=False,resume=args.resume)
        else:
            finetuning(model,fine_tuning_loader, test_loader,args.try_id,args.test_cycle,save_path,epochs=epoch,early_stop=early_stop, wandblogger=False,resume=None)
    
    elif args.run_type=='test':
        save_path=os.path.join(args.output,'test')
        os.makedirs(save_path,exist_ok=True)
        #실험 코드
        with open( os.path.join(save_path,f"{args.try_id}_test_running_log.json"), "w") as json_file:
            json.dump(running_log, json_file)

        ckpt = torch.load(args.model_weights)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)

        test(model, test_dataset, test_loader, save_path,args.try_id,)

if __name__=="__main__":
    main()