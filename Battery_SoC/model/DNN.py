import argparse
import io
import os
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torchmetrics.regression import MeanSquaredError, R2Score
import lightning as L    
import warnings
warnings.filterwarnings("ignore")

class DNN(L.LightningModule):
    def __init__(self, in_dim, hidden=[64, 32]):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )
        self.save_hyperparameters()
        self.loss=nn.MSELoss()
        self.r2 = R2Score()
        self.mse = MeanSquaredError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs)
        loss = self.loss(output,target)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        inputs, target = batch
        output = self.model(inputs)
        loss = self.loss(output,target)
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _shared_eval_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        r2 = self.r2(y_hat, y)
        mse = self.mse(y_hat, y)
        return r2, mse

    def test_step(self, batch):
        r2, mse = self._shared_eval_step(batch)
        metrics = {"test_mse": mse, "test_r2": r2}
        self.log_dict(metrics)
        return metrics

    def predict_step(self, batch):
        inputs, target = batch
        return self.model(inputs)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=0.001)