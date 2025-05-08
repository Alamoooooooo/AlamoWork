import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from config import parse_args

# Custom R2 metric for validation
def r2_val(y_true, y_pred, sample_weight):
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return r2

class GaussianNoiseLayer(nn.Module):
    def __init__(self, stddev = 0.01):
        super(GaussianNoiseLayer, self).__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:  # 仅在训练时添加噪声
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        else:
            return x

class AE_MLP(pl.LightningModule):
    def __init__(
        self, 
        num_columns = 79, 
        num_labels = 1, 
        hidden_units = [96, 96, 896, 448, 448, 256], 
        dropout_rates = [0.03, 0.03, 0.4, 0.1, 0.4, 0.3, 0.2, 0.4], 
        lr=1e-3, 
        weight_decay = 5e-4
    ):
        
        super(AE_MLP, self).__init__()
        
        # Encoder part
        self.batch_norm_inp = nn.BatchNorm1d(num_columns)
        
        self.encoder = nn.Sequential(
            # 添加高斯噪声
            GaussianNoiseLayer(stddev=dropout_rates[0]),
            nn.Linear(num_columns, hidden_units[0]),
            nn.BatchNorm1d(hidden_units[0]),
            nn.SiLU()  # Swish activation
        )
        
        # Decoder part
        self.decoder = nn.Sequential(
            nn.Dropout(dropout_rates[1]),
            nn.Linear(hidden_units[0], num_columns)
        )

        # AE part
        self.x_ae = nn.Sequential(
            nn.Linear(num_columns, hidden_units[1]),
            nn.BatchNorm1d(hidden_units[1]),
            nn.SiLU(),
            nn.Dropout(dropout_rates[2])
        )

        self.out_ae = nn.Sequential(
            nn.Linear(hidden_units[1], num_labels),
            nn.Tanh()
            #nn.Tanh()
        )
                
        # Main part
        mlp_hidden_units = hidden_units[2:]
        mlp_dropouts = dropout_rates[3:]
        layers = []
        in_dim = num_columns + hidden_units[0]
        for i , hidden_dim in enumerate(mlp_hidden_units):
            layers.append(nn.BatchNorm1d(in_dim))
            if i > 0:
                layers.append(nn.SiLU())
            if i < len(dropout_rates):
                layers.append(nn.Dropout(dropout_rates[i]))
            layers.append(nn.Linear(in_dim, hidden_dim))
            # layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1)) 
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
        self.lr = lr
        self.weight_decay = weight_decay
        self.validation_step_outputs = []
        self.train_epoch_record = []
        

    def forward(self, x):
        # Encoder part
        x0 = self.batch_norm_inp(x)
        enc = self.encoder(x0)
        
        # Decoder part
        dec = self.decoder(enc)
        
        # AE part
        out_ae = self.x_ae(dec)
        out_ae = self.out_ae(out_ae).squeeze(-1)
        
        # Main part
        x_combined = torch.cat([x0, enc], dim=1)
        out = self.model(x_combined).squeeze(-1)  
        
        return 5*out_ae, 5*out

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        out_ae, out = self(x)
        
        # Using MSE loss for both autoencoder and MLP
        loss_ae  = F.mse_loss(out_ae, y, reduction='none') * w
        loss_mlp  = F.mse_loss(out, y, reduction='none') * w
        loss_ae  = loss_ae.mean()
        loss_mlp = loss_mlp.mean()
        loss = loss_ae + loss_mlp
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        out_ae, out = self(x)
        
        # Using MSE loss for both autoencoder and MLP
        loss_ae  = F.mse_loss(out_ae, y, reduction='none') * w
        loss_mlp  = F.mse_loss(out, y, reduction='none') * w
        loss_ae  = loss_ae.mean()
        loss_mlp = loss_mlp.mean()
        loss = loss_ae + loss_mlp
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.validation_step_outputs.append((out, y, w))        
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }
        
    def on_validation_epoch_end(self):
        """Calculate validation WRMSE at the end of the epoch."""
        y = torch.cat([x[1] for x in self.validation_step_outputs]).cpu().numpy()
        if self.trainer.sanity_checking:
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
        else:
            prob = torch.cat([x[0] for x in self.validation_step_outputs]).cpu().numpy()
            weights = torch.cat([x[2] for x in self.validation_step_outputs]).cpu().numpy()
            # r2_val
            val_r_square = r2_val(y, prob, weights)
            self.log("val_r_square", val_r_square, prog_bar=True, on_step=False, on_epoch=True)
        self.validation_step_outputs.clear()
    
    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        epoch = self.trainer.current_epoch
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in self.trainer.logged_metrics.items()}
        formatted_metrics = {k: f"{v:.5f}" for k, v in metrics.items()}
        print(f"Epoch {epoch}: {formatted_metrics}")
        self.train_epoch_record.append({"epoch":epoch, "val_r_square": float(self.trainer.logged_metrics["val_r_square"]), "val_loss": float(self.trainer.logged_metrics["val_loss"]) })
