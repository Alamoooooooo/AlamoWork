import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import wandb
from torch.utils.data import DataLoader
import gc
from config import parse_args
from model import AE_MLP
from datafunc import DataModule
import json

args = parse_args()
# 在最后用最后一个fold最好的epoch训练一个日期最新的模型，模型训练集长度和之前一样
# 检查设备

# 从 JSON 文件中读取 saved_epochs
with open("saved_epochs.json", "r") as f:
    saved_epochs = json.load(f)
print("Loaded epochs information from 'saved_epochs.json'")
print(saved_epochs)

device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() and args.usegpu else 'cpu')
accelerator = 'gpu' if torch.cuda.is_available() and args.usegpu else 'cpu'
loader_device = 'cpu'

# 初始化数据模块
if args.FULLTRAIN:
    data_module = DataModule([args.train_path +args.valid_path], args.valid_path, batch_size=args.bs, accelerator=loader_device, TEST = args.TEST)
else:
    data_module = DataModule(args.train_path, args.valid_path, batch_size=args.bs, accelerator=loader_device, TEST = args.TEST)

# 手动选一个最佳的epoch 看最后一个fold的最佳epoch
last_fold_best_epoch = 2
train_end_dt = 1698
train_dt_length = 372

# 设置数据模块为最后一个fold的日期
data_module.setup(list(range(train_end_dt-train_dt_length, train_end_dt + 1)), None, args.time_col)

# 初始化模型
input_dim = data_module.train_dataset.features.shape[1]
print(input_dim, args.n_hidden)
model = AE_MLP(
    num_columns=input_dim,
    num_labels=1,
    hidden_units=args.n_hidden,
    dropout_rates=args.dropouts,
    lr=args.lr,
    weight_decay=args.weight_decay
)
print(model)

# 初始化日志记录器
if args.use_wandb:
    wandb_run = wandb.init(project=args.project, config=vars(args), reinit=True)
    logger = WandbLogger(experiment=wandb_run)
elif args.use_tb:
    logger = TensorBoardLogger(args.tbroot, name=f"{args.project}_newest")
else:
    logger = None

checkpoint_callback = ModelCheckpoint(
    dirpath=args.save_model_root,  # 保存路径
    filename="newest.model" ,  # 文件名
    save_last=True,  # 保存最后一个epoch的模型
    verbose=True,
    monitor="val_r_square",  # 可根据需要监控的指标来调整
    mode="max"   
)

# 初始化回调函数
timer = Timer()
# 初始化Trainer
trainer = Trainer(
    max_epochs=last_fold_best_epoch,
    accelerator=accelerator,
    devices=[args.gpuid] if args.usegpu else None,
    logger=logger,
    callbacks=[timer, checkpoint_callback],
    enable_progress_bar=True,
)

# 开始训练
trainer.fit(model, data_module.train_dataloader(args.loader_workers), data_module.val_dataloader(args.loader_workers))
print(f"Training completed in {timer.time_elapsed('train'):.2f}s")
print(f"Best model saved at epoch {last_fold_best_epoch} with score {trainer.callback_metrics['val_r_square']}")