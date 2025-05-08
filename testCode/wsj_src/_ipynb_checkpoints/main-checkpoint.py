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


# 解析参数
args = parse_args()

# 检查设备
device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() and args.usegpu else 'cpu')
accelerator = 'gpu' if torch.cuda.is_available() and args.usegpu else 'cpu'
loader_device = 'cpu'

if args.FULLTRAIN:
    data_module = DataModule([args.train_path , args.valid_path ], args.valid_path, batch_size=args.bs, accelerator=loader_device, TEST = args.TEST)
else:
    data_module = DataModule(args.train_path, args.valid_path, batch_size=args.bs, accelerator=loader_device, TEST = args.TEST)

dates_each_fold = data_module.get_purged_cv_and_plot(args.N_fold, args.test_train_ratio, args.group_gap , args.time_col )

# 设置全局随机种子
pl.seed_everything(args.seed)

saved_epochs = {}

for fold, (train_dates, valid_dates) in enumerate(dates_each_fold):
    print(f"setup date for fold {fold}")
    print(f"fold {fold} train: {train_dates[0]} - {train_dates[-1]}")
    print(f"fold {fold} test: {valid_dates[0]} - {valid_dates[-1]}")
    data_module.setup_purged_cv(train_dates, valid_dates, args.time_col)
    
    print(f"success setup fold {fold}")

    # 获取输入维度
    input_dim = data_module.train_dataset.features.shape[1]

    # 初始化模型
    model = AE_MLP(
        num_columns=input_dim,
        num_labels=1,
        hidden_units=args.n_hidden,
        dropout_rates=args.dropouts,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 初始化日志记录器
    if args.use_wandb:
        wandb_run = wandb.init(project=args.project, config=vars(args), reinit=True)
        logger = WandbLogger(experiment=wandb_run)
    elif args.use_tb:
        logger = TensorBoardLogger(args.tbroot, name=f"{args.project}_fold_{fold}")
    else:
        logger = None

    # 初始化回调函数
    early_stopping = EarlyStopping('val_r_square', patience=args.patience, mode='max', verbose=False)
    checkpoint_callback = ModelCheckpoint(monitor='val_r_square', mode='max', save_top_k=1, verbose=True, filename=os.path.join(args.save_model_root, f"nn_{fold}.model"))
    timer = Timer()

    # 初始化Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=[args.gpuid] if args.usegpu else None,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback, timer],
        enable_progress_bar=True,
    )

    # 开始训练
    trainer.fit(model, data_module.train_dataloader(args.loader_workers), data_module.val_dataloader(args.loader_workers))
    
    saved_epochs[fold] = model.train_epoch_record
    print(f"Fold-{fold} Training completed in {timer.time_elapsed('train'):.2f}s")

# 保存 saved_epochs 到 JSON 文件
with open(os.path.join(args.save_model_root, f"saved_epochs.json") , "w") as f:
    json.dump(saved_epochs, f, indent=4)
print("Saved epochs information to 'saved_epochs.json'")