import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import (LightningDataModule, LightningModule)
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args
from matplotlib import pyplot as plt
import gc
os.environ["POLARS_MAX_THREADS"] = "4"
import polars as pol
from config import parse_args

args = parse_args()

class PurgedGroupTimeSeriesSplit(_BaseKFold):
    
    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose
        
    def split(self, X, y=None, groups=None):
        
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            #train_array = []
            #test_array = []
            train_indices = set()  # 使用集合代替列表
            test_indices = set()   # 使用集合代替列表
            
            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                #train_array_tmp = group_dict[train_group_idx]
                #train_array = np.sort(np.unique(np.concatenate((train_array,train_array_tmp)),axis=None), axis=None)
            #train_end = train_array.size
                train_indices.update(group_dict[train_group_idx])  # 使用 update 添加数据并自动去重
 
            
            
            for test_group_idx in unique_groups[group_test_start:group_test_start +group_test_size]:
                #test_array_tmp = group_dict[test_group_idx]
                #test_array = np.sort(np.unique(np.concatenate((test_array,test_array_tmp)),axis=None), axis=None)
            #test_array  = test_array[group_gap:]
                test_indices.update(group_dict[test_group_idx])  # 使用 update 添加数据并自动去重
            
            #Optionally remove the group_gap from the test indices
            #if self.group_gap:
                #test_indices.difference_update(set(range(max(train_indices) + 1, max(train_indices) + 1 + self.group_gap)))
            
            if self.verbose > 0:
                    pass
                    
            #yield [int(i) for i in train_array], [int(i) for i in test_array]
            yield sorted(train_indices), sorted(test_indices) 


class CustomDataset(Dataset):
    def __init__(self, df,   accelerator):
        self.features = torch.FloatTensor(df[args.feature_names].values).to(accelerator)
        self.labels = torch.FloatTensor(df[args.label_name].values).to(accelerator)
        self.weights = torch.FloatTensor(df[args.weight_name].values).to(accelerator)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        w = self.weights[idx]
        return x, y, w


class DataModule(LightningDataModule):
    def __init__(self, train_path, valid_path, batch_size, valid_df=None, time_col = "date_id", accelerator='cpu', TEST = False):
        super().__init__()
        self.train_path = train_path
        self.valid_path = valid_path
        self.dates = pol.scan_parquet(train_path).select(time_col).unique().collect().to_numpy().squeeze(-1).tolist()
        self.batch_size = batch_size
        self.accelerator = accelerator
        self.train_dataset = None
        self.val_dataset = None
        self.TEST = TEST
        

    def setup(self,  train_selected_dates = None, valid_selected_dates = None, time_col = "date_id"):
        
        train = pol.scan_parquet(self.train_path).fill_null(strategy="forward").fill_null(0)
        valid = pol.scan_parquet(self.valid_path).fill_null(strategy="forward").fill_null(0)
        #train = pol.scan_parquet(f"{input_path}/training_data.parquet")
        #valid = pol.scan_parquet(f"{input_path}/validation_data.parquet")

        if train_selected_dates:
            train = train.filter(pol.col(time_col).is_in(train_selected_dates))
        if valid_selected_dates:
            valid = valid.filter(pol.col(time_col).is_in(valid_selected_dates))

        if self.TEST:
            train = train.head(1000000)
            valid = valid.head(100000)

        
        train = train.collect().to_pandas()
        valid = valid.collect().to_pandas()

        
        print(train.shape, valid.shape )

        self.train_dataset = CustomDataset(train, self.accelerator)
    
        self.val_dataset = CustomDataset(valid, self.accelerator)
        
        gc.collect()


    def get_purged_cv_and_plot(self, N_fold, test_train_ratio, group_gap , time_col, save_figure=True):
        """
        根据给定的时间列，生成PurgedGroupTimeSeriesSplit的CV划分并绘制图表。
        
        Args:
         N_fold, test_train_ratio, group_gap ,
            time_col: 时间列的名称。
            save_figure: 是否保存图表。
        """
        # 根据测试模式选择数据集
        if self.TEST:
            train = (
                pol.scan_parquet(self.train_path)
                .fill_null(strategy="forward")
                .fill_null(0)
                .filter(pol.col(time_col) >= 1500)
                .collect()
            )
        else:
            train = (
                pol.scan_parquet(self.train_path)
                .fill_null(strategy="forward")
                .fill_null(0)
                .collect()
            )
        
        # 数据处理
        train = train.sort(time_col).to_pandas()
        len_timeids = len(train[time_col].unique())
        print(f"总共天数: {len_timeids}")
        
        # 计算划分参数
        max_test_group_size = int(len_timeids / (N_fold + test_train_ratio))
        max_train_group_size = max_test_group_size * test_train_ratio
        print(f"max_test_group_size: {max_test_group_size}")
        print(f"max_train_group_size: {max_train_group_size}")
        print(f"group_gap: {group_gap}")
        
        # 创建CV划分对象
        cv = PurgedGroupTimeSeriesSplit(
            n_splits=N_fold,
            max_train_group_size=max_train_group_size,
            group_gap=group_gap,
            max_test_group_size=max_test_group_size,
        )
        
        # 生成和可视化CV划分
        splits = cv.split(train, groups=train[time_col].to_numpy())
        plt.figure(figsize=(24, 8))
        dates_each_fold = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            print(f"Processing fold {i}")
            
            # 获取训练和测试时间范围
            train_start_date = train[time_col][train_idx[0]]
            train_end_date = train[time_col][train_idx[-1]]
            test_start_date = train[time_col][test_idx[0]]
            test_end_date = train[time_col][test_idx[-1]]
            
            # 记录日期范围
            train_dates = range(train_start_date, train_end_date + 1)
            test_dates = range(test_start_date, test_end_date + 1)
            dates_each_fold.append((train_dates, test_dates))
            
            # 打印范围信息
            print(f"fold {i} train: {train_start_date} - {train_end_date}")
            print(f"fold {i} test: {test_start_date} - {test_end_date}")
            
            # 绘制训练数据
            plt.plot([train_start_date, train_end_date], [i, i], marker='o', color='blue')
            plt.text(train_start_date, i + 0.1, f"{train_start_date}", ha='right')
            plt.text(train_end_date, i + 0.1, f"{train_end_date}", ha='right')
            
            # 绘制测试数据
            plt.plot([test_start_date, test_end_date], [i, i], marker='o', color='red')
            plt.text(test_start_date, i - 0.2, f"{test_start_date}", ha='right')
            plt.text(test_end_date, i - 0.2, f"{test_end_date}", ha='right')
        
        # 清理资源
        del train
        gc.collect()
        
        # 图表设置与保存
        plt.title('Purged_Group_TimeSeries_Split')
        plt.xlabel('Time_Col')
        plt.ylabel('CV_Iteration')
        if save_figure:
            plt.savefig('./cv_plan.png')

        return dates_each_fold

    def setup_purged_cv(self, train_selected_dates, valid_selected_dates, time_col):
        
        df = pol.scan_parquet(self.train_path).fill_null(strategy="forward").fill_null(0)
        
        train = df.filter(pol.col(time_col).is_in(train_selected_dates))
        valid = df.filter(pol.col(time_col).is_in(valid_selected_dates))
            
        train = train.collect().to_pandas()
        valid = valid.collect().to_pandas()
        
        print(train.shape, valid.shape)

        self.train_dataset = CustomDataset(train, self.accelerator)

        self.val_dataset = CustomDataset(valid, self.accelerator)

        
        gc.collect()


    def train_dataloader(self, n_workers=0):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=n_workers)#, pin_memory=True, persistent_workers=True)#

    def val_dataloader(self, n_workers=0):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=n_workers)#, persistent_workers=True)#, pin_memory=True)
