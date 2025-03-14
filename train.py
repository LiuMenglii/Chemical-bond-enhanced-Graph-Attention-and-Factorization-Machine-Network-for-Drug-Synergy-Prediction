import os
import pickle

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import DataLoader as GDataLoader

import wandb
from dataset import ModelDataset, PyGDataset
from models.DeepDDS import DeepDDS
from models.Model import PLModel
from utils import collate_fn, collate_pyg


def train_dgl():
    torch.set_float32_matmul_precision('high')
    with open("./encodings/dgl/train.pkl", "rb") as f:  # 使用二进制读取模式 'rb'
        train_dataset = pickle.load(f)

    with open("./encodings/dgl/cv.pkl", "rb") as f:
        cv_dataset = pickle.load(f)

    train_dataset = ModelDataset(train_dataset)
    cv_dataset = ModelDataset(cv_dataset)

    train_dataset = DataLoader(train_dataset, batch_size=192, shuffle=True, collate_fn=collate_fn, num_workers=os.cpu_count())
    cv_dataset = DataLoader(cv_dataset, batch_size=512, shuffle=False, collate_fn=collate_fn, num_workers=os.cpu_count())

    model = PLModel()

    # 初始化早停和模型保存回调
    # early_stop = EarlyStopping(
    #     monitor="val_loss",
    #     patience=100,
    #     mode="min",
    # )

    checkpoint_loss = ModelCheckpoint(
        dirpath="./checkpoints/dgl",
        filename="loss-{epoch:02d}-{val_loss:.3f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
    )

    checkpoint_acc = ModelCheckpoint(
        dirpath="./checkpoints/dgl",
        filename="acc-{epoch:02d}-{val_acc:.3f}",
        monitor="val_acc",
        save_top_k=3,
        mode="max",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    wandb.init(project='drug combination')
    wandb_logger = WandbLogger()

    trainer = Trainer(
        logger=wandb_logger,
        accelerator="gpu",
        callbacks=[checkpoint_acc, checkpoint_loss, lr_monitor],
        log_every_n_steps=1,
        max_epochs=500,
    )
    trainer.fit(model, train_dataset, cv_dataset)

    # 加载测试集
    with open("./encodings/dgl/test.pkl", "rb") as f:
        test_dataset = pickle.load(f)

    test_dataset = ModelDataset(test_dataset)
    test_dataset = DataLoader(
        test_dataset,
        batch_size=512,
        collate_fn=collate_fn,
        shuffle=False
    )

    best_acc_model = PLModel.load_from_checkpoint(
        checkpoint_acc.best_model_path
    )
    best_loss_model = PLModel.load_from_checkpoint(
        checkpoint_loss.best_model_path
    )

    # 分别测试
    trainer.test(best_loss_model, dataloaders=test_dataset)
    trainer.test(best_acc_model, dataloaders=test_dataset)


def train_pyg():
    torch.set_float32_matmul_precision('high')
    with open("./encodings/pyg/train.pkl", "rb") as f:
        train_dataset = pickle.load(f)

    with open("./encodings/pyg/cv.pkl", "rb") as f:
        cv_dataset = pickle.load(f)

    train_dataset = PyGDataset(train_dataset)
    cv_dataset = PyGDataset(cv_dataset)

    train_dataset = GDataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=collate_pyg,
        num_workers=os.cpu_count()//2,
    )
    cv_dataset = GDataLoader(
        cv_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=os.cpu_count()//2,
        collate_fn=collate_pyg
    )

    model = DeepDDS()

    checkpoint_loss = ModelCheckpoint(
        dirpath="./checkpoints/pyg",
        filename="loss-{epoch:02d}-{val_loss:.3f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
    )

    checkpoint_acc = ModelCheckpoint(
        dirpath="./checkpoints/pyg",
        filename="acc-{epoch:02d}-{val_acc:.3f}",
        monitor="val_acc",
        save_top_k=3,
        mode="max",
    )

    wandb.init(project='drug combination')
    wandb_logger = WandbLogger()

    trainer = Trainer(
        logger=wandb_logger,
        accelerator="gpu",
        log_every_n_steps=1,
        callbacks=[checkpoint_loss, checkpoint_acc],
        max_epochs=100
    )

    # 训练模型（包含验证集）
    trainer.fit(model, train_dataset, cv_dataset)

    # 加载测试集
    with open("./encodings/pyg/test.pkl", "rb") as f:
        test_dataset = pickle.load(f)

    test_dataset = PyGDataset(test_dataset)
    test_dataset = GDataLoader(
        test_dataset,
        batch_size=512,
        collate_fn=collate_pyg,
        shuffle=False
    )

    best_acc_model = DeepDDS.load_from_checkpoint(
        checkpoint_acc.best_model_path
    )
    best_loss_model = DeepDDS.load_from_checkpoint(
        checkpoint_loss.best_model_path
    )

    # 分别测试
    trainer.test(best_loss_model, dataloaders=test_dataset)
    trainer.test(best_acc_model, dataloaders=test_dataset)

