import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from models.GAT import GAT

class Cell(nn.Module):
    def __init__(self, in_size, hidden_1, hidden_2, out_num):
        super(Cell, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_size, hidden_1),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_2, out_num)
        )

    def forward(self, input):
        return self.mlp(input)

class StackedFactorizationMachine(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(StackedFactorizationMachine, self).__init__()
        self.n = in_dim
        self.k = hidden_dim
        self.num_fms = out_dim
        self.linear = nn.Linear(self.n, self.num_fms, bias=True)
        self.dropout = nn.Dropout(0.2)
        self.W = nn.Parameter(torch.Tensor(self.num_fms, self.n, self.k))
        nn.init.xavier_uniform_(self.W)

        self.ReLU = nn.ReLU()

    def forward(self, x):
        linear_part = self.linear(x)  # 线性部分
        interactions = 0.5 * torch.sum(
            + torch.pow(torch.einsum('ij,fjk->ifk', x, self.W), 2)
            - torch.einsum('ij,fjk->ifk', torch.pow(x, 2), torch.pow(self.W, 2)),
            dim=2
        )
        linear_part = self.dropout(linear_part)
        interactions = self.dropout(interactions)
        return self.ReLU(linear_part + interactions)


class Model(nn.Module):

    def __init__(
            self,
            # 初始节点特征维度
            node_dim=36,
            # 初始边特征维度
            edge_dim=40,
            # GAT输出特征维度
            gat_out_dim=64,
            # Gate输出特征数量
            gate_out_num=5,
            # 提取细胞系特征的mlp输出维度
            mlp_out_dim=128,
            dropout=0.2,
            # GAT隐藏层维度
            gat_hidden_dim=32,
            fm_hidden_dim=32,
            num_heads=10,
    ):
        super(Model, self).__init__()

        # classifier_in = 2 * (gate_out_num+6) * gat_out_dim + mlp_out_dim

        classifier_in = 2 * 2 * gat_out_dim + mlp_out_dim

        self.gat = GAT(
            in_dim=node_dim,
            hidden_dim=gat_hidden_dim,
            num_heads=num_heads,
            out_dim=gat_out_dim,
            edge_dim=edge_dim,
            dropout=dropout
        )

        # self.gate = Gate(in_dim=gat_out_dim, out_num=gate_out_num)
        self.cell = Cell(in_size=954, hidden_1=1024, hidden_2=512, out_num=128)

        self.fm = StackedFactorizationMachine(in_dim=classifier_in, hidden_dim=fm_hidden_dim, out_dim=1024)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.readout_att = nn.Sequential(
            nn.Linear(in_features=gat_out_dim, out_features=1),
            nn.ELU()
        )

    def forward(self, d_1, d_2, cell_line):

        f_1 = self.gat(d_1)
        f_2 = self.gat(d_2)
        cell_line = self.cell(cell_line)

        out = self.dropout(torch.cat((f_1, f_2, cell_line), dim=1))
        out = F.normalize(out)
        out = self.fm(out)
        out = self.dropout(out)
        out = F.normalize(out)
        out = self.classifier(out)

        return out.view(-1)

def acc(y_hat, y):
    preds = (y_hat > 0.5).float()
    return (preds == y).float().mean()

class PLModel(pl.LightningModule):
    def __init__(self):
        super(PLModel, self).__init__()
        self.model = Model()
        # self.model.compile()
        self.criterion = nn.BCELoss()

    def training_step(self, batch, batch_idx):
        d_1, d_2, cell, y = batch
        y_hat = self.model(d_1, d_2, cell)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc(y_hat, y))
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        d_1, d_2, cell, y = batch
        y_hat = self.model(d_1, d_2, cell)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        model_acc = acc(y_hat, y)
        self.log('val_acc', model_acc)
        return {'val_loss': loss, 'val_acc': model_acc}

    def test_step(self, batch, batch_idx):
        d_1, d_2, cell, y = batch
        y_hat = self.model(d_1, d_2, cell)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc(y_hat, y))
        return {'test_loss': loss}

    def configure_optimizers(self):
        # 定义优化器
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, weight_decay=5e-5)
        # 定义调度器：每 10 个 epoch 学习率乘以 0.9
        scheduler = {
            "scheduler": StepLR(optimizer, step_size=5, gamma=0.99),
            "interval": "epoch",  # 按 epoch 调整（默认）
        }
        return [optimizer], [scheduler]
