import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from jinja2.optimizer import optimize
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp

# GAT model
class GATNet(torch.nn.Module):
    def __init__(
        self,
        num_features_xd=78,
        n_output=2,
        num_features_xt=954,
        output_dim=128,
        dropout=0.2,
        file=None,
    ):
        super(GATNet, self).__init__()

        # graph drug layers
        self.drug_gat1 = GATConv(num_features_xd, output_dim, heads=10, dropout=dropout)
        self.drug_gat2 = GATConv(output_dim * 10, output_dim, dropout=dropout)
        self.drug_fc_g1 = nn.Linear(output_dim, output_dim)
        self.filename = file

        # DL cell featrues
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim * 2),
            nn.ReLU(),
        )

        # combined layers
        self.fc1 = nn.Linear(output_dim * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim

    def forward(self, drug_1, drug_2, cell):
        # Extract data from PyG Data objects
        index_1 = drug_1.edge_index
        batch_1 = drug_1.batch
        
        index_2 = drug_2.edge_index
        batch_2 = drug_2.batch
        
        drug_1 = drug_1.x
        drug_2 = drug_2.x



        # deal drug1
        drug_1 = self.drug_gat1(drug_1, index_1)
        drug_1 = F.elu(drug_1)
        drug_1 = F.dropout(drug_1, p=0.2, training=self.training)
        drug_1 = self.drug_gat2(drug_1, index_1)
        drug_1 = F.elu(drug_1)
        drug_1 = F.dropout(drug_1, p=0.2, training=self.training)
        drug_1 = gmp(drug_1, batch_1)  # global max pooling
        drug_1 = self.drug_fc_g1(drug_1)
        drug_1 = self.relu(drug_1)

        # deal drug2
        drug_2 = self.drug_gat1(drug_2, index_2)
        drug_2 = F.elu(drug_2)
        drug_2 = F.dropout(drug_2, p=0.2, training=self.training)
        drug_2 = self.drug_gat2(drug_2, index_2)
        drug_2 = F.elu(drug_2)
        drug_2 = F.dropout(drug_2, p=0.2, training=self.training)
        drug_2 = gmp(drug_2, batch_2)  # global max pooling
        drug_2 = self.drug_fc_g1(drug_2)
        drug_2 = self.relu(drug_2)

        # deal cell
        cell = F.normalize(cell, 2, 1)
        cell_vector = self.reduction(cell)

        # concat
        xc = torch.cat((drug_1, drug_2, cell_vector), 1)
        xc = F.normalize(xc, 2, 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc3(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)

        return out
def acc(y_hat, y):
    return ((y_hat>0.5)==y).float().mean()

class DeepDDS(pl.LightningModule):
    def __init__(self):
        super(DeepDDS, self).__init__()
        self.model = GATNet()
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        d_1 = batch['drug1']
        d_2 = batch['drug2']
        cell = batch['cell']
        y = batch['label']
        y_hat = self.model(d_1, d_2, cell)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc(y_hat, y))
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        d_1 = batch['drug1']
        d_2 = batch['drug2']
        cell = batch['cell']
        y = batch['label']
        y_hat = self.model(d_1, d_2, cell)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, batch_size=y.shape[0])
        model_acc = acc(y_hat, y)
        self.log('val_acc', model_acc, batch_size=y.shape[0])
        return {'val_loss': loss, 'val_acc': model_acc}

    def test_step(self, batch, batch_idx):
        d_1 = batch['drug1']
        d_2 = batch['drug2']
        cell = batch['cell']
        y = batch['label']
        y_hat = self.model(d_1, d_2, cell)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc(y_hat, y))
        return {'test_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        return optimizer
