import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data


class ModelDataset(Dataset):
    def __init__(self, data):
        self.graph1_list = []
        self.graph2_list = []
        self.cell_list = []
        self.labels = []
        for d in data:
            self.graph1_list.append(d['d_1'])
            self.graph2_list.append(d['d_2'])
            self.cell_list.append(d['cell'])
            self.labels.append(d['label'])


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        graph1 = self.graph1_list[idx]
        graph2 = self.graph2_list[idx]
        vector = self.cell_list[idx]
        label = self.labels[idx]
        return graph1, graph2, vector, label


class PyGDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            drug1_graphs: List[Tuple] - 预处理的drug1图数据列表，每个元素为(c_size, features, edge_index)
            drug2_graphs: List[Tuple] - 预处理的drug2图数据列表，每个元素为(c_size, features, edge_index)
            cell_features: List[Tensor] - 预处理的细胞特征列表
            labels: List[float] - 标签列表
        """
        self.drug1_graphs = []
        self.drug2_graphs = []
        self.cell_features = []
        self.labels = []
        for d in data:
            self.drug1_graphs.append(d['d_1'])
            self.drug2_graphs.append(d['d_2'])
            self.cell_features.append(d['cell'])
            self.labels.append(d['label'])


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 直接使用预处理好的drug1图数据
        d1_size, d1_feat, d1_edge = self.drug1_graphs[idx]
        # Convert features to numpy arrays first
        d1_feat = np.array(d1_feat)
        
        drug1_data = Data(
            x=torch.FloatTensor(d1_feat),
            edge_index=torch.LongTensor(d1_edge).t().contiguous(),
            c_size=torch.LongTensor([d1_size])
        )

        # 直接使用预处理好的drug2图数据
        d2_size, d2_feat, d2_edge = self.drug2_graphs[idx]
        d2_feat = np.array(d2_feat)
        drug2_data = Data(
            x=torch.FloatTensor(d2_feat),
            edge_index=torch.LongTensor(d2_edge).t().contiguous(),
            c_size=torch.LongTensor([d2_size])
        )

        # 直接使用预处理好的细胞特征
        cell_tensor = self.cell_features[idx]
        label = self.labels[idx]

        # 返回样本数据
        return {
            'drug1': drug1_data,
            'drug2': drug2_data,
            'cell': cell_tensor,
            'label': label
        }
