from dgl import DGLGraph
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import MaxPooling, GlobalAttentionPooling
from torch.nn import Linear, ELU, Sequential, ReLU, Dropout


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim):
        super(GATLayer, self).__init__()
        # (1) 中的 g(x) 函数
        self.gfc = nn.Sequential(
            nn.Linear(edge_dim, out_dim, bias=True),
            nn.Dropout(0.2),
        )
        # (5) 中的 f(x) 函数
        self.ffc = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.Dropout(0.2),
        )
        # (6) 中的 a(x) 函数
        self.attn_fc = nn.Sequential(
            nn.Linear(2 * out_dim, 1, bias=True),
            nn.Dropout(0.2),
            nn.ELU()
        )

    def message_func(self, edges):
        dst = edges.data['k'] * edges.dst["z"]
        src = edges.data['k'] * edges.src["z"]
        # 消息传递函数
        return {"z": src, "e": self.attn_fc(torch.cat([dst, src], dim=1))}

    def reduce_func(self, nodes):
        # 消息聚合函数
        # 公式 (7) 原始注意力的正则化
        # nodes.mailbox["e"]是一个用来存储每个节点收到的来自邻居节点的注意力分数的容器
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        # 公式 (8) 节点特征更新
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, g_0: DGLGraph):
        g = g_0.clone()
        h = g.ndata['h']
        k = g.edata['k']
        # 公式 (1) 边特征增维
        k = self.gfc(k)
        # 公式 (5) 节点特征变换
        z = self.ffc(h)
        # 将计算出的新特征z存储到图self.g的节点特征字典中
        g.edata["k"] = k
        g.ndata["z"] = z
        # 公式 (7) 和 (8)
        # 执行更新：遍历所有的节点和边，首先执行message_func进行消息传递，再执行reduce_func进行消息聚合
        g.update_all(self.message_func, self.reduce_func)
        # 删除临时生成的特征
        g.ndata.pop("z")
        return g


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, edge_dim, merge="cat"):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, edge_dim))
        self.merge = merge

    def forward(self, g: DGLGraph):
        head_outs = [attn_head(g) for attn_head in self.heads]
        h = [out.ndata['h'] for out in head_outs]
        k = [out.edata['k'] for out in head_outs]
        if self.merge == "cat":
            # 通过拼接合并多个头的结果
            g.ndata['h'] = torch.cat(h, dim=1)
            g.edata['k'] = torch.cat(k, dim=1)
            return g
        else:
            # 通过平均合并多个头的结果
            g.ndata['h'] = torch.mean(torch.tensor(h), dim=1)
            return g


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_dim, num_heads=1, dropout=0.2):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(
            in_dim=in_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            edge_dim=edge_dim,
            merge="cat"
        )

        self.dropout = nn.Dropout(p=dropout)
        self.layer2 = MultiHeadGATLayer(
            in_dim=hidden_dim*num_heads,
            out_dim=out_dim,
            num_heads=1,
            edge_dim=hidden_dim*num_heads,
            merge="cat"
        )
        self.shortcut = Sequential(
            Linear(hidden_dim*num_heads, out_dim),
            Dropout(p=dropout),
            ReLU()
        )
        self.attn_layer = Sequential(
            Linear(out_dim, 1),
            ELU()
        )

        self.max_pool = MaxPooling()
        self.attn_pool = GlobalAttentionPooling(self.attn_layer)

    def forward(self, g):
        g = self.layer1(g)
        ndata = g.ndata['h'] = F.normalize(F.elu(self.dropout(g.ndata['h'])))
        g.edata['k'] = F.normalize(F.elu(self.dropout(g.edata['k'])))
        g = self.layer2(g)
        g.ndata['h'] = self.dropout(self.shortcut(ndata) + F.elu(g.ndata['h']))
        F.normalize(g.ndata['h'])


        f = torch.cat([
            self.attn_pool(g, g.ndata['h']),
            self.max_pool(g, g.ndata['h']),
        ], dim=1)

        return f