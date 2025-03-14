import os
import pickle

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
from pandarallel import pandarallel
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch

# 定义元素和杂化类型
atom_types = ['C', 'O', 'S', 'N', 'P', 'F', 'Cl', 'B', 'Br', 'I', 'Pt']
hyb_types = ['SP', 'SP2', 'SP3', 'SP3D']
bond_types = [
    Chem.rdchem.BondType.SINGLE,  # 单键
    Chem.rdchem.BondType.DOUBLE,  # 双键
    Chem.rdchem.BondType.TRIPLE,  # 三键
    Chem.rdchem.BondType.AROMATIC,  # 芳香键
]

node_feature_dim = 36
edge_feature_dim = 40

def atom_feature(mol):
    hyb_type = str(mol.GetHybridization())
    atom_type = mol.GetSymbol()
    atom = np.zeros(12)
    hyb = np.zeros(5)
    atom[atom_types.index(atom_type) if atom_type in atom_types else 11] = 1
    hyb[hyb_types.index(hyb_type) if hyb_type in hyb_types else 4] = 1
    return np.concat([atom, hyb], axis=None)

def smile_to_dgl(smile: str) -> dgl.graph:
    """
        将SMILE字符串转换为DGL图结构
        :arg smile: 化合物的SMILE字符串
        :return g: 表示化合物结构的DGL图
    """
    # 使用RDKit解析SMILE字符串
    mol = Chem.MolFromSmiles(smile)

    # 创建DGL图
    g = dgl.graph(([], []))

    # 获取原子特征
    # 此处不能在括号中添加None
    for atom in mol.GetAtoms():
        # 提取原子特征
        atom_degree = atom.GetDegree()
        atom_valence = atom.GetImplicitValence()
        num_h = atom.GetTotalNumHs()
        atom_type = atom.GetSymbol()
        hyb_type = str(atom.GetHybridization())

        atom_num = (atom.GetAtomicNum()-5) / (78-5)
        atom_degree_code = np.zeros(8)
        atom_valence_code = np.zeros(5)
        atom_type_code = np.zeros(12)
        hyb_type_code = np.zeros(5)
        num_h_code = np.zeros(5)

        atom_degree_code[atom_degree] = 1
        atom_valence_code[atom_valence] = 1
        num_h_code[num_h] = 1

        # 生成元素类型的独热编码向量
        atom_type_code[atom_types.index(atom_type) if atom_type in atom_types else 11] = 1
        # 生成杂化类型的独热编码向量
        hyb_type_code[hyb_types.index(hyb_type) if hyb_type in hyb_types else 4] = 1

        node_feature = \
            {'h':
                torch.tensor(
                    np.concat([atom_num, atom_degree_code, atom_valence_code,
                               atom_type_code, hyb_type_code, num_h_code], axis=None),
                    dtype=torch.float32
                ).view(1, node_feature_dim)
            }
        # 添加节点
        g.add_nodes(1, data=node_feature)

        edge_feat = np.zeros(6)
        edge_feat[5] = 1
        edge_feat = torch.tensor(
            np.concat([edge_feat, atom_type_code, hyb_type_code, atom_type_code, hyb_type_code]),
            dtype=torch.float32
        ).view(1, edge_feature_dim)
        # 添加自环
        g.add_edges(atom.GetIdx(), atom.GetIdx(), data={'k': edge_feat})

    # 添加边
    # 此处不能在括号中添加None
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()

        # 创建独热编码的边特征向量
        edge_feat = np.zeros(6)

        if bond_type in bond_types:
            edge_feat[bond_types.index(bond_type)] = 1

        if bond.IsInRing():
            edge_feat[4] = 1

        u_feat = atom_feature(bond.GetBeginAtom())
        v_feat = atom_feature(bond.GetEndAtom())

        uv_feat = {'k': torch.tensor(
            np.concat([edge_feat, u_feat, v_feat], axis=None),
            dtype=torch.float32).view(1, edge_feature_dim)
        }
        vu_feat = {'k': torch.tensor(
            np.concat([edge_feat, v_feat, u_feat], axis=None),
            dtype=torch.float32).view(1, edge_feature_dim)
        }
        # 为无向图
        g.add_edges(u, v, data=uv_feat)
        g.add_edges(v, u, data=vu_feat)

    return g

def smile_process(data):
    drug_1 = smile_to_dgl(data['Drug1_smile'])
    drug_2 = smile_to_dgl(data['Drug2_smile'])
    cell = torch.tensor(data.iloc[6:])
    label = float(data['classification'])
    return {'d_1': drug_1, 'd_2': drug_2, 'cell': cell, 'label': label}


def join_data():
    # 读取数据
    combination = pd.read_csv(
        "./data/combination.csv",
        usecols=['Drug1', 'Drug2', 'Cell line', 'classification']
    )
    smiles = pd.read_csv("./data/smiles.csv")
    cell_line_features = pd.read_csv("./data/cell line features.csv")
    # 镜像处理数据集
    swapped = combination.copy()
    swapped[['Drug1', 'Drug2']] = combination[['Drug2', 'Drug1']]
    combination = pd.concat([combination, swapped], ignore_index=True)
    combination = combination.drop_duplicates(subset=['Drug1', 'Drug2', 'Cell line'])

    # 连接多个表的数据
    result = pd.merge(combination, smiles, left_on='Drug1', right_on='name', how='inner')
    result = result.rename(columns={'smile': 'Drug1_smile'})
    result = pd.merge(result, smiles, left_on='Drug2', right_on='name', how='inner')
    result = result.rename(columns={'smile': 'Drug2_smile'})
    result = pd.merge(result, cell_line_features, left_on='Cell line', right_on='gene_id', how='inner')
    result.drop(columns=['name_x', 'name_y', 'gene_id'], inplace=True)
    result.loc[result['classification'] == 'synergy', 'classification'] = 1
    result.loc[result['classification'] == 'antagonism', 'classification'] = 0
    result.to_csv('./data/data.csv', index=False)
    return result

def split_data(data):
    train_data = data.sample(frac=0.8, random_state=21)
    scaler = StandardScaler()
    columns_to_scale = train_data.columns[6:]
    train_data[columns_to_scale] = scaler.fit_transform(train_data[columns_to_scale])
    test_data = data.drop(train_data.index)
    test_data[columns_to_scale] = scaler.transform(test_data[columns_to_scale])
    cv_data = test_data.sample(frac=0.5, random_state=21)
    test_data = test_data.drop(cv_data.index)

    return train_data, test_data, cv_data

def save_dgl_data(train_data, test_data, cv_data):
    pandarallel.initialize(nb_workers=os.cpu_count(), progress_bar=True)

    encodings_path = "./encodings/dgl"
    if not os.path.exists(encodings_path):
        os.makedirs(encodings_path)

    train_data.to_csv("./data/train.csv", index=False)
    cv_data.to_csv("./data/cv.csv", index=False)
    test_data.to_csv("./data/test.csv", index=False)

    # 保存到.pkl文件
    train_data = train_data.parallel_apply(smile_process, axis=1).tolist()
    with open(os.path.join(encodings_path, "train.pkl"), "wb") as f:  # 注意使用二进制写入模式 'wb'
        pickle.dump(train_data, f)
    del train_data

    cv_data = cv_data.parallel_apply(smile_process, axis=1).tolist()
    with open(os.path.join(encodings_path, "cv.pkl"), "wb") as f:  # 注意使用二进制写入模式 'wb'
        pickle.dump(cv_data, f)
    del cv_data

    test_data = test_data.parallel_apply(smile_process, axis=1).tolist()
    with open(os.path.join(encodings_path, "test.pkl"), "wb") as f:  # 注意使用二进制写入模式 'wb'
        pickle.dump(test_data, f)
    del test_data

def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            ["C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na",
            "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag",
            "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd",
            "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown", ],
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(
            atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        + one_of_k_encoding_unk(
            atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        + [atom.GetIsAromatic()]
    )


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def pyg_smile_process(data):
    drug_1 = smile_to_graph(data['Drug1_smile'])
    drug_2 = smile_to_graph(data['Drug2_smile'])
    cell = torch.tensor(data.iloc[6:])
    label = torch.zeros(2, dtype=torch.float32)
    label[data['classification']] = 1
    return {'d_1': drug_1, 'd_2': drug_2, 'cell': cell, 'label': label}

def save_pyg_data(train_data, test_data, cv_data):
    pandarallel.initialize(nb_workers=os.cpu_count(), progress_bar=True)

    encodings_path = "./encodings/pyg"
    if not os.path.exists(encodings_path):
        os.makedirs(encodings_path)

    train_data = train_data.parallel_apply(pyg_smile_process, axis=1).tolist()
    with open(os.path.join(encodings_path, "train.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    del train_data
    cv_data = cv_data.parallel_apply(pyg_smile_process, axis=1).tolist()
    with open(os.path.join(encodings_path, "cv.pkl"), "wb") as f:
        pickle.dump(cv_data, f)
    del cv_data
    test_data = test_data.parallel_apply(pyg_smile_process, axis=1).tolist()
    with open(os.path.join(encodings_path, "test.pkl"), "wb") as f:
        pickle.dump(test_data, f)
    del test_data


def data_process():
    print("读取数据......")
    result = join_data()
    train_data, test_data, cv_data = split_data(result)
    print("保存dgl数据......")
    save_dgl_data(train_data, test_data, cv_data)
    print("保存pyg数据......")
    save_pyg_data(train_data, test_data, cv_data)

def collate_pyg(batch):
    return {
        'drug1': Batch.from_data_list([item['drug1'] for item in batch]),
        'drug2': Batch.from_data_list([item['drug2'] for item in batch]),
        'cell': torch.stack([item['cell'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch])
    }

def collate_fn(batch):
    d_1, d_2, cell, label = zip(*batch)
    return dgl.batch(d_1), dgl.batch(d_2), torch.stack(cell), torch.tensor(label)