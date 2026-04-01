import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import torch
import math
from torch import Tensor
from typing import Optional
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules.module import Module

def Pear_corr(x):
    x = x.T
    x_pd = pd.DataFrame(x)
    dist_mat = x_pd.corr()
    return dist_mat.to_numpy()


def Eu_dis(x):
    dist_mat = cdist(x, x, 'euclid')
    print('dist_mat shape:', dist_mat.shape)
    return dist_mat


# Concat the features from multiple modalities
def Multi_omics_feature_concat(*F_list, normal_col=False):
    features = None
    for f in F_list:
        if f is not None and f != []:
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


# Concat the modality-specific hyperedges
def Multi_omics_hyperedge_concat(*H_list):
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


# generate G from incidence matrix H
def generate_G_from_H(H, variable_weight=False):
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge, here we use 1 for each hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1, edge_type='euclid'):
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))

    if edge_type == 'euclid':
        for center_idx in range(n_obj):
            dis_mat[center_idx, center_idx] = 0
            dis_vec = dis_mat[center_idx]

            nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
            avg_dis = np.average(dis_vec)
            if not np.any(nearest_idx[:k_neig] == center_idx):
                nearest_idx[k_neig - 1] = center_idx

            for node_idx in nearest_idx[:k_neig]:
                if is_probH:
                    H[node_idx, center_idx] = np.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)

                else:
                    H[node_idx, center_idx] = 1.0
        print('use euclid for H construction')

    elif edge_type == 'pearson':
        for center_idx in range(n_obj):
            dis_mat[center_idx, center_idx] = -999.
            dis_vec = dis_mat[center_idx]
            nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
            nearest_idx = nearest_idx[::-1]

            avg_dis = np.average(dis_vec)
            if not np.any(nearest_idx[:k_neig] == center_idx):
                nearest_idx[k_neig - 1] = center_idx

            for node_idx in nearest_idx[:k_neig]:
                if is_probH:
                    H[node_idx, center_idx] = 1. - np.exp(-(dis_vec[node_idx] + 1.0) ** 2)
                else:
                    H[node_idx, center_idx] = 1.0
        print('use pearson for H construction')

    return H


def construct_H_with_KNN(X, K_neigs=[10], is_probH=True, m_prob=1, edge_type='euclid'):
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    if edge_type == 'euclid':
        dis_mat = Eu_dis(X)

    elif edge_type == 'pearson':
        dis_mat = Pear_corr(X)

    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob, edge_type)
        H = Multi_omics_hyperedge_concat(H, H_tmp)

    return H


def sample_(label, y):
    idx = np.where(y == label)[0]
    return np.random.choice(idx)


def add_self_loops(edge_index, edge_weight: Optional[torch.Tensor] = None,
                   fill_value: float = 1., num_nodes: Optional[int] = None):
    N = num_nodes

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    # if edge_index.min() > 0:
    #     loop_index = loop_index + edge_index.min()

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((N,), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight

class SparseLinear(Module):
    r"""
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # init.ones_(self.weight)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # wb=torch.sparse.mm(input,self.weight.T).to_dense()+self.bias
        # print(self.weight.T)
        wb=torch.sparse.mm(input,self.weight.T)
        if self.bias is not None:
            out = wb + self.bias
        else:
            out = wb
        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
