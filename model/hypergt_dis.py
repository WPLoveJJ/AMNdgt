import math,os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from utils import SparseLinear

BIG_CONSTANT = 1e8

def create_projection_matrix(m, d, seed=0, scaling=0, struct_mode=False):#创建一个随机正交投影矩阵，用于线性复杂度注意力（把高维特征投影到低维，避免 O (N²)）
    '''
    m: 投影矩阵的行数
    d: 投影矩阵的列数
    seed: 随机数种子
    scaling: 0 或 1
    struct_mode: 是否使用结构化模式
    ϕ(q)=1/√M ⋅ relu(ωq+b)
    随机傅里叶特征（RFF） 用来近似 softmax 注意力，让复杂度从 O (N²) → O (N)
    '''
    nb_full_blocks = int(m/d)#0#计算能完整放下多少个 d×d 正交块
    block_list = []#创建空列表，用来存放生成的正交矩阵块
    current_seed = seed#把传入的种子赋值给当前种子
    for _ in range(nb_full_blocks):#循环生成完整的 d×d 正交块
        torch.manual_seed(current_seed)#固定随机种子，保证每次生成的随机矩阵一样
        if struct_mode:#判断是否使用结构化正交矩阵（吉文斯旋转），保证投影矩阵正交性 W^TW=I。
            q = create_products_of_givens_rotations(d, current_seed)#调用吉文斯旋转函数，生成结构化正交矩阵；比 QR 更快、更稳定
        else:#不使用结构化 → 走标准随机正交矩阵
            unstructured_block = torch.randn((d, d))#生成随机 d×d 的标准正态分布随机矩阵
            q, _ = torch.qr(unstructured_block)#QR 分解 → 得到正交矩阵 Q
            q = torch.t(q)#把 QR 分解得到的正交矩阵 转置
        block_list.append(q)#将刚才生成好的 d×d 正交矩阵块 放进列表 block_list
        current_seed += 1#把当前随机种子 +1，保证下一次生成的矩阵块 和上一次完全独立
    remaining_rows = m - nb_full_blocks * d#算一算还差几行才能凑够目标行数 m，剩余行数 = 目标行数 - 已生成完整块的总行数
    if remaining_rows > 0:#判断是否还有不够一个完整块的剩余行数，让最终矩阵刚好是 m 行
        torch.manual_seed(current_seed)#给剩余行的矩阵设置随机种子
        if struct_mode:# 剩余部分同样判断是否使用结构化正交矩阵
            q = create_products_of_givens_rotations(d, current_seed)#生成结构化正交矩阵
        else:#用随机矩阵 + QR 分解生成正交矩阵
            unstructured_block = torch.randn((d, d))#生成一个 d×d 的随机高斯矩阵
            q, _ = torch.qr(unstructured_block)#对随机矩阵执行 QR 分解，得到列正交矩阵 q
            q = torch.t(q)#对 QR 分解得到的正交矩阵进行转置
        block_list.append(q[0:remaining_rows])#把截取的剩余部分加入列表，q[:remaining_rows]：只取前 remaining_rows 行
    final_matrix = torch.vstack(block_list)#把 block_list 里的所有矩阵块，按行拼接，形状：(m, d) → 目标随机投影矩阵

    current_seed += 1#保证下次生成矩阵时不重复、独立
    torch.manual_seed(current_seed)#固定随机种子，保证可复现性
    if scaling == 0:#scaling=0：行范数缩放
        multiplier = torch.norm(torch.randn((m, d)), dim=1)#给每一行乘上一个随机范数值，保持投影矩阵特性
    elif scaling == 1:#固定常数缩放
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)#所有行统一乘 √d，保证数值稳定性（常用在随机特征）
    else:
        raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)#非法参数直接报错，保证代码健壮性

    return torch.matmul(torch.diag(multiplier), final_matrix)#对角矩阵 × 投影矩阵 = 完成缩放,返回最终可用的随机正交投影矩阵

def create_products_of_givens_rotations(dim, seed):#随机正交投影矩阵 W；Q⊤Q=I，QQ⊤=I;用 Givens 旋转乘积快速生成正交矩阵，替代 QR 分解，满足正交性 Q⊤Q=I，给 线性注意力 / 核化 softmax 提供随机投影矩阵
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))#旋转次数 = 维度 × log (维度)；少量旋转就足够让矩阵接近均匀随机正交矩阵
    q = np.eye(dim, dim)#初始化为单位矩阵：
    np.random.seed(seed)#固定随机种子，保证可复现，和主函数的种子保持一致。
    for _ in range(nb_givens_rotations):#循环执行 Givens 旋转乘积：Q=G1G2…Gk
        random_angle = math.pi * np.random.uniform()#θ∼Uniform(0,π)
        random_indices = np.random.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])#随机选两个不同维度 i, j，只在这两个维度做旋转
        slice_i = q[index_i]
        slice_j = q[index_j]#取出矩阵的第 i 行 和 j 行。
        #结构化正交矩阵构造，用来替代 QR 分解，更快、更省显存
        new_slice_i = math.cos(random_angle) * slice_i + math.cos(random_angle) * slice_j#q`i = cosθqi + cosθqj
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j#q`j = -sinθqi + cosθqj
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j#旋转后的行写回矩阵
    return torch.tensor(q, dtype=torch.float32)#结构化正交矩阵 Q

def relu_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.001):#ReLU 核的线性化随机特征映射,把标准注意力 O(N2) 变成 线性复杂度 O(N)
    '''
     KReLU(x,z)=max(x⊤z,0)
    '''
    del is_query## ReLU 核对称，Query 和 Key 用同一变换，无需区分
    if projection_matrix is None:
        return F.relu(data) + numerical_stabilizer#ϕ(x)=ReLU(x)+ϵ,ϵ = 数值稳定项，防止除 0。
    else:#随机特征线性化 ReLU 核分支（核心）
        ratio = 1.0 / torch.sqrt(
            torch.tensor(projection_matrix.shape[0], torch.float32)
        )#归一化系数,ratio=1/√M，M = 随机特征维度（projection_matrix.shape [0]），保证随机映射后方差不变，满足无偏估计
        data_dash = ratio * torch.einsum("bnhd,md->bnhm", data, projection_matrix)#~x =1/√M · x W^T data: [B, N, H, d] → 图节点特征;projection_matrix W: [M, d] → 正交随机矩阵;data_dash: [B, N, H, M] → 低维随机特征
        return F.relu(data_dash) + numerical_stabilizer#ϕ(x)=ReLU( 1/√M ·xW^⊤)+ϵ ；ϕ(x)^⊤ϕ(z)≈max(x^⊤z,0),把原本的 QK⊤ 点积，替换成 低维随机特征内积，复杂度从 O(N2) → O(N)。

def softmax_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.000001):#核化 Softmax 的随机特征映射 exp(​QK^⊤/√d​)≈ϕ(Q)ϕ(K)^⊤，把 O(N2) 降为 O(N);K(x,z)=exp(x^⊤z/√d),K(x,z)≈ϕ(x)^⊤ϕ(z)
    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))#1/√d
    data = data_normalizer * data#ˉx=x/d^1/4
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32))#ratio=1/√M ;M = 随机特征维度（projection_matrix.shape[0]）
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)# 3. 随机正交投影（核心：高维 → 低维）#xW^⊤
    ## 4. 计算指数的二次项部分：-0.5 * ||x||²
    diag_data = torch.square(data)## 平方
    diag_data = torch.sum(diag_data, dim=len(data.shape)-1)# 最后一维求和（二范数平方）
    diag_data = diag_data / 2.0# 除以 2
    diag_data = torch.unsqueeze(diag_data, dim=len(data.shape)-1) # 保持维度对齐
    last_dims_t = len(data_dash.shape) - 1    # M 维度
    attention_dims_t = len(data_dash.shape) - 3  # N 维度（节点数）
    '''
exp( ˉx^⊤ˉz − 1/2||x||^2 - 1/2||ˉz||^2)
    '''
    if is_query: ## 5. Query 分支：指数 + 归一化 + 稳定  ϕ(x)=1/√M exp( ˉx^⊤ˉz − 1/2||x||^2 -max (xW^T)) + ϵ  ;max(...)：数值稳定技巧，防止指数上溢,ϵ：防止除 0 /log (0)
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash, dim=last_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )
    else:# # 6. Key 分支：全局 max 保证稳定
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                    dim=attention_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )#ϕ(x)=1/√M exp( ˉx^⊤ˉz − 1/2||x||^2 -max_all (xW^T)) + ϵ,Key 做全局 max，保证所有 Key 归一化一致
    return data_dash#核化 Softmax 随机特征ϕ(x)         exp( x^T z / √d ) ≈ ϕ(x)^T ϕ(z)


def numerator(qs, ks, vs):#线性注意力的分子计算,复杂度：O(N);数学等价：Q(K⊤V)
    ## 第一步：对所有节点，计算全局上下文矩阵 K^T V
    # 对应论文：U_k = sum(key^T value)
    kvs = torch.einsum("nbhm,nbhd->bhmd", ks, vs) # kvs refers to U_k in the paper #KV=∑_n^N=1 ​K_n^⊤​V_n​,∈R^B×H×M×D
    # 第二步：每个节点 query 直接乘以全局矩阵，得到注意力输出
    return torch.einsum("nbhm,bhmd->nbhd", qs, kvs)#Z=Q⋅(K^⊤V)

def denominator(qs, ks):#计算核化注意力的分母，Denominator=Q⋅(∑K)
    all_ones = torch.ones([ks.shape[0]]).to(qs.device)#生成全 1 向量，用来对 Key 做求和
    ks_sum = torch.einsum("nbhm,n->bhm", ks, all_ones) # ks_sum refers to O_k in the paper #O_k=∑_n=1^N​ K_n​ #全局 Key 和

    return torch.einsum("nbhm,bhm->nbh", qs, ks_sum)#Denominator_n​=Q_n ⋅O_k​=Q_n​ ⋅ (∑_n=1^N​ K_n)   Z_den =Q∑K

def numerator_gumbel(qs, ks, vs):#对应论文：Gumbel 增强版 Q(K⊤V)#训练时加入 Gumbel 噪声，让注意力更稳定
    kvs = torch.einsum("nbhkm,nbhd->bhkmd", ks, vs) # kvs refers to U_k in the paper #KV=∑_n^N=1 ​K_n^⊤​V_n​,∈R^B×H×M×D
    return torch.einsum("nbhm,bhkmd->nbhkd", qs, kvs)#Numeratorn​=Qn​⋅Uk​
    '''
    Attention(Q,K,V)=  Q(K^⊤*V) /​ Q∑K
    '''

def denominator_gumbel(qs, ks):#对应论文：Gumbel 版归一化项
    all_ones = torch.ones([ks.shape[0]]).to(qs.device)#全 1 向量 → 节点求和
    ks_sum = torch.einsum("nbhkm,n->bhkm", ks, all_ones) # ks_sum refers to O_k in the paper #O_k=∑_n=1^N​ K_n​ #全局 Key 和
    return torch.einsum("nbhm,bhkm->nbhk", qs, ks_sum)#Denominatorn​=Qn​⋅Ok​

def kernelized_softmax(query, key, value, kernel_transformation, projection_matrix=None, edge_index=None, tau=0.25, return_weight=True):#HyperGT 的注意力核心
    '''
    fast computation of all-pair attentive aggregation with linear complexity
    input: query/key/value [B, N, H, D]
    return: updated node emb, attention weight (for computing edge loss)
    B = graph number (always equal to 1 in Node Classification), N = node number, H = head number,
    M = random feature dimension, D = hidden size
    '''
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)#Q←Q/√τ,K←K/√τ，温度缩放，稳定指数计算。
    query_prime = kernel_transformation(query, True, projection_matrix) # [B, N, H, M] #
    key_prime = kernel_transformation(key, False, projection_matrix) # [B, N, H, M]#Q′=ϕ(Q),K′=ϕ(K) #对应论文 4.2 节：这里用随机特征近似：exp(QK^⊤)≈ϕ(Q)ϕ(K)^⊤
    #维度对齐，为全局求和做准备。
    query_prime = query_prime.permute(1, 0, 2, 3) # [N, B, H, M]
    key_prime = key_prime.permute(1, 0, 2, 3) # [N, B, H, M]
    value = value.permute(1, 0, 2, 3) # [N, B, H, D]

    # compute updated node emb, this step requires O(N)
    z_num = numerator(query_prime, key_prime, value)#Z_num=Q′(∑_jK_j′^⊤V_j) 对应论文 4.2 节更新规则：
    z_den = denominator(query_prime, key_prime)#Z_den=Q^′(∑_j K_j′)  对应 softmax 归一化项：∑_j=1^n+m -​A_ij^(ℓ)​

    z_num = z_num.permute(1, 0, 2, 3)  # [B, N, H, D]
    z_den = z_den.permute(1, 0, 2)
    z_den = torch.unsqueeze(z_den, len(z_den.shape)) #最后一维升维到 1，让分母可以和分子 [B, N, H, D] 做广播除法， # [B, N, H] → [B, N, H, 1]
    z_output = z_num / z_den # [B, N, H, D]#对应 HyperGT 论文核心公式（4.2 节），完全等价于超图全局注意力聚合
    if return_weight: # query edge prob for computing edge-level reg loss, this step requires O(E)# Aij =ϕ(Qi)⊤ϕ(K j)≈exp((z_i W_Q)(z_j W_K)^T)
        start, end = edge_index
        query_end, key_start = query_prime[end], key_prime[start] # [E, B, H, M]
        edge_attn_num = torch.einsum("ebhm,ebhm->ebh", query_end, key_start) # [E, B, H]
        edge_attn_num = edge_attn_num.permute(1, 0, 2) # [B, E, H]
        attn_normalizer = denominator(query_prime, key_prime) # [N, B, H]
        edge_attn_dem = attn_normalizer[end]  # [E, B, H]
        edge_attn_dem = edge_attn_dem.permute(1, 0, 2) # [B, E, H]
        A_weight = edge_attn_num / edge_attn_dem # [B, E, H] #对应论文 4.3 节结构正则

        return z_output, A_weight

    else:
        return z_output

def kernelized_gumbel_softmax(query, key, value, kernel_transformation, projection_matrix=None, edge_index=None,
                                K=10, tau=0.25, return_weight=False):
    #训练时加噪声，让注意力更稳定、更强泛化                            
    # print('query:',query)
    # print('key:',key)
    # print('value',value)           
    # print('kernel_transformation',kernel_transformation)                 
    '''
    kernel_transformation: 一个函数，用于将Q,K映射到高维随机特征空间（这是核技巧的关键）。
    K: Gumbel 采样的次数（默认为 10）。tau: 温度参数，控制 Gumbel-Softmax 的平滑程度。
    fast computation of all-pair attentive aggregation with linear complexity
    input: query/key/value [B, N, H, D]，B: Batch size (通常是 1)。N: 节点数量。H: 多头注意力的头数。D: 每个头的特征维度。
    return: updated node emb, attention weight (for computing edge loss)
    B = graph number (always equal to 1 in Node Classification), N = node number, H = head number,
    M = random feature dimension, D = hidden size, K = number of Gumbel sampling
    '''
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)
    query_prime = kernel_transformation(query, True, projection_matrix) # [B, N, H, M] kernel_transformation 将原始维度D映射到一个随机特征维度M
    key_prime = kernel_transformation(key, False, projection_matrix) # [B, N, H, M]
    query_prime = query_prime.permute(1, 0, 2, 3) # [N, B, H, M] 为了方便后续的矩阵乘法，将 N (节点数) 移到第一维。
    key_prime = key_prime.permute(1, 0, 2, 3) # [N, B, H, M] 为了方便后续的矩阵乘法，将 N (节点数) 移到第一维。
    value = value.permute(1, 0, 2, 3) # [N, B, H, D] 为了方便后续的矩阵乘法，将 N (节点数) 移到第一维。

    # compute updated node emb, this step requires O(N)
    #生成 Gumbel 分布的噪声。 G=−log(−log(U))，其中 U∼Uniform(0,1)。代码里用了 exponential_().log() 等价实现。
    #形状: [N, B, H, K]。这意味着为每个节点、每个头生成了 K组独立的噪声。除以 tau: 引入温度系数
    gumbels = (
        -torch.empty(key_prime.shape[:-1]+(K, ), memory_format=torch.legacy_contiguous_format).exponential_().log()
    ).to(query.device) / tau # [N, B, H, K]
    key_t_gumbel = key_prime.unsqueeze(3) * gumbels.exp().unsqueeze(4) # [N, B, H, K, M]  #这里的 gumbels.exp() 实际上是在模拟从 Attention 分布中采样的过程。这里把噪声直接乘到了 Key 上，用于后续的聚合。
    '''
    Attention: O=softmax(QK^T)V≈ϕ(Q)(ϕ(K)^TV)/ϕ(Q)ϕ(K)^T
    '''
    z_num = numerator_gumbel(query_prime, key_t_gumbel, value) # [N, B, H, K, D]#先计算 分子：∑jϕ(Kj)⋅Vj（这一步是 O(N)，因为只遍历一次节点）。再乘以 ϕ(Qi),结果: 包含了 Value 的聚合信息。
    z_den = denominator_gumbel(query_prime, key_t_gumbel) # [N, B, H, K] #计算分母：归一化因子

    z_num = z_num.permute(1, 0, 2, 3, 4) # [B, N, H, K, D]
    z_den = z_den.permute(1, 0, 2, 3) # [B, N, H, K]
    z_den = torch.unsqueeze(z_den, len(z_den.shape))# [B, N, H, K, 1]
    z_output = torch.mean(z_num / z_den, dim=3) # [B, N, H, D]  完成了类似 Softmax 的归一化，因为我们采样了 K次（dim=3 是 K 维度），这 K 次采样的结果取平均，作为最终的节点表示，期望的近似

    if return_weight: # query edge prob for computing edge-level reg loss, this step requires O(E)  return_weight=True，代码会额外计算边上的 Attention 权重，通常用于正则化 Loss。
        start, end = edge_index
        # 取出边的起点和终点的映射特征
        query_end, key_start = query_prime[end], key_prime[start] # [E, B, H, M]
        # 计算点积 Attention Score (未归一化)
        edge_attn_num = torch.einsum("ebhm,ebhm->ebh", query_end, key_start) # [E, B, H]
        edge_attn_num = edge_attn_num.permute(1, 0, 2) # [B, E, H]
         # 计算分母 (归一化因子)
        attn_normalizer = denominator(query_prime, key_prime) # [N, B, H]
        edge_attn_dem = attn_normalizer[end]  # [E, B, H]
        edge_attn_dem = edge_attn_dem.permute(1, 0, 2) # [B, E, H]
         # 得到最终的 Edge Attention Weight
        A_weight = edge_attn_num / edge_attn_dem # [B, E, H]

        return z_output, A_weight

    else:
        return z_output
    #Gumbel 采样: 引入随机噪声 G，让模型能够探索���同的注意力分布，而不是总是盯着概率最大的那个。    

def add_conv_relational_bias(x, edge_index, b, trans='sigmoid'):
    '''
    图卷积网络 (GCN) 风格的结构偏差
    Transformer 本身是全连接的（Global），但图通常具有局部性（Local）。这个函数的作用就是 在全局 Attention 计算之外，显式地加入图的邻接结构信息，作为一种“关系偏置” (Bias)。
    通俗来说：它让模型在看全图的同时，也重点关注一下自己的直系邻居。
    x: 形状 [B, N, H, D]。也就是当前层的节点特征（Query/Key/Value 或中间结果）。 
    edge_index: 形状 [2, E]。这是图的边列表（邻接矩阵的稀疏表示）
    b: 形状 [H] (Head Number)。这是一个 可学习的参数，每个 Attention Head 有一个独立的偏置值bi。
    trans: 激活函数类型，默认为 sigmoid
    compute updated result by the relational bias of input adjacency
    the implementation is similar to the Graph Convolution Network with a (shared) scalar weight for each edge
    '''
    #标准的 GCN 归一化公式：D^−1/2 A D^−1/2
    row, col = edge_index
    d_in = degree(col, x.shape[1]).float()# 计算入度 degree(col): 计算每个节点的度（Degree）
    d_norm_in = (1. / d_in[col]).sqrt()# 归一化因子的左半部分 D^(-1/2)
    d_out = degree(row, x.shape[1]).float()# 计算出度
    d_norm_out = (1. / d_out[row]).sqrt()# 归一化因子的右半部分 D^(-1/2)
    # 多头循环计算,逐个 Head 处理
    conv_output = []
    for i in range(x.shape[2]):# x.shape[2] 是头数 H
        if trans == 'sigmoid':
            b_i = F.sigmoid(b[i])#F.sigmoid(b[i]): 将其限制在 (0,1)之间，作为一个门控系数 (Gating Coefficient)。这意味着这个 Bias 最多只能以 1.0 的强度加入，最少是 0.0（完全忽略图结构）。
        elif trans == 'identity':
            b_i = b[i] #b[i]: 第 i 个头的可学习偏置参数
        else:
            raise NotImplementedError
        value = torch.ones_like(row) * b_i * d_norm_in * d_norm_out #GCN 归一化: 1/√(d_u*d_v),这是一个形状为 [E] 的向量，每条边对应一个值
        adj_i = SparseTensor(row=col, col=row, value=value, sparse_sizes=(x.shape[1], x.shape[1]))#构建稀疏邻接矩阵 Abias,row=col, col=row: 注意这里的转置，通常 PyG 的 edge_index 是 [Source, Target]，而矩阵乘法通常是 AX，所以行索引对应 Target，列索引对应 Source。
        conv_output.append( matmul(adj_i, x[:, :, i]) )  # [B, N, D] # X_out=A_bias⋅X_in
    conv_output = torch.stack(conv_output, dim=2) # [B, N, H, D] #将 H个头的计算结果堆叠回原来的形状 [B, N, H, D]
    #这个函数的作用是： X_new=X_attn+GCN(X_in,bias)  Global + Local 的结合
    return conv_output 

class NodeFormerConv(nn.Module):
    '''
    NodeFormer 是一个基于 Transformer 的图神经网络，其核心思想是：利用线性复杂度的 Kernelized Attention 进行全局信息聚合，同时融合传统的图结构偏置 (Relational Bias)
    one layer of NodeFormer that attentive aggregates all nodes over a latent graph
    return: node embeddings for next layer, edge loss at this layer
    '''
    def __init__(self, in_channels, out_channels, num_heads, kernel_transformation=softmax_kernel_transformation, projection_matrix_type='a',
                 nb_random_features=10, use_gumbel=True, nb_gumbel_sample=10, rb_order=0, rb_trans='sigmoid', use_edge_loss=True):
        '''
        in_channels, out_channels: 输入/输出特征维度。
        num_heads: 多头注意力头数。
        kernel_transformation: 核函数类型（默认 Softmax）。
        projection_matrix_type: 随机投影矩阵类型（默认为 'a'，通常指某种特定的正交或高斯分布）。
        nb_random_features: 随机特征维度 M核技巧中的关键参数）。
        use_gumbel: 是否使用 Gumbel-Softmax 采样（这就是刚才那个 kernelized_gumbel_softmax 函数）。
        nb_gumbel_sample: Gumbel 采样次数 K。
        rb_order: 关系偏置阶数（Relational Bias Order）。如果是 1，就是 GCN（一阶邻居）；如果是 2，就是二阶邻居，以此类推。
        use_edge_loss: 是否计算边正则化 Loss。
        '''         
        super(NodeFormerConv, self).__init__()
        #定义标准的 Transformer 投影矩阵 W_Q,W_K,W_V和输出矩阵 W_O
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wv = nn.Linear(in_channels, out_channels * num_heads)
        self.Wo = nn.Linear(out_channels * num_heads, out_channels)
        if rb_order >= 1:#定义可学习的偏置参数 b。形状 [Order, Heads]，每个阶数、每个头都有独立的偏置
            self.b = torch.nn.Parameter(torch.FloatTensor(rb_order, num_heads), requires_grad=True)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.kernel_transformation = kernel_transformation
        self.projection_matrix_type = projection_matrix_type
        self.nb_random_features = nb_random_features
        self.use_gumbel = use_gumbel
        self.nb_gumbel_sample = nb_gumbel_sample
        self.rb_order = rb_order
        self.rb_trans = rb_trans
        self.use_edge_loss = use_edge_loss

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        self.Wv.reset_parameters()
        self.Wo.reset_parameters()
        if self.rb_order >= 1:
            if self.rb_trans == 'sigmoid':
                torch.nn.init.constant_(self.b, 0.1)
            elif self.rb_trans == 'identity':
                torch.nn.init.constant_(self.b, 1.0)

    def forward(self, args, z, adjs, tau):
        '''
        z: 节点特征 [B, N, D]
        adjs: 邻接矩阵列表。adjs[0] 是一阶邻居，adjs[1] 是二阶，以此类推
        tau: Gumbel-Softmax 温度
        '''
        B, N = z.size(0), z.size(1)
        #将输入特征 Z投影到 Query, Key, Value 空间，并 reshape 为多头形式 [B, N, H, D]
        query = self.Wq(z).reshape(-1, N, self.num_heads, self.out_channels)
        key = self.Wk(z).reshape(-1, N, self.num_heads, self.out_channels)
        value = self.Wv(z).reshape(-1, N, self.num_heads, self.out_channels)
        #生成随机投影矩阵 
        if self.projection_matrix_type is None:
            projection_matrix = None
        else:
            dim = query.shape[-1]
            seed = torch.ceil(torch.abs(torch.sum(query) * BIG_CONSTANT)).to(torch.int32)#seed 的计算是为了保证在同一个 Batch 内生成相同的随机矩阵（但实际上通常是随机的，这里为了复现性或特定的哈希 trick 做了处理）
            # seed = 0
            projection_matrix = create_projection_matrix(
                self.nb_random_features, dim, seed=seed).to(query.device)#create_projection_matrix 生成一个随机矩阵 Ω，用于将特征 x 映射为 ϕ(x)=exp(Ωx)等形式

        # compute all-pair message passing update and attn weight on input edges, requires O(N) or O(N + E)
        #全局注意力聚合
        #训练时: 使用 Gumbel-Softmax (kernelized_gumbel_softmax) 引入随机性，增强鲁棒性并模拟离散结构学习。
        #推理时: 使用标准的 Kernel Softmax (kernelized_softmax)，去除随机噪声，得到确定性结果。adjs[0] 传入是为了在 return_weight=True 时计算真实边上的 Attention 权重，用于正则化
        if self.use_gumbel and self.training:  # only using Gumbel noise for training  ## 训练阶段使用 Gumbel 噪声
            if self.use_edge_loss:
                # 调用 kernelized_gumbel_softmax (带噪声采样)
                # 计算得到 z_next (聚合后的特征) 和 weight (边权重, 用于 Loss)
                z_next, weight = kernelized_gumbel_softmax(query,key,value,self.kernel_transformation,projection_matrix,adjs[0],
                                                    self.nb_gumbel_sample, tau, self.use_edge_loss)
            else:
                z_next = kernelized_gumbel_softmax(query,key,value,self.kernel_transformation,projection_matrix,adjs[0],
                                                    self.nb_gumbel_sample, tau, self.use_edge_loss)
        else:
            if self.use_edge_loss:
                # 推理阶段或不使用 Gumbel
                # 调用 kernelized_softmax (确定性计算)
                z_next, weight = kernelized_softmax(query, key, value, self.kernel_transformation, projection_matrix, adjs[0],
                                                    tau, self.use_edge_loss)
            else:
                z_next = kernelized_gumbel_softmax(query,key,value,self.kernel_transformation,projection_matrix,adjs[0],
                                                    self.nb_gumbel_sample, tau, self.use_edge_loss)
       
        # compute update by relational bias of input adjacency, requires O(E)
        for i in range(self.rb_order):
            z_next += add_conv_relational_bias(value, adjs[i], self.b[i], self.rb_trans) #add_conv_relational_bias 加入 GCN 卷积;self.b[i] 是可学习的系数，控制局部信息的权重。如果 rb_order=0，这步跳过，就是一个纯 Transformer。rb_order=1，这步加上了一层 GCN。
        #输出投影
        #引导 Attention 关注真实的图结构
        #weight 是 Attention Map 在真实边上的值,d_norm 是归一化因子（度数的倒数）,link_loss: 计算加权对数似然损失
        # aggregate results of multiple heads
        z_next = self.Wo(z_next.flatten(-2, -1))
        #边正则化 Loss
        if self.use_edge_loss: # compute edge regularization loss on input adjacency
            row, col = adjs[0]
            d_in = degree(col, query.shape[1]).float()
            d_norm = 1. / d_in[col]
            d_norm_ = d_norm.reshape(1, -1, 1).repeat(1, 1, weight.shape[-1])
            link_loss = torch.mean(weight.log() * d_norm_)
            return z_next, link_loss
        else:
            return z_next
    '''
    Global: 使用 kernelized_gumbel_softmax 进行线性复杂度的全局注意力聚合。
    Local: 使用 add_conv_relational_bias 进行 GCN 式的局部邻居增强。
    Train: 使用 Gumbel 噪声和 Edge Loss 辅助训练。
    Inference: 使用确定性 Attention。
    '''        

        
class HyperGT(nn.Module):
    '''
    HyperGT model implementation
    return: predicted node labels, a list of edge losses at every layer
    num_nodes：节点数 n,
    num_hes：超边数 m,
    in_channels：输入特征维度,
    hidden_channels：隐藏层维度 d,
    out_channels：分类类别数,
    use_edge_loss=True：开启结构正则化损失 Ls
    '''
    def __init__(self,num_tokens, num_nodes, in_channels, hidden_channels, out_channels, num_hes, num_layers=2, num_heads=4, dropout=0.0,
                 kernel_transformation=softmax_kernel_transformation, nb_random_features=30, use_bn=True, use_gumbel=True,
                 use_residual=True, use_act=False, use_jk=False, nb_gumbel_sample=10, rb_order=0, rb_trans='sigmoid', use_edge_loss=True):
        super(HyperGT, self).__init__()

        self.convs = nn.ModuleList()#创建一个空的模块列表，用来存放所有HyperGT 层（NodeFormerConv），自动管理参数、训练、保存加载；self.convs 存图卷积 / 注意力层
        self.fcs = nn.ModuleList()#创建另一个空的模块列表，专门存放全连接层（包括输入层、输出层），self.fcs 存普通线性变换层
        #公式里的 Z(0)=W0​X（初始特征投影）
        self.fcs.append(nn.Linear(in_channels, hidden_channels))#给全连接层列表添加第一个线性层，功能：把输入特征维度 in_channels 映射到隐藏层维度 hidden_channels
        self.bns = nn.ModuleList()#创建存放层归一化（LayerNorm） 的模块列表，加速训练、防止梯度爆炸 / 消失
        self.bns.append(nn.LayerNorm(hidden_channels))#层归一化列表添加第一个 LayerNorm 层， y=(x−μ)/(√(σ²+ϵ)​)​∗γ+β，其中 μ/σ 是节点特征的均值 / 方差，γ/β 是可训练参数
        for i in range(num_layers):#循环创建num_layers 层 HyperGT 核心卷积层（NodeFormerConv），公式中的 Z(ℓ)，ℓ 从 1 到 num_layers
            #超图注意力层 Attn(Z^(ℓ−1))
            #输入 / 输出维度都是隐藏层维度，
            #kernel_transformation=softmax_kernel_transformation：注意力核函数（对应论文公式中的 softmax 注意力计算，替代传统的点积注意力）
            #nb_random_features=30：随机特征维度 M（论文中用随机投影降低注意力计算复杂度，从 O (N²) 降到 O (N)）
            #use_gumbel=use_gumbel：是否用 Gumbel 采样（训练时加入噪声增强泛化，对应论文中的正则化策略）
            #use_edge_loss=use_edge_loss：是否计算边损失（对应论文中的结构正则化损失 Ls）
            self.convs.append(
                NodeFormerConv(hidden_channels, hidden_channels, num_heads=num_heads, kernel_transformation=kernel_transformation,
                              nb_random_features=nb_random_features, use_gumbel=use_gumbel, nb_gumbel_sample=nb_gumbel_sample,
                               rb_order=rb_order, rb_trans=rb_trans, use_edge_loss=use_edge_loss))#给卷积层列表 self.convs 添加 num_layers 个 NodeFormerConv 层（HyperGT 的核心注意力层）
            self.bns.append(nn.LayerNorm(hidden_channels))#给层归一化列表 self.bns 追加对应每层 NodeFormerConv 的 LayerNorm，“层归一化 + 激活 + dropout” 
        #创建模型的输出层（分类头）
        if use_jk:
            self.fcs.append(nn.Linear(hidden_channels * num_layers + hidden_channels, out_channels))#输入维度 = 所有层输出拼接（初始层 + num_layers 层卷积），融合多层特征提升表达能力
        else:
            self.fcs.append(nn.Linear(hidden_channels, out_channels))#仅用最后一层的隐藏特征（维度 = hidden_channels）映射到输出类别数（out_channels），公式中的最终分类层 Y=W_final*​Z^(L)（L 是最后一层）

        self.dropout = dropout#把传入的 dropout 参数（默认 0.0）赋值给模型实例变量 self.dropout，dropout 是正则化手段，随机让部分神经元输出为 0，防止过拟合
        self.activation = F.elu#指定模型使用的激活函数为 ELU，ELU(x)={x,​x>0 或 α(e^x−1),x≤0​（PyTorch 中α默认 1.0）
        self.use_bn = use_bn#把传入的 use_bn 参数（默认 True）赋值给实例变量，作为 “是否使用层归一化” 的开关，方便 ablation study（消融实验）—— 比如对比 “用 / 不用 LayerNorm” 对模型性能的影响
        self.use_residual = use_residual#把传入的 use_residual 参数（默认 True）赋值给实例变量，作为 “是否使用残差连接” 的开关，Z(ℓ)=Attn(Z^(ℓ−1))+Z^(ℓ−1)，解决深度模型的梯度消失问题 —— 让梯度可以直接从高层流回低层，训练更深的网络
        self.use_act = use_act#把传入的 use_act 参数（默认 False）赋值给实例变量，作为 “卷积层后是否额外使用激活函数” 的开关
        self.use_jk = use_jk#把传入的 use_jk 参数（默认 False）赋值给实例变量，作为 “是否使用 JK 连接” 的开关
        self.use_edge_loss = use_edge_loss#把传入的 use_edge_loss 参数（默认 True）赋值给实例变量，作为 “是否计算结构正则化损失（边损失）” 的开关，Ls​=−E(u,v)∈E​ log ~A_uv​（惩罚注意力权重偏离真实边结构）;总损失 L=Lc​+λLs,Ls​ 就是这个边损失，λ 是权重系数（训练代码中设置）
        self.n = num_tokens#把传入的 num_tokens 参数赋值给实例变量 self.n
        #H → 给节点加超边相关的位置编码
        self.he_sparse_encoder = SparseLinear(num_hes, hidden_channels)#创建超边位置编码（HEPE） 的稀疏线性层，SparseLinear：专门处理超图邻接矩阵 H（稀疏矩阵）的线性变换；num_hes：超边数量（输入维度），hidden_channels：隐藏层维度（输出维度）P_V​=HW_V， 是超图关联矩阵（n×m），WV​ 是这个稀疏层的权重（m×d）​
        #H^⊤ → 给超边加节点相关的位置编码
        self.hte_sparse_encoder = SparseLinear(num_nodes, hidden_channels)#创建转置超边位置编码（HtEPE） 的稀疏线性层，输入维度：num_nodes（节点数量），输出维度：hidden_channels（隐藏层维度) P_E​=H^⊤W_E​（超边的节点位置编码）——H⊤ 是超图关联矩阵的转置（m×n），WE​ 是这个稀疏层的权重（n×d）

    def reset_parameters(self):#定义 reset_parameters 方法，用于重置所有层的可训练参数（权重 / 偏置），PyTorch 模型的标准方法 —— 训练前 / 多次训练时，重置参数到初始状态，避免参数初始化的随机性影响实验结果
        for conv in self.convs:#遍历 self.convs 中的每一个 NodeFormerConv 层（超图注意力层），为每个卷积层执行参数重置（后续会调用 conv.reset_parameters()）
            conv.reset_parameters()#调用每个 NodeFormerConv 层自身的 reset_parameters 方法，重置该层的所有可训练参数
        #遍历对象：self.bns 包含 1 个输入层的 LayerNorm + num_layers 个卷积层的 LayerNorm
        #循环目的：为每个归一化层执行参数重置（LayerNorm 有可训练的缩放参数γ和偏移参数β）    
        for bn in self.bns:#遍历 self.bns 中的每一个 LayerNorm 层（层归一化层）
            bn.reset_parameters()#调用每个 LayerNorm 层自身的 reset_parameters() 方法，重置其可训练参数；如果模型训练中断后继续训练，γ/β 可能偏离初始值，重置能恢复 “无偏归一化” 的初始状态
        for fc in self.fcs:#遍历 self.fcs 中的每一个全连接层（输出分类头）；循环目的：为输出层执行参数重置（全连接层的权重W和偏置b）
            fc.reset_parameters()#调用输出层（全连接层）自身的 reset_parameters() 方法，重置其权重和偏置参数
        self.he_sparse_encoder.reset_parameters()#重置超边位置编码（HEPE）的稀疏线性层参数
        self.hte_sparse_encoder.reset_parameters()#重置转置超边位置编码（HtEPE）的稀疏线性层参数

    def forward(self, args, x, adjs, H, tau=1.0):#定义 HyperGT 模型的前向传播核心方法，是数据流转、特征计算、损失生成的总入口
        '''
        self：模型实例本身（PyTorch 类方法的标准参数）
        args：命令行参数 / 配置对象（包含位置编码类型 pe 等关键配置，如 args.pe = 'HEPE' 或 'HtEPE'）
        x：节点初始特征矩阵（shape: [num_nodes, in_channels]，一维批次的原始节点特征）
        adjs：超图邻接结构列表（每个元素是 [2, num_edges] 的稀疏索引，对应 NodeFormerConv 所需的边结构）
        H：超图关联矩阵（稀疏 / 稠密矩阵，shape: [num_nodes, num_hes]，核心超图结构输入）
        tau：注意力温度系数（默认 1.0，控制注意力分布的平滑度，tau 越小注意力越集中）
        特征投影→位置编码→多层注意力→残差 / 归一化→输出分类→损失收集
        '''
        #给节点特征矩阵 x 增加批次维度（B）
        x = x.unsqueeze(0) # [B, N, H], B=1 denotes number of graph
        #创建两个空列表，用于存储中间计算结果
        layer_ = []#存储每一层的节点特征（包括初始投影层 + 所有卷积层的输出）
        link_loss_=[]#存储每一层的边损失（link_loss）

        z = self.fcs[0](x)#将原始节点特征投影到隐藏层维度（对应论文公式 Z^(0)=W_0*​X）；x shape [1, N, D_in] → z shape [1, N, D_hid]（D_hid=hidden_channels）
        #self.he_sparse_encoder：稀疏线性层，H：超图关联矩阵 [N, M]（N = 节点数，M = 超边数）；线性变换H × W_V（W_V 是稀疏层权重，[M, D_hid]）→ 输出 [N, D_hid]，unsqueeze(0)：增加批次维度 → he_pe shape [1, N, D_hid]
        he_pe = self.he_sparse_encoder(H).unsqueeze(0)#计算节点的超边位置编码（HEPE）（对应论文公式 P_V​=H*W_V​）
        #H.transpose(0,1)：超图关联矩阵转置 → shape [M, N]（M = 超边数，N = 节点数），self.hte_sparse_encoder：稀疏线性层（SparseLinear(num_nodes, hidden_channels)）
        #输入 H^\top：[M, N] → 线性变换 H^\top × W_E（W_E 权重 [N, D_hid]）→ 输出 [M, D_hid]，unsqueeze(0)：增加批次维度 → hte_pe shape [1, M, D_hid]
        hte_pe = self.hte_sparse_encoder(H.transpose(0,1)).unsqueeze(0)#计算超边的节点位置编码（HtEPE）（对应论文公式 PE​=H^⊤W_E​）
        #如果配置开启 HEPE，将超边位置编码叠加到节点特征中
        if  'HEPE' in args.pe:
            #创建零填充张量，解决 he_pe 与 z 的维度不匹配问题：z.shape[1]：节点数 N；he_pe.shape[1]：超边数 M（通常 M≠N）;padding shape：[1, N-M, D_hid]（如果 M<N）或 [1, 0, D_hid]（如果 M=N）
            padding=torch.zeros(z.shape[0],z.shape[1]-he_pe.shape[1],z.shape[2],requires_grad=False,device=z.device)
            he_pe = torch.cat((he_pe,padding),dim=1)#拼接 HEPE 和零填充 → shape [1, N, D_hid]（与 z 维度一致）
            z = z + he_pe#位置编码残差叠加到节点特征中（保留原始特征，仅增加结构信息），零填充保证维度匹配，残差叠加避免位置编码覆盖原始特征
        #如果配置开启 HtEPE，将转置超边位置编码叠加到节点特征中  
        # 填充逻辑相反：hte_pe 原始 shape [1, M, D_hid]（M = 超边数），零填充在前（torch.cat((padding, hte_pe), dim=1)），HEPE 填充在后  
        #HtEPE 是 “超边→节点” 的反向编码，叠加后节点同时感知 “自己所属的超边” 和 “包含自己的超边”
        if 'HtEPE' in args.pe:
            padding=torch.zeros(z.shape[0],z.shape[1]-hte_pe.shape[1],z.shape[2],requires_grad=False,device=z.device)
            hte_pe = torch.cat((padding,hte_pe),dim=1)
            z = z + hte_pe#残差叠加 z = z + hte_pe，保证原始特征不被覆盖
        #判断是否启用层归一化（初始化时传入的 use_bn=True/False）,用来做消融实验：对比用不用归一化对效果的影响
        if self.use_bn:
            z = self.bns[0](z)#self.bns[0] 是第一层归一化（对应最开始的全连接层之后）;z=γ⋅ z−μ/√(σ²+ϵ)+β
        #完成 投影 → 加位置编码 → 归一化。    
        z = self.activation(z)#对归一化后的节点特征执行激活函数（ELU），引入非线性表达能力；ELU(x)={x,​x>0 α(e^x−1),x≤0​
        #p=self.dropout：dropout 概率（初始化时默认 0.0，可通过参数调整），比如 p=0.2 表示随机丢弃 20% 的神经元；training=self.training：核心开关—— 仅在训练阶段（model.train()）执行 dropout，测试阶段（model.eval()）不执行，保证测试结果稳定
        z = F.dropout(z, p=self.dropout, training=self.training)#对激活后的特征执行 Dropout 正则化，防止模型过拟合
        #投影→编码→归一化→激活→dropout
        #存入「输入层（投影 + 编码 + 归一化 + 激活 + dropout）」的输出；shape：[1, N, D_hid]
        layer_.append(z)#将输入层处理后的节点特征存入 layer_ 列表，作为后续残差连接 / JK 连接的基础
        #遍历模型中所有的 NodeFormerConv 层（超图注意力层），按顺序执行每一层的前向计算
        #self.convs：模型初始化时创建的 ModuleList，包含 num_layers 个 NodeFormerConv 层（HyperGT 的核心注意力层）
        #enumerate(self.convs)：同时返回层索引 i 和层实例 conv；
        #Z(ℓ)=Attn(Z(ℓ−1),H)（ℓ 从 1 到 num_layers）
        for i, conv in enumerate(self.convs):
            ##无论是否计算边损失，输出的 z 始终保持 [1, N, D_hid] 
            if self.use_edge_loss:#判断是否开启「边损失（结构正则化损失）」计算（初始化时 use_edge_loss=True 为默认）； Ls​=−∑(u,v)∈E​ *logA~uv​惩罚注意力权重偏离真实超边结构
                ##调用当前 NodeFormerConv 层的前向方法，传入：args：配置参数、z：上一层输出特征、adjs：超边索引、tau：注意力温度系数
                #返回值z：当前层注意力计算后的节点特征（shape [1, N, D_hid]）;link_loss：当前层的边损失值（标量 tensor），末尾多余的逗号是兼容写法（无实际影响）
                z, link_loss, = conv(args, z, adjs, tau)
                #将当前层的边损失存入 link_loss_ 列表，最终列表长度 = num_layers;后续总损失计算：分类损失 + λ × 所有层边损失的均值（λ 是损失权重，训练时指定）
                link_loss_.append(link_loss)
            else:
                #仅调用 NodeFormerConv 前向方法，只返回节点特征 z，不计算 / 返回边损失
                z = conv(args, z, adjs, tau)
            
            #判断是否启用残差连接，并将当前层输出与上一层特征做残差叠加（对应论文中的残差结构 Z(ℓ)=Attn(Z(ℓ−1))+Z(ℓ−1)）   
            if self.use_residual:
                #layer_[i]：取「上一层的输出特征」——；i 是当前卷积层的索引（从 0 开始），layer_ 列表中：layer_[0] = 输入层输出（Z(0)），layer_[1] = 第一层卷积输出（Z(1)）
                #残差叠加逻辑：z += layer_[i] 是逐元素相加，要求两者维度完全一致（均为 [1, N, D_hid]）
                z += layer_[i]
            #对残差连接后的特征执行对应卷积层的层归一化，稳定每层的特征分布    
            if self.use_bn:
                #self.bns[i+1]：取对应卷积层的 LayerNorm 层 ——self.bns 列表结构：[输入层BN, 第一层卷积BN, 第二层卷积BN, ...]，i 是当前卷积层索引（从 0 开始），因此 i+1 恰好对应当前卷积层的 BN 层：第 0 层卷积 → self.bns[1]，第 1 层卷积 → self.bns[2]
                #归一化对象：残差连接后的特征 z（shape [1, N, D_hid]），归一化后维度不变
                z = self.bns[i+1](z)
            #根据开关参数，对归一化后的特征执行激活函数（ELU），为卷积层引入非线性    
            #与输入层的激活逻辑区分开 —— 输入层固定执行激活（z = self.activation(z)），保证初始特征的非线性表达； 卷积层可选执行激活，用于消融实验（对比 “卷积层加 / 不加激活” 的效果）
            #self.activation：与输入层共用 ELU 激活函数，保证整个模型激活函数的一致性
            #维度不变：输入 z shape [1, N, D_hid] → 输出 shape 仍为 [1, N, D_hid]
            if self.use_act:
                z = self.activation(z)
            #对卷积层后处理（残差 / BN / 激活）的特征执行 Dropout 正则化，防止卷积层过拟合   
            #与输入层 Dropout 逻辑完全一致，但执行时机不同： 输入层：激活后 → dropout（处理初始特征）； 卷积层：（残差→BN→激活）后 → dropout（处理每层注意力输出）
            #维度不变：输入 z shape [1, N, D_hid] → 输出 shape 仍为 [1, N, D_hid]
            z = F.dropout(z, p=self.dropout, training=self.training)
            #将当前卷积层完整处理后的节点特征存入 layer_ 列表，更新多层特征缓存
            #存储时机：卷积层全流程（注意力计算→残差→BN→激活→dropout）完成后，保证存入的是 “最终稳定的层输出”
            layer_.append(z)
        #判断是否启用 JK 连接（Jump Knowledge Connection），并拼接所有层的特征作为最终特征
        if self.use_jk: # use jk connection for each layer：明确该分支的作用是 “对每层特征做 JK 连接”
            #按特征维度拼接所有层的输出——layer_：包含 num_layers + 1 个元素（输入层 + 所有卷积层输出），每个元素 shape [1, N, D_hid]； dim=-1：按最后一维（特征维度）拼接，拼接后 shape 变化：单元素：[1, N, D_hid] → 拼接后：[1, N, D_hid × (num_layers + 1)]
            z = torch.cat(layer_, dim=-1)
        #将最终特征（单层 / 多层融合）通过输出层映射到分类维度，并移除批次维度，得到最终预测结果
        #self.fcs[-1]：取全连接层列表的最后一个元素（输出分类头）——
        # 若关闭 JK 连接：self.fcs[-1] 是 nn.Linear(hidden_channels, out_channels)（输入维度 = 隐藏层维度）； 若开启 JK 连接：self.fcs[-1] 是 nn.Linear(hidden_channels × (num_layers + 1), out_channels)（输入维度 = 多层特征拼接维度）
        #输出维度固定为 out_channels（分类类别数，比如 10 分类任务则输出维度 = 10）
        #self.fcs[-1](z)：执行输出层线性变换，维度变化：关闭 JK：[1, N, D_hid] → [1, N, C]（C=out_channels）； 开启 JK：[1, N, D_hid×(L+1)] → [1, N, C]（L=num_layers）
        #.squeeze(0)：移除批次维度（第 0 维）——输入 shape [1, N, C] → 输出 shape [N, C]；节点分类任务中批次维度 B=1 是冗余的，移除后符合下游损失计算的输入要求（比如交叉熵损失要求输入为 [N, C]）
        #赋值：x_out 是模型最终的节点分类预测（未做 softmax，直接输出 logits）；输出 logits 而非概率：保留原始预测值，既可以直接计算损失，也可以后续通过 F.softmax(x_out, dim=1) 转换为概率
        x_out = self.fcs[-1](z).squeeze(0)
        #根据是否开启边损失，返回不同的结果 ——开启：返回「分类预测 + 所有层边损失列表」； 关闭：仅返回「分类预测」
        if self.use_edge_loss:
            # print('x_out:',x_out)   
            return x_out, link_loss_#返回值 1：x_out → 节点分类预测（shape [N, C]，C = 分类类别数）； 返回值 2：link_loss_ → 所有卷积层的边损失列表（长度 = num_layers，每个元素是标量 tensor）；
            #训练时计算总损失 = 分类损失（CrossEntropy） + λ × 平均边损失
        else:
            # print('x_out:',x_out)   
            #仅返回分类预测 x_out（shape [N, C]）
            return x_out
            #测试 / 推理阶段（无需计算损失），或不需要结构正则化的训练场景
            '''
                    x_out: tensor([[ 0.0343,  0.0509,  0.2399,  ...,  0.1559,  0.2357,  0.2608],
                [ 0.0113, -0.6769,  0.5730,  ...,  0.4313, -0.6512,  1.0146],
                [ 0.0248, -0.0410,  0.4525,  ..., -0.5832, -0.0620,  0.9720],
                ...,
                [-0.3862, -0.1118, -0.1130,  ..., -0.3614,  0.4227,  0.7052],
                [ 0.7517,  0.4369,  0.9890,  ..., -0.3798,  0.5646, -0.1825],
                [ 0.6285, -0.2596,  0.1966,  ...,  0.4034,  0.0928,  0.5798]],
            device='cuda:0', grad_fn=<SqueezeBackward1>)
            '''
            
