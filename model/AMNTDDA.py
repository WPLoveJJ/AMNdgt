import dgl.nn.pytorch
import torch
import torch.nn as nn
from . import gt_net_drug, gt_net_disease, hypergt_drug, hypergt_dis

device = torch.device('cuda')


class AMNTDDA(nn.Module):
    def __init__(self, args):
        super(AMNTDDA, self).__init__()
        self.args = args
        self.drug_linear = nn.Linear(300, args.hgt_in_dim)
        self.disease_linear = nn.Linear(64, args.hgt_in_dim)
        self.protein_linear = nn.Linear(320, args.hgt_in_dim)
        # self.gt_drug = gt_net_drug.GraphTransformer(device, args.gt_layer, args.drug_number, args.gt_out_dim, args.gt_out_dim,
        #                                             args.gt_head, args.dropout)
        # self.gt_disease = gt_net_disease.GraphTransformer(device, args.gt_layer, args.disease_number, args.gt_out_dim,
        #                                             args.gt_out_dim, args.gt_head, args.dropout)
        self.hgt_drug = hypergt_drug.HyperGT(
            args.dr_num_binodes,
            args.drug_number,  # 使用原始节点数（不包含超边节点）
            args.drug_feat_dim,
            300,
            args.gt_out_dim,  # 输出维度应该是gt_out_dim
            args.dr_num_hyperedges,
            num_layers=args.gt_layer,
            dropout=args.dropout,
            num_heads=args.num_heads,
            use_bn=args.use_bn,
            nb_random_features=args.M,
            use_gumbel=args.use_gumbel,
            use_residual=args.use_residual,
            use_act=args.use_act,
            use_jk=args.use_jk,
            nb_gumbel_sample=args.K,
            rb_order=args.rb_order,
            rb_trans=args.rb_trans

        )

        # Disease HyperGT model
        self.hgt_dis = hypergt_dis.HyperGT(
            args.di_num_binodes,
            args.disease_number,  # 使用原始节点数（不包含超边节点）
            args.dis_feat_dim,
            64,
            args.gt_out_dim,  # 输出维度应该是gt_out_dim
            args.di_num_hyperedges,
            num_layers=args.gt_layer,
            dropout=args.dropout,
            num_heads=args.num_heads,
            use_bn=args.use_bn,
            nb_random_features=args.M,
            use_gumbel=args.use_gumbel,
            use_residual=args.use_residual,
            use_act=args.use_act,
            use_jk=args.use_jk,
            nb_gumbel_sample=args.K,
            rb_order=args.rb_order,
            rb_trans=args.rb_trans

        )
        # self.gt_drug = HyperGT(
        #     n=data['adjs_dis'][0].max().item() + 1,     # 对应 self.n 二分图的节点数量
        #     num_nodes=args.drug_number,      # 节点数
        #     d =args.drug_number,        # 超边数 (KNN建图，M=N)
        #     in_channels=300,                 # 输入维度 (药物特征维度)
        #     hidden_channels=args.gt_out_dim, # 输出/隐藏层维度 (200)
        #     out_channels=args.gt_out_dim,    # 分类头输出维度 (这里保持一致)
        #     num_layers=args.gt_layer,        # 层数
        #     num_heads=args.gt_head,          # 头数
        #     dropout=args.amdgt_dropout,      # Dropout
        #     use_edge_loss=True,             # 关闭边损失 (因为不需要 adjs)
        #     use_bn=True,                    #批归一化
        #     use_residual=True,              #残差连接
        #     use_jk=False,                    # 通常不开启 JK
        #     use_act=True                     # 使用激活函数
        # )

        self.hgt_dgl = dgl.nn.pytorch.conv.HGTConv(args.hgt_in_dim, int(args.hgt_in_dim/args.hgt_head), args.hgt_head, 3, 3, args.dropout)
        self.hgt_dgl_last = dgl.nn.pytorch.conv.HGTConv(args.hgt_in_dim, args.hgt_head_dim, args.hgt_head, 3, 3, args.dropout)
        self.hgt = nn.ModuleList()
        for l in range(args.hgt_layer-1):
            self.hgt.append(self.hgt_dgl)
        self.hgt.append(self.hgt_dgl_last)

        encoder_layer = nn.TransformerEncoderLayer(d_model=args.gt_out_dim, nhead=args.tr_head)
        self.drug_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)
        self.disease_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)

        self.drug_tr = nn.Transformer(d_model=args.gt_out_dim, nhead=args.tr_head, num_encoder_layers=3, num_decoder_layers=3, batch_first=True)
        self.disease_tr = nn.Transformer(d_model=args.gt_out_dim, nhead=args.tr_head, num_encoder_layers=3, num_decoder_layers=3, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(args.gt_out_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )


    def forward(self, H_drug,  H_dis, drdipr_graph, drug_feature, disease_feature, protein_feature, sample,data):
        # Debug: 检查输入维度
        # print(f'drug_feature shape: {drug_feature.shape}')
        # print(f'disease_feature shape: {disease_feature.shape}')
        # print(f'protein_feature shape: {protein_feature.shape}')
        # HyperGT expects features with hyperedge nodes
        dr_out= self.hgt_drug(
                     args= self.args,
                       x = data['drug_feature_with_he'],  # [1326, 300] 包含超边
                     adjs = data['adjs_drug'],
                     H   = H_drug,
                     tau  = 1.0
                    )
        if isinstance(dr_out, tuple):
            dr_out, dr_edge_loss = dr_out
        else:
            dr_out, dr_edge_loss = dr_out, None
        # print('dr_out:',dr_out,dr_out.shape)    
        dr_sim = dr_out[:self.args.drug_number, :]    
        # # 只保留原始节点特征，去除超边节点特征
        # dr_sim = dr_sim[:self.args.drug_number, :]

        di_out = self.hgt_dis(
                     args= self.args,
                       x = data['dis_feature_with_he'],  # [818, 300] 包含超边
                     adjs = data['adjs_dis'],
                     H   = H_dis,
                     tau  = 1.0
                    )
        if isinstance(di_out, tuple):
            di_out, di_edge_loss = di_out
        else:
            di_out, di_edge_loss = di_out, None
        di_sim = di_out[:self.args.disease_number, :]              
        # # 只保留原始节点特征，去除超边节点特征
        # di_sim = di_sim[:self.args.disease_number, :]

        # # 使用原始节点特征（不含超边）进行 HGT
        # drug_feature_orig = drug_feature[:self.args.drug_number, :]
        # disease_feature_orig = disease_feature[:self.args.disease_number, :]

        # drug_feature = self.drug_linear(drug_feature_orig)
        # print('drug_feature:',drug_feature.shape)
        drug_feature = self.drug_linear(drug_feature)
        disease_feature = self.disease_linear(disease_feature)
        protein_feature = self.protein_linear(protein_feature)

        feature_dict = {
            'drug': drug_feature,
            'disease': disease_feature,
            'protein': protein_feature
        }

        drdipr_graph.ndata['h'] = feature_dict
        g = dgl.to_homogeneous(drdipr_graph, ndata='h')
        feature = torch.cat((drug_feature, disease_feature, protein_feature), dim=0)

        for layer in self.hgt:
            hgt_out = layer(g, feature, g.ndata['_TYPE'], g.edata['_TYPE'], presorted=True)
            feature = hgt_out

        # 只保留原始节点特征
        dr_hgt = hgt_out[:self.args.drug_number, :]
        di_hgt = hgt_out[self.args.drug_number:self.args.drug_number+self.args.disease_number, :]

        dr = torch.stack((dr_sim, dr_hgt), dim=1)
        di = torch.stack((di_sim, di_hgt), dim=1)

        dr = self.drug_trans(dr)
        di = self.disease_trans(di)

        dr = dr.view(self.args.drug_number, 2 * self.args.gt_out_dim)
        di = di.view(self.args.disease_number, 2 * self.args.gt_out_dim)

        drdi_embedding = torch.mul(dr[sample[:, 0]], di[sample[:, 1]])

        output = self.mlp(drdi_embedding)

        return dr, output



