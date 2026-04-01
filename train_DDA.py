import timeit
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as fn
from data_preprocess import *
from model.AMNTDDA import AMNTDDA
from metric import *
from utils import *
from torch_geometric.utils import remove_self_loops
# 🔥 允许使用所有 4 块 GPU（关键！）
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
torch.cuda.empty_cache()
device = torch.device('cuda')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=10, help='k-fold cross validation')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--neighbor', type=int, default=20, help='neighbor')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    parser.add_argument('--dataset', default='C-dataset', help='dataset')
    parser.add_argument('--dropout', default='0.2', type=float, help='dropout')
    parser.add_argument('--gt_layer', default='2', type=int, help='graph transformer layer')
    parser.add_argument('--gt_head', default='2', type=int, help='graph transformer head')
    parser.add_argument('--gt_out_dim', default='200', type=int, help='graph transformer output dimension')
    parser.add_argument('--hgt_layer', default='2', type=int, help='heterogeneous graph transformer layer')
    parser.add_argument('--hgt_head', default='8', type=int, help='heterogeneous graph transformer head')
    parser.add_argument('--hgt_in_dim', default='64', type=int, help='heterogeneous graph transformer input dimension')
    parser.add_argument('--hgt_head_dim', default='25', type=int, help='heterogeneous graph transformer head dimension')
    parser.add_argument('--hgt_out_dim', default='200', type=int, help='heterogeneous graph transformer output dimension')
    parser.add_argument('--tr_layer', default='2', type=int, help='transformer layer')
    parser.add_argument('--tr_head', default='4', type=int, help='transformer head')
    
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--M', type=int,
                        default=30, help='number of random features')
    parser.add_argument('--use_gumbel', action='store_true', help='use gumbel softmax for message passing')
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--use_act', action='store_true', help='use non-linearity for each layer')
    parser.add_argument('--use_jk', action='store_true', help='concat the layer-wise results in the final layer')
    parser.add_argument('--K', type=int, default=10, help='num of samples for gumbel softmax sampling')
    parser.add_argument('--tau', type=float, default=0.25, help='temperature for gumbel softmax')
    parser.add_argument('--lamda', type=float, default=0.1, help='weight for edge reg loss')
    parser.add_argument('--rb_order', type=int, default=0, help='order for relational bias, 0 for not use')
    parser.add_argument('--rb_trans', type=str, default='sigmoid', choices=['sigmoid', 'identity'],
                        help='non-linearity for relational bias')
    parser.add_argument('--pe', type=str, default=['HEPE','HtEPE'], help='use positional encoding')

    # ===================== 药物transformer的 4 个参数 =====================
    parser.add_argument('--dr_num_binodes', type=int, default=663*2, help='二分图总节点数 = drug节点数 + drug超边数')
    parser.add_argument('--drug_number', type=int, default=663, help='药物节点数')
    parser.add_argument('--drug_feat_dim', type=int, default=300, help='药物特征维度')
    parser.add_argument('--dr_num_hyperedges', type=int, default=663, help='超边数量')
    # ===================== 疾病transformer的 4 个参数 =====================
    parser.add_argument('--di_num_binodes', type=int, default=409*2, help='二分图总节点数 = drug数 + disease数')
    parser.add_argument('--dis_number', type=int, default=409, help='药物节点数')
    parser.add_argument('--dis_feat_dim', type=int, default=300, help='药物特征维度')
    parser.add_argument('--di_num_hyperedges', type=int, default=409, help='超边数量')

    args = parser.parse_args()
    args.data_dir = 'data/' + args.dataset + '/'
    args.result_dir = 'Result/' + args.dataset + '/AMNTDDA/'

    data = get_data(args)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    data = data_processing(data, args)
    data = k_fold(data, args)


    '''生成超图'''
    H_drug = construct_H_with_KNN(data['drs'], K_neigs=[10], is_probH=False, m_prob=1, edge_type='euclid')
    H_dis = construct_H_with_KNN(data['dis'], K_neigs=[10], is_probH=False, m_prob=1, edge_type='euclid')
    # H_drug = torch.from_numpy(H_drug).to_sparse_coo()
    # H_dis = torch.from_numpy(H_dis).to_sparse_coo()
    # 2. 直接转稀疏张量（你原来的代码，最安全）
    H_drug = torch.from_numpy(H_drug).to_sparse_coo().coalesce().to(device).float()
    H_dis = torch.from_numpy(H_dis).to_sparse_coo().coalesce().to(device).float()
    '''生成超图二分图'''
    adjs_drug = convert_H_to_HyperGT_adjs(H_drug)
    adjs_dis = convert_H_to_HyperGT_adjs(H_dis)
    # print('adjs_dis:',adjs_dis)
    data['adjs_drug'] = adjs_drug
    data['adjs_dis'] = adjs_dis


    # print(data['adjs_dis'])
    
    # print('adjs_drug:',data['adjs_drug'])
    args.dr_num_binodes   = data['adjs_drug'][0].max().item() + 1  # n
    args.drug_number   = data['drugfeature'].shape[0]                # num_nodes
    args.drug_feat_dim = data['drugfeature'].shape[1]                # d
    args.dr_num_hyperedges = H_drug.shape[1]                      # e

    args.di_num_binodes   = data['adjs_dis'][0].max().item() + 1  # n
    args.disease_number   = data['diseasefeature'].shape[0]         # num_nodes
    args.dis_feat_dim = data['diseasefeature'].shape[1]             # d
    args.di_num_hyperedges = H_dis.shape[1]                      # e

    # print(args.dr_num_binodes,args.drug_number,args.drug_feat_dim,args.dr_num_hyperedges)
    # print(args.di_num_binodes,args.disease_number,args.dis_feat_dim,args.di_num_hyperedges)
    # print('H_drug:', H_drug)
    # print('H_dis:', H_dis)
    # # 👇 改成这个 👇 【完整正确版】
    # H_drug = H_drug.to(device).float()
    # H_dis  = H_dis.to(device).float()           # 超图矩阵 → cuda
    drug_feature = torch.FloatTensor(data['drugfeature']).to(device)
    dr_feat = torch.FloatTensor(data['drugfeature'])
    disease_feature = torch.FloatTensor(data['diseasefeature']).to(device)
    di_feat = torch.FloatTensor(data['diseasefeature'])
    protein_feature = torch.FloatTensor(data['proteinfeature']).to(device)
    # print('disease_feature:',disease_feature.shape)
    # 为drug和disease添加超边节点特征（零特征）
    num_drug_hyperedges = H_drug.shape[1]
    num_dis_hyperedges = H_dis.shape[1]
    drug_he_feat = torch.zeros(num_drug_hyperedges, drug_feature.shape[1],requires_grad=False)
    dis_he_feat = torch.zeros(num_dis_hyperedges, disease_feature.shape[1],requires_grad=False)
    drug_he_feat = torch.rand(num_drug_hyperedges, drug_feature.shape[1],requires_grad=True)
    dis_he_feat = torch.rand(num_dis_hyperedges, disease_feature.shape[1],requires_grad=True)

    drug_feature_with_he = torch.cat([dr_feat, drug_he_feat], dim=0).to(device)
    disease_feature_with_he = torch.cat([di_feat, dis_he_feat], dim=0).to(device)
    data['drug_feature_with_he'] = drug_feature_with_he
    data['dis_feature_with_he'] = disease_feature_with_he  
    # print('drug_feature_with_he:',drug_feature_with_he.shape) 
    # print('disease_feature_with_he:',torch.FloatTensor(data['diseasefeature']).shape)
    
   
    # # 保存原始节点数（不含超边）
    # args.drug_number_original = data['drug_number']
    # args.disease_number_original = data['disease_number']

    # # 更新args中的节点数以包含超边节点（用于HyperGT）
    # args.drug_number = drug_feature.shape[0]
    # args.disease_number = disease_feature.shape[0]
    all_sample = torch.tensor(data['all_drdi']).long()

    start = timeit.default_timer()

    cross_entropy = nn.CrossEntropyLoss()

    Metric = ('Epoch\t\tTime\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')
    AUCs, AUPRs = [], []

    print('Dataset:', args.dataset)

    for i in range(args.k_fold):

        print('fold:', i)
        print(Metric)

        model = AMNTDDA(args)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

        best_auc, best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = 0, 0, 0, 0, 0, 0, 0
        X_train = torch.LongTensor(data['X_train'][i]).to(device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device)
        X_test = torch.LongTensor(data['X_test'][i]).to(device)
        Y_test = data['Y_test'][i].flatten()

        drdipr_graph, data = dgl_heterograph(data, data['X_train'][i], args)
        drdipr_graph = drdipr_graph.to(device)
        best_val_loss = float("inf")
        counter = 0
        patience = 50         # 早停容忍轮数，可自定义
        best_model_path = "best_model.pth"
        for epoch in range(args.epochs):
            model.train()
            _, train_score = model(H_drug,  H_dis, drdipr_graph, drug_feature, disease_feature, protein_feature, X_train,data)
            train_loss = cross_entropy(train_score, torch.flatten(Y_train))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                dr_representation, test_score = model(H_drug,  H_dis, drdipr_graph, drug_feature, disease_feature, protein_feature, X_test,data)
                val_loss = cross_entropy(
                test_score,
                torch.flatten(torch.from_numpy(Y_test).to(test_score.device).long())
                ).item()


            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss
                }, best_model_path)
                print(f"Saved best model at epoch {epoch}, Val Loss: {best_val_loss:.4f}")
                counter = 0  # 重置计数器
            else:
                counter += 1
                print(f"No improvement. Counter: {counter}/{patience}") 
             # --- 中断训练 ---
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}!")
                break        
            test_prob = fn.softmax(test_score, dim=-1)
            test_score = torch.argmax(test_score, dim=-1)

            test_prob = test_prob[:, 1]
            test_prob = test_prob.cpu().numpy()

            test_score = test_score.cpu().numpy()

            AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_score, test_prob)

            end = timeit.default_timer()
            time = end - start
            show = [epoch + 1, round(time, 2), round(AUC, 5), round(AUPR, 5), round(accuracy, 5),
                       round(precision, 5), round(recall, 5), round(f1, 5), round(mcc, 5)]
            print('\t\t'.join(map(str, show)))
            if AUC > best_auc:
                best_epoch = epoch + 1
                best_auc = AUC
                best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = AUPR, accuracy, precision, recall, f1, mcc
                print('AUC improved at epoch ', best_epoch, ';\tbest_auc:', best_auc)

        AUCs.append(best_auc)
        AUPRs.append(best_aupr)

    print('AUC:', AUCs)
    AUC_mean = np.mean(AUCs)
    AUC_std = np.std(AUCs)
    print('Mean AUC:', AUC_mean, '(', AUC_std, ')')

    print('AUPR:', AUPRs)
    AUPR_mean = np.mean(AUPRs)
    AUPR_std = np.std(AUPRs)
    print('Mean AUPR:', AUPR_mean, '(', AUPR_std, ')')



