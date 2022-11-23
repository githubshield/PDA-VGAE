import random
import time

import dgl
import matplotlib.pyplot as plt
import numpy as np
import torch
from dgl.nn import EdgeWeightNorm
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
from sklearn.neural_network import MLPClassifier

# 导入本地工具模块
from tools import *
from model import *


# 输出运算资源请况
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
# device =  torch.device('cpu')
print("使用第{0}块GPU进行训练".format(device))

'''
1.准备数据集
需要准备的数据集有：ncRNA与疾病的关联数据集，ncRNA与疾病特征数据集
ncRNA与疾病的关联数据集可以直接获得
'''
# 获得ncRNA的序列相似性矩阵R_R
# R_R = read_txt("./piRNA/PiKmerFeature.csv")
# # 得到peason相似矩阵
# R_R = pearson_similarity(R_R)
# np.save("./tmp/R_R.npy",R_R)
R_R = np.load("./piRNA/R_R.npy")  # 得到piRNA的序列相似性矩阵
N_R = R_R.shape[0]
print('有{0}种RNA'.format(N_R))
# 获得疾病与疾病的关联数据集D_D
D_D = read_txt("./piRNA/DiseaseSemanticFeature.csv")
N_d = D_D.shape[0]
print('有{0}种疾病'.format(N_d))
# 获得ncRNA与疾病的关联数据集R_D
R_D = read_txt("./piRNA/pi_association_matrix.csv")
print("有{0}个关联".format(sum(sum(R_D))))

# 读取数据集样本，便于做交叉验证，得到一个三元组的数据集[R,D,E],R:ncRNA的索引，D:疾病的索引，E:关联边的取值(0,1)
samples = get_all_the_samples(R_D)
print("有{0}个样本，其中负样本有{1}个，正样本有{2}个"
      .format(len(samples), samples[samples[:, 2] == 0].shape[0], samples[samples[:, 2] == 1].shape[0]))
R_R = Knormalized(R_R)
D_D = Knormalized(D_D)
# 进行5折交叉验证
kf = KFold(n_splits=5, shuffle=True)
shiyan_samples = samples
iter = 0
AUC = []
AUPR = []

lw = 2
plt.figure(figsize=(8, 5))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')

for train_index, test_index in kf.split(shiyan_samples):
    if iter < 6:
        iter = iter + 1
        # 这里根据索引得到训练样本集train_samples和测试样本test_samples ndarray:[[260,19,1],……]
        # train_samples = shiyan_samples[train_index, :]  # 训练样本
        # np.save("./piRNA/train_samples_{0}.npy".format(iter),train_samples)
        train_samples = np.load("./piRNA/train_samples_{0}.npy".format(iter))  # 读取每一次的数据
        # test_samples = shiyan_samples[test_index, :]  # 测试样本
        # np.save("./piRNA/test_samples_{0}.npy".format(iter), test_samples)
        test_samples = np.load("./piRNA/test_samples_{0}.npy".format(iter))  # 读取每一次的数据
        print("训练样本共有{0}对，其中正样本有{1}对，未标记样本有{2}对"
              .format(train_samples.shape[0],
                      train_samples[train_samples[:, 2] == 1].shape[0],
                      train_samples[train_samples[:, 2] == 0].shape[0]))
        print("测试样本共有{0}对，其中正样本有{1}对，未标记样本有{2}对 \n"
              .format(test_samples.shape[0],
                      test_samples[test_samples[:, 2] == 1].shape[0],
                      test_samples[test_samples[:, 2] == 0].shape[0]))
        '''
        *****************************对数据集进行处理***************************************
        '''
        # 去掉原关联矩阵中测试集的关联边
        R_D_new = R_D.copy()  # 这里将原数据集进行复制，防止改变原关联矩阵
        R_D_new[test_samples[:, 0], test_samples[:, 1]] = 0  # 原关联矩阵中的测试集部分的关联置为0
        # 未进行标准化的GIP相似矩阵
        # GIP_R = GIP_kernel(R_D_new)  # 得到RNA的GIP相似性矩阵
        # np.save("./piRNA/GIP_R_{0}.npy".format(iter), GIP_R)
        GIP_R = np.load("./piRNA/GIP_R_{0}.npy".format(iter))  # 读取每一次的数据
        # GIP_D = GIP_kernel(R_D_new.T)  # 得到疾病的GIP相似性矩阵
        # np.save("./piRNA/GIP_D_{0}.npy".format(iter), GIP_D)
        GIP_D = np.load("./piRNA/GIP_D_{0}.npy".format(iter))  # 读取每一次的数据
        # 对数据进行正则化处理
        GIP_R = Knormalized(GIP_R)
        GIP_D = Knormalized(GIP_D)

        '''
        ***************************** 使用RNA的序列特征构建图提取RNA的序列特征 ******************************
        '''
        R_u, R_v, R_w = get_edges(R_R)  # 获得RNA与RNA系列相似网络的节点和边
        # 使用RNA的相似网络，以RDA关联网络作为特征，提取RNA节点的特征表示
        g_R = dgl.graph((torch.tensor(R_u), torch.tensor(R_v))).to(device)  # 构建网络图
        # 将边的权重进行归一化处理
        # norm = EdgeWeightNorm(norm='right')
        edge_weight = torch.tensor(R_w).to(device)
        # norm_edge_weight = norm(g_R, edge_weight)
        g_R.edata["weights"] = torch.tensor(edge_weight).to(device)  # 网络图的权重（关联边的特征）
        '''
        零度数节点将导致无效的输出值。这是因为不会向这些节点传递任何消息，聚合函数将应用于空输入。
        避免这种情况的常见做法是，如果图中的每个节点是齐次的，则为它添加一个自环，这可以通过以下方式实现：
        '''
        g_R = dgl.add_self_loop(g_R)
        print(g_R)
        features = torch.Tensor(R_D_new).to(device)  # 这里我们以RDA关联矩阵作为特征输入
        labels = torch.Tensor(R_R).to(device)
        # 初始化模型，优化器以及损失函数
        model = VGAEModel(input_dim=features.shape[1], hidden_dim1=48, hidden_dim2=16, dropout=0.6).to(device)
        # model = GCN(input_dim=features.shape[1], hidden_dim1=48, out_put=16, feat_drop=0.6).to(device)
        # model = GAT(num_layers=1, in_dim=features.shape[1], num_hidden=48, heads=([8] * 1) + [1],
        #             out_put=16, feat_drop=0.6, attn_drop=0.6, negative_slope=0.2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
        # criterion = nn.MSELoss(reduction='sum')
        weight_tensor, norm = compute_loss_para(labels)  # compute loss parameters
        # 开始进行训练，得到RNA属性特征表示
        for epoch in range(200):
            # 进行训练，得到最后的特征表示
            optimizer.zero_grad()  # 每一次更新将梯度归零
            pre, RC_features = model(g_R, features)  # 得到点积预测值pre和RNA与疾病的特征表示RD_features
            # 计算模型的损失值
            loss = norm * F.binary_cross_entropy_with_logits(pre.view(-1), labels.view(-1), weight=weight_tensor)
            # loss = F.mse_loss(pre.view(-1), labels.view(-1))
            # 计算K1散度
            kl_divergence = 0.5 / pre.size(0) * (
                    1 + 2 * model.log_std - model.mean ** 2 - torch.exp(model.log_std) ** 2) \
                .sum(1).mean()
            loss -= kl_divergence

            # loss = criterion(pre, labels)
            # 计算训练时的准确率
            acc = accuracy(pre, labels)
            # 方向传播loss值
            loss.backward()
            # 进行梯度优化
            optimizer.step()
            # 输出模型的迭代次数，损失函数值和准确率
            # print("RNA的序列特征提取：Epoch {:03d} | Loss {:.4f} | Acc {:.4f} "
            #       .format(epoch + 1, loss.item(), acc))

        '''
         ***************************** 使用疾病的语义特征构建图提取疾病的语义特征 ******************************
         '''
        D_u, D_v, D_w = get_edges(D_D)  # 获得疾病与疾病的语义相似网络的节点和边
        # 使用RNA的相似网络，以RDA关联网络作为特征，提取RNA节点的特征表示
        g_D = dgl.graph((torch.tensor(D_u), torch.tensor(D_v))).to(device)  # 构建网络图
        # 将边的权重进行归一化处理
        # norm = EdgeWeightNorm(norm='right')
        edge_weight = torch.tensor(D_w).to(device)
        # norm_edge_weight = norm(g_R, edge_weight)
        g_D.edata["weights"] = torch.tensor(edge_weight).to(device)  # 网络图的权重（关联边的特征）
        '''
        零度数节点将导致无效的输出值。这是因为不会向这些节点传递任何消息，聚合函数将应用于空输入。
        避免这种情况的常见做法是，如果图中的每个节点是齐次的，则为它添加一个自环，这可以通过以下方式实现：
        '''
        g_D = dgl.add_self_loop(g_D)
        print(g_D)
        features = torch.Tensor(R_D_new.T).to(device)  # 这里我们以RDA关联矩阵作为特征输入
        labels = torch.Tensor(D_D).to(device)
        # 初始化模型，优化器以及损失函数
        model = VGAEModel(input_dim=features.shape[1], hidden_dim1=48, hidden_dim2=16, dropout=0.6).to(device)
        # model = GCN(input_dim=features.shape[1], hidden_dim1=48, out_put=16, feat_drop=0.6).to(device)
        # model = GAT(num_layers=1, in_dim=features.shape[1], num_hidden=48, heads=([8] * 1) + [1],
        #             out_put=16, feat_drop=0.6, attn_drop=0.6, negative_slope=0.2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
        # criterion = nn.MSELoss(reduction='sum')
        weight_tensor, norm = compute_loss_para(labels)  # compute loss parameters
        # weight_tensor, norm = 25, 0.5  # compute loss parameters
        # 开始进行训练，得到RNA属性特征表示
        for epoch in range(200):
            # 进行训练，得到最后的特征表示
            optimizer.zero_grad()  # 每一次更新将梯度归零
            pre, DC_features = model(g_D, features)  # 得到点积预测值pre和RNA与疾病的特征表示RD_features
            # 计算模型的损失值
            loss = norm * F.binary_cross_entropy_with_logits(pre.view(-1), labels.view(-1), weight=weight_tensor)
            # loss = F.mse_loss(pre.view(-1), labels.view(-1))
            # 计算K1散度
            kl_divergence = 0.5 / pre.size(0) * (
                    1 + 2 * model.log_std - model.mean ** 2 - torch.exp(model.log_std) ** 2) \
                .sum(1).mean()
            loss -= kl_divergence

            # loss = criterion(pre, labels)
            # 计算训练时的准确率
            acc = accuracy(pre, labels)
            # 方向传播loss值
            loss.backward()
            # 进行梯度优化
            optimizer.step()
            # 输出模型的迭代次数，损失函数值和准确率
            # print("疾病的语义特征提取：Epoch {:03d} | Loss {:.4f} | Acc {:.4f} "
            #       .format(epoch + 1, loss.item(), acc))
        '''
        ***************************** 使用RNA的GIP特征构建图提取RNA的GIP特征 ******************************
        '''
        RG_u, RG_v, RG_w = get_edges(GIP_R)  # 获得疾病与疾病的语义相似网络的节点和边
        # 使用RNA的相似网络，以RDA关联网络作为特征，提取RNA节点的特征表示
        g_RG = dgl.graph((torch.tensor(RG_u), torch.tensor(RG_v))).to(device)  # 构建网络图
        # 将边的权重进行归一化处理
        # norm = EdgeWeightNorm(norm='right')
        edge_weight = torch.tensor(RG_w).to(device)
        # norm_edge_weight = norm(g_R, edge_weight)
        g_RG.edata["weights"] = torch.tensor(edge_weight).to(device)  # 网络图的权重（关联边的特征）
        '''
        零度数节点将导致无效的输出值。这是因为不会向这些节点传递任何消息，聚合函数将应用于空输入。
        避免这种情况的常见做法是，如果图中的每个节点是齐次的，则为它添加一个自环，这可以通过以下方式实现：
        '''
        g_RG = dgl.add_self_loop(g_RG)
        print(g_RG)
        features = torch.Tensor(R_D_new).to(device)  # 这里我们以RDA关联矩阵作为特征输入
        labels = torch.Tensor(GIP_R).to(device)
        # 初始化模型，优化器以及损失函数
        model = VGAEModel(input_dim=features.shape[1], hidden_dim1=48, hidden_dim2=16, dropout=0.6).to(device)
        # model = GCN(input_dim=features.shape[1], hidden_dim1=48, out_put=16, feat_drop=0.6).to(device)
        # model = GAT(num_layers=1, in_dim=features.shape[1], num_hidden=48, heads=([8] * 1) + [1],
        #             out_put=16, feat_drop=0.6, attn_drop=0.6, negative_slope=0.2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
        # criterion = nn.MSELoss(reduction='sum')
        weight_tensor, norm = compute_loss_para(labels)  # compute loss parameters
        # 开始进行训练，得到RNA属性特征表示
        for epoch in range(200):
            # 进行训练，得到最后的特征表示
            optimizer.zero_grad()  # 每一次更新将梯度归零
            pre, RG_features = model(g_RG, features)  # 得到点积预测值pre和RNA与疾病的特征表示RD_features
            # 计算模型的损失值
            loss = norm * F.binary_cross_entropy_with_logits(pre.view(-1), labels.view(-1), weight=weight_tensor)
            # loss = F.mse_loss(pre.view(-1), labels.view(-1))
            # 计算K1散度
            kl_divergence = 0.5 / pre.size(0) * (
                    1 + 2 * model.log_std - model.mean ** 2 - torch.exp(model.log_std) ** 2) \
                .sum(1).mean()
            loss -= kl_divergence

            # loss = criterion(pre, labels)
            # 计算训练时的准确率
            acc = accuracy(pre, labels)
            # 方向传播loss值
            loss.backward()
            # 进行梯度优化
            optimizer.step()
            # 输出模型的迭代次数，损失函数值和准确率
            # print("RNA的GIP特征提取：Epoch {:03d} | Loss {:.4f} | Acc {:.4f} "
            #       .format(epoch + 1, loss.item(), acc))

        '''
        ***************************** 使用疾病的GIP特征构建图提取疾病的GIP特征 ******************************
        '''
        DG_u, DG_v, DG_w = get_edges(GIP_D)  # 获得疾病与疾病的语义相似网络的节点和边
        # 使用RNA的相似网络，以RDA关联网络作为特征，提取RNA节点的特征表示
        g_DG = dgl.graph((torch.tensor(DG_u), torch.tensor(DG_v))).to(device)  # 构建网络图
        # 将边的权重进行归一化处理
        # norm = EdgeWeightNorm(norm='right')
        edge_weight = torch.tensor(DG_w).to(device)
        # norm_edge_weight = norm(g_R, edge_weight)
        g_DG.edata["weights"] = torch.tensor(edge_weight).to(device)  # 网络图的权重（关联边的特征）
        '''
        零度数节点将导致无效的输出值。这是因为不会向这些节点传递任何消息，聚合函数将应用于空输入。
        避免这种情况的常见做法是，如果图中的每个节点是齐次的，则为它添加一个自环，这可以通过以下方式实现：
        '''
        g_DG = dgl.add_self_loop(g_DG)
        print(g_DG)
        features = torch.Tensor(R_D_new.T).to(device)  # 这里我们以RDA关联矩阵作为特征输入
        labels = torch.Tensor(GIP_D).to(device)
        # 初始化模型，优化器以及损失函数
        model = VGAEModel(input_dim=features.shape[1], hidden_dim1=48, hidden_dim2=16, dropout=0.6).to(device)
        # model = GCN(input_dim=features.shape[1], hidden_dim1=48, out_put=16, feat_drop=0.6).to(device)
        # model = GAT(num_layers=1, in_dim=features.shape[1], num_hidden=48, heads=([8] * 1) + [1],
        #             out_put=16, feat_drop=0.6, attn_drop=0.6, negative_slope=0.2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
        # criterion = nn.MSELoss(reduction='sum')
        weight_tensor, norm = compute_loss_para(labels)  # compute loss parameters
        # 开始进行训练，得到RNA属性特征表示
        for epoch in range(200):
            # 进行训练，得到最后的特征表示
            optimizer.zero_grad()  # 每一次更新将梯度归零
            pre, DG_features = model(g_DG, features)  # 得到点积预测值pre和RNA与疾病的特征表示RD_features
            # 计算模型的损失值
            loss = norm * F.binary_cross_entropy_with_logits(pre.view(-1), labels.view(-1), weight=weight_tensor)
            # loss = F.mse_loss(pre.view(-1), labels.view(-1))
            # 计算K1散度
            kl_divergence = 0.5 / pre.size(0) * (
                    1 + 2 * model.log_std - model.mean ** 2 - torch.exp(model.log_std) ** 2) \
                .sum(1).mean()
            loss -= kl_divergence

            # loss = criterion(pre, labels)
            # 计算训练时的准确率
            acc = accuracy(pre, labels)
            # 方向传播loss值
            loss.backward()
            # 进行梯度优化
            optimizer.step()
            # 输出模型的迭代次数，损失函数值和准确率
            # print("疾病的GIP特征提取：Epoch {:03d} | Loss {:.4f} | Acc {:.4f} "
            #       .format(epoch + 1, loss.item(), acc))


        # 求属性特征的点积作为预测
        pre_adj1 = torch.sigmoid(torch.mm(RC_features, DC_features.t()))
        # 求关联特征的点积作为预测
        pre_adj2 = torch.sigmoid(torch.mm(RG_features, DG_features.t()))
        # 求二者的平均值作为预测的最终矩阵
        pre_adj = 0.5*(pre_adj1+pre_adj2)
        AA = pre_adj1.detach().cpu().numpy()


        # 得到测试集部分的预测标签
        y_pred = pre_adj[test_samples[:,0],test_samples[:,1]]
        test_label = test_samples[:,2]
        # print("test")


        auc = roc_auc_score(test_label.reshape(-1,1), y_pred.reshape(-1,1).cpu().detach().numpy())
        # 得到每一次的测试集的标签值和预测值，然后进行数据可视化（画图，画表格）
        print("概率值auc：", auc)
        ap = average_precision_score(test_label.reshape(-1,1), y_pred.reshape(-1,1).cpu().detach().numpy())
        print("概率值ap：", ap)
        # fpr, tpr, thr = roc_pr_curve(test_label, y_pred)
        # pre, rec, thr = precision_recall_curve(test_label, y_pred)
        # # plt.figure(figsize=(8, 5))
        # plt.plot(fpr, tpr, lw=lw, label='ROC curve fold-{}(AUC ={:.4f})'.format(iter, auc))  ###假正率为横坐标，真正率为纵坐标做曲线

        # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        # plt.legend(loc="lower right")
        # plt.show()
        # print("概率值auc：",auc)
        # ap = average_precision_score(test_label, y_pred)
        # y_pred[y_pre .5] = 0
        # print(classification_report(test_label, y_pred.astype('int')))
        AUC.append(auc)
        AUPR.append(ap)


print("random_auc:", np.array(AUC))
print("random_auc_mean:", sum(np.array(AUC)) / 5)


print("random_aupr:", np.array(AUPR))
print("co_aupr_mean:", sum(np.array(AUPR)) / 5)
# plt.plot(fpr, tpr, lw=lw, label='Average ROC curve(AUC ={:.4f})'.format(sum(np.array(AUC)) / 5))
# plt.legend(loc="lower right")
# plt.show()
