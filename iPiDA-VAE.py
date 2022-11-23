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
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
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
        labels = torch.Tensor(R_R).to(device)
        # 初始化模型，优化器以及损失函数
        model = VAEModel(input_dim=R_R.shape[1], hidden_dim1=48, hidden_dim2=16, dropout=0.6).to(device)

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
            pre, RC_features = model(R_R)  # 得到点积预测值pre和RNA与疾病的特征表示RD_features
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
        labels = torch.Tensor(D_D).to(device)
        # 初始化模型，优化器以及损失函数
        model = VAEModel(input_dim=D_D.shape[1], hidden_dim1=48, hidden_dim2=16, dropout=0.6).to(device)
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
            pre, DC_features = model(D_D)  # 得到点积预测值pre和RNA与疾病的特征表示RD_features
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
        labels = torch.Tensor(GIP_R).to(device)
        # 初始化模型，优化器以及损失函数
        model = VAEModel(input_dim=GIP_R.shape[1], hidden_dim1=48, hidden_dim2=16, dropout=0.6).to(device)
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
            pre, RG_features = model(GIP_R)  # 得到点积预测值pre和RNA与疾病的特征表示RD_features
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
        labels = torch.Tensor(GIP_D).to(device)
        # 初始化模型，优化器以及损失函数
        model = VAEModel(input_dim=GIP_D.shape[1], hidden_dim1=48, hidden_dim2=16, dropout=0.6).to(device)
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
            pre, DG_features = model(GIP_D)  # 得到点积预测值pre和RNA与疾病的特征表示RD_features
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

        '''
                        ***************************** 使用以上特征进行训练 ******************************
                       '''
        # 将两种特征进行拼接
        # 以下为训练的特征和标签值
        train_feature_bal, train_label_bal = get_two_feature(train_samples,
                                                             RC_features, DC_features,
                                                             RG_features, DG_features)
        # 以下为测试集的特征和标签值
        test_feature, test_label = get_two_feature(test_samples,
                                                   RC_features, DC_features,
                                                   RG_features, DG_features)
        # 初始化神经网络模型
        MLP_model = MLPClf(input_dim=train_feature_bal.shape[1],
                           hidden_dim1=48, hidden_dim2=16, output_dim=1, epoch=200)
        MLP_model.fit(train_feature_bal, train_label_bal)
        y_pred = MLP_model.predict_proba(test_feature)
        auc = roc_auc_score(test_label, y_pred)
        # 得到每一次的测试集的标签值和预测值，然后进行数据可视化（画图，画表格）
        np.save("./result/VAE/VAE2_test_label_{0}".format(iter), test_label)
        np.save("./result/VAE/VAE2_y_pred_{0}".format(iter), y_pred)
        print("概率值auc：", auc)
        ap = average_precision_score(test_label, y_pred)
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
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        print(classification_report(test_label, y_pred.astype('int')))
        AUC.append(auc)
        AUPR.append(ap)


print("random_auc:", np.array(AUC))
print("random_auc_mean:", sum(np.array(AUC)) / 5)


print("random_aupr:", np.array(AUPR))
print("co_aupr_mean:", sum(np.array(AUPR)) / 5)
# plt.plot(fpr, tpr, lw=lw, label='Average ROC curve(AUC ={:.4f})'.format(sum(np.array(AUC)) / 5))
# plt.legend(loc="lower right")
# plt.show()
