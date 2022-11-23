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
# R_R = Knormalized(R_R)
# D_D = Knormalized(D_D)
# 进行5折交叉验证
kf = KFold(n_splits=5, shuffle=True)
shiyan_samples = samples
iter = 0


AUC_w_VGAE = []
AUPR_w_VGAE = []


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
        # 去掉原关联矩阵中除正样本之外关联边
        R_D_new = np.zeros_like(R_D)  # 生成一个与原关联矩阵一样大的矩阵
        # 将正样本的关联赋值给新的关联矩阵
        # R_D_new[half_pos_samples[:, 0], half_pos_samples[:, 1]] = 1
        # np.save("./tmp/R_D_new_{0}.npy".format(iter),R_D_new)
        # 未进行标准化的GIP相似矩阵
        # GIP_R = GIP_kernel(R_D_new)  # 得到RNA的GIP相似性矩阵
        # # np.save("./data_scale/GIP_R_10_{0}.npy".format(iter), GIP_R)
        # # GIP_R = np.load("./tmp/GIP_R_{0}.npy".format(iter))  # 读取每一次的数据
        # GIP_D = GIP_kernel(R_D_new.T)  # 得到疾病的GIP相似性矩阵
        # np.save("./data_scale/GIP_D_10_{0}.npy".format(iter), GIP_D)

        # 未进行标准化的GIP相似矩阵
        # GIP_R = GIP_kernel(R_D_new)  # 得到RNA的GIP相似性矩阵
        # np.save("./piRNA/GIP_R_{0}.npy".format(iter), GIP_R)
        GIP_R = np.load("./piRNA/GIP_R_{0}.npy".format(iter))  # 读取每一次的数据
        # GIP_D = GIP_kernel(R_D_new.T)  # 得到疾病的GIP相似性矩阵
        # np.save("./piRNA/GIP_D_{0}.npy".format(iter), GIP_D)
        GIP_D = np.load("./piRNA/GIP_D_{0}.npy".format(iter))  # 读取每一次的数据
        # 对数据进行正则化处理
        # GIP_R = Knormalized(GIP_R)
        # GIP_D = Knormalized(GIP_D)

        '''
                                ***************************** 不使用VGAE提取特征进行训练 ******************************
        '''
        # 将两种特征进行拼接
        # 以下为训练的特征和标签值
        p_features = R_R
        d_features = D_D
        p_G_features = GIP_R
        d_G_features = GIP_D

        n = train_samples.shape[0]
        vect_len1 = p_features.shape[1]
        vect_len2 = d_features.shape[1]
        vect_len3 = p_G_features.shape[1]
        vect_len4 = d_G_features.shape[1]
        train_feature_bal = np.zeros([n, (vect_len1 + vect_len2 + vect_len3 + vect_len4)])
        train_label_bal = np.zeros([n])
        for i in tqdm(range(n), desc="get_feature"):
            train_feature_bal[i, 0:vect_len1] = p_features[int(train_samples[i, 0]), :]
            train_feature_bal[i, vect_len1:(vect_len1 + vect_len2)] = d_features[int(train_samples[i, 1]), :]
            train_feature_bal[i, (vect_len1 + vect_len2):(vect_len1 + vect_len2 + vect_len3)] = \
                p_G_features[int(train_samples[i, 0]), :]
            train_feature_bal[i, (vect_len1 + vect_len2 + vect_len3):(vect_len1 + vect_len2 + vect_len3 + vect_len4)] = \
                d_G_features[int(train_samples[i, 1]), :]
            train_label_bal[i] = int(train_samples[i, 2])
        # train_feature_bal, train_label_bal = get_two_feature(train_samples, R_R, D_D, GIP_R, GIP_D)
        # 以下为测试集的特征和标签值
        n = test_samples.shape[0]
        vect_len1 = p_features.shape[1]
        vect_len2 = d_features.shape[1]
        vect_len3 = p_G_features.shape[1]
        vect_len4 = d_G_features.shape[1]
        test_feature = np.zeros([n, (vect_len1 + vect_len2 + vect_len3 + vect_len4)])
        test_label = np.zeros([n])
        for i in tqdm(range(n), desc="get_feature"):
            test_feature[i, 0:vect_len1] = p_features[int(test_samples[i, 0]), :]
            test_feature[i, vect_len1:(vect_len1 + vect_len2)] = d_features[int(test_samples[i, 1]), :]
            test_feature[i, (vect_len1 + vect_len2):(vect_len1 + vect_len2 + vect_len3)] = \
                p_G_features[int(test_samples[i, 0]), :]
            test_feature[i, (vect_len1 + vect_len2 + vect_len3):(vect_len1 + vect_len2 + vect_len3 + vect_len4)] = \
                d_G_features[int(test_samples[i, 1]), :]
            test_label[i] = int(test_samples[i, 2])

        # test_feature, test_label = get_two_feature(test_samples,R_R, D_D,GIP_R, GIP_D)
        # 初始化神经网络模型
        MLP_model = MLPClf(input_dim=train_feature_bal.shape[1],
                           hidden_dim1=48, hidden_dim2=16, output_dim=1, epoch=200)
        MLP_model.fit(train_feature_bal, train_label_bal)
        y_pred = MLP_model.predict_proba(test_feature)
        auc = roc_auc_score(test_label, y_pred)
        print("概率值auc：", auc)
        ap = average_precision_score(test_label, y_pred)
        print("概率值ap：", ap)
        AUC_w_VGAE.append(auc)
        AUPR_w_VGAE.append(ap)



# print("random_auc:", np.array(AUC))
# print("random_auc_mean:", sum(np.array(AUC)) / 5)
# print("random_auc_s:", np.array(AUC_S))
# print("random_auc_s_mean:", sum(np.array(AUC_S)) / 5)
# print("random_auc_g:", np.array(AUC_G))
# print("random_auc_g_mean:", sum(np.array(AUC_G)) / 5)
#
#
# print("random_aupr:", np.array(AUPR))
# print("co_aupr_mean:", sum(np.array(AUPR)) / 5)
# print("random_aupr_s:", np.array(AUPR_S))
# print("co_aupr_smean:", sum(np.array(AUPR_S)) / 5)
# print("random_aupr_g:", np.array(AUPR_G))
# print("co_aupr_gmean:", sum(np.array(AUPR_G)) / 5)
# plt.plot(fpr, tpr, lw=lw, label='Average ROC curve(AUC ={:.4f})'.format(sum(np.array(AUC)) / 5))
# plt.legend(loc="lower right")
# plt.show()
print("random_auc_w_VGAE:", np.array(AUC_w_VGAE))
print("random_auc_w_VGAE_mean:", sum(np.array(AUC_w_VGAE)) / 5)
print("random_aupr_w_VGAE:", np.array(AUPR_w_VGAE))
print("co_aupr_w_VAGE_mean:", sum(np.array(AUPR_w_VGAE)) / 5)
