import numpy as np
import pandas as pd
import torch.nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import *

from sklearn.tree import DecisionTreeClassifier
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
# 读取矩阵文件
def read_txt(path):
    '''
    读取矩阵文件
    :param path: 文件的路径
    :return: 文件数据的矩阵（ndarray类型）
    '''
    with open(path, 'r', newline='') as txt_file:
        md_data = []
        reader = txt_file.readlines()
        for row in reader:
            line = row.split(',')
            row = []
            for k in line:
                row.append(float(k))
            md_data.append(row)
        md_data = np.array(md_data)
        return md_data
# 计算准确率
def accuracy(adj_rec, adj_label):
    '''
    :param adj_rec: 预测值
    :param adj_label: 目标值
    :return: accuracy
    '''
    labels_all = adj_label.view(-1)
    preds_all = adj_rec .view(-1)
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy
# 计算损失函数的参数值
def compute_loss_para(adj):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm
# 计算GIP相似矩阵
def getGosiR (Asso_RNA_Dis):
# calculate the r in GOsi Kerel

    nc = Asso_RNA_Dis.shape[0]
    summ = 0
    for i in range(nc):
        x_norm = np.linalg.norm(Asso_RNA_Dis[i,:])
        x_norm = np.square(x_norm)
        summ = summ + x_norm
    r = summ / nc
    return r
# 利用GIP核相似算法求得GIP相似矩阵
def GIP_kernel (Asso_RNA_Dis):
    # the number of row
    nc = Asso_RNA_Dis.shape[0]
    #initate a matrix as result matrix
    matrix = np.zeros((nc, nc))
    # calculate the down part of GIP fmulate
    r = getGosiR(Asso_RNA_Dis)
    #calculate the result matrix
    for i in tqdm(range(nc),desc="GIP"):
        for j in range(nc):
            #calculate the up part of GIP formulate
            temp_up = np.square(np.linalg.norm(Asso_RNA_Dis[i,:] - Asso_RNA_Dis[j,:]))
            if r == 0:
                matrix[i][j]=0
            elif i==j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e**(-temp_up/r)
    return matrix
# 标准化处理
def Knormalized(K):
    #kernel normilization
    K = np.abs(K)
    min_v = np.min(K)
    K[K == 0] = min_v
    D = np.diag(K)
    D = np.sqrt(D).reshape(-1,1)
    S = np.divide(K, np.dot(D,D.T))
    return S

def get_all_the_samples(A):
    m,n = A.shape
    pos = []
    neg = []
    for i in range(m):
        for j in range(n):
            if A[i,j] ==1:
                pos.append([i,j,1])
            else:
                neg.append([i,j,0])
    n = len(pos)
    neg_new = random.sample(neg, n)
    tep_samples = pos + neg_new
    samples = random.sample(tep_samples, len(tep_samples))
    samples = random.sample(samples, len(samples))
    samples = np.array(samples)
    return samples
# 获取实验样本的三元组形式
def get_all_the_sample2(A):
    m,n = A.shape
    all_samples = []
    for i in range(m):
        for j in range(n):
            all_samples.append([i,j,A[i,j]])
    samples = np.array(all_samples,dtype="int64")
    return samples
# 该函数主要是得到（piRNA，disease）样本的特征向量
def get_two_feature(samples, p_features, d_features, p_G_features, d_G_features):
    '''
    :param samples: 所有样本的集合
    :param p_features: piRNA的特征
    :param d_features: 疾病的特征
    :param p_G_features: piRNA的GIP相似特征
    :param d_G_features: 疾病的GIP相似特征
    :return:
    train_feature_bal：组合之后的特征向量
    train_label_bal：每个特征向量的标签
    '''
    p_features = p_features.cpu().detach().numpy()
    d_features = d_features.cpu().detach().numpy()
    p_G_features = p_G_features.cpu().detach().numpy()
    d_G_features = d_G_features.cpu().detach().numpy()


    # p_features = p_features
    # d_features = d_features
    # p_G_features = p_G_features
    # d_G_features = d_G_features

    n = samples.shape[0]
    vect_len1 = p_features.shape[1]
    vect_len2 = d_features.shape[1]
    vect_len3 = p_G_features.shape[1]
    vect_len4 = d_G_features.shape[1]

    train_feature_bal = np.zeros([n, (vect_len1+vect_len2+vect_len3+vect_len4)])
    train_label_bal = np.zeros([n])
    for i in tqdm(range(n),desc="get_feature"):
        train_feature_bal[i, 0:vect_len1] = p_features[int(samples[i, 0]), :]
        train_feature_bal[i, vect_len1:(vect_len1+vect_len2)] = d_features[int(samples[i, 1]), :]

        train_feature_bal[i, (vect_len1+vect_len2):(vect_len1+vect_len2+vect_len3)] = \
            p_G_features[int(samples[i, 0]), :]
        train_feature_bal[i, (vect_len1 + vect_len2 + vect_len3):(vect_len1 + vect_len2+vect_len3+vect_len4)] = \
            d_G_features[int(samples[i, 1]), :]

        train_label_bal[i] = int(samples[i, 2])

    return train_feature_bal, train_label_bal
# 该函数主要是得到单个（piRNA，disease）样本的特征向量
def get_sig_feature(samples, p_features, d_features, type=None):
    '''
    :param samples: 所有样本的集合
    :param p_features: piRNA的特征
    :param d_features: 疾病的特征
    :param p_G_features: piRNA的GIP相似特征
    :param d_G_features: 疾病的GIP相似特征
    :return:
    train_feature_bal：组合之后的特征向量
    train_label_bal：每个特征向量的标签
    '''
    if type=="torch":
        p_features = p_features.cpu().detach().numpy()
        d_features = d_features.cpu().detach().numpy()
    else:
        p_features = p_features
        d_features = d_features

    n = samples.shape[0]
    vect_len1 = p_features.shape[1]
    vect_len2 = d_features.shape[1]
    train_feature_bal = np.zeros([n, vect_len1+vect_len2])
    train_label_bal = np.zeros([n])
    for i in range(n):
        train_feature_bal[i, 0:vect_len1] = p_features[int(samples[i, 0]), :]
        train_feature_bal[i, vect_len1:(vect_len1+vect_len2)] = d_features[int(samples[i, 1]), :]
        # train_feature_bal[i, vect_len1:(vect_len1 + vect_len2)] = d_features[int(samples[i, 1]), :]
        train_label_bal[i] = int(samples[i, 2])
    return np.float32(train_feature_bal), train_label_bal
# 该函数主要是得到到样本的节点u，v（RNA，disease）以及边的权重
def get_edges(A):
    '''
    :param A: 一个邻接矩阵
    :return: 返回两个节点的边，u，v
    u:Tensor
    v.Tensor
    w:weight
    '''
    m,n = A.shape
    u = []
    v = []
    w = []
    negative_samples = []
    for i in tqdm(range(m), desc="get_edges"):
        for j in range(n):
            if A[i,j] == 0 :
                negative_samples.append([i,j,1])
            else:
                u.append(i)
                v.append(j)
                w.append(float(A[i][j]))
    negative_samples = np.array(negative_samples)
    u = np.array(u)
    v = np.array(v)
    w = np.array(w)
    return u,v,w

'''
得到与正样本等量的可靠的负样本
'''
def pul_get_the_samples(samples, p_features, d_features, RG_features, DG_features):
    # 获得训练的样本特征X以及标签y
    X, y = get_two_feature(samples, p_features, d_features, RG_features, DG_features)
    # 将全部的样本转化pandas类型，便于索引操作
    samples = pd.DataFrame(samples)
    # X则是一个DataFrame类型
    X = pd.DataFrame(X)
    # y为一个索引类型
    y = pd.Series(y)
    # Keep track of the indices of positive and unlabeled data points
    # 跟踪正数据点和未标记数据点的索引
    iP = y[y > 0].index  # 正样本的索引
    iU = y[y <= 0].index  # 未标记样本的索引
    # step 1.随机从未知样本中选取与正样本等量的作为负样本
    iN = np.random.choice(iU, len(iP))

    # 使用该正负样本进行训练
    Xb = X.loc[iN].append(X.loc[iP])
    yb = y.loc[iN].append(y.loc[iP])
    Xb = torch.Tensor(np.array(Xb)).to(device)
    yb = torch.Tensor(np.array(yb)).to(device)
    # 使用深度学习的方法来进行预测训练
    # 定义模型
    model = MLPModel(input_dim=Xb.shape[1],hidden_dim1=128,hidden_dim2=64,
                     output_dim=2).to(device)
    # 定义优化函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
    # 定义损失函数
    loss_function = torch.nn.BCEWithLogitsLoss()
    # 定义训练轮次
    epoch = 200
    for i in range(epoch):
        t = time.time()
        pre = model(Xb)[:, 1]
        loss = loss_function(pre, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 打印损失
        print("Epoch:", '%04d' % (i + 1), "train_loss=", "{:.5f}".format(loss.item()),
        	  "time=", "{:.5f}".format(time.time() - t))
    # 需要标记的样本
    i_oob = list(set(iU) - set(iN))
    # 得到预测分数
    oob = torch.Tensor(np.array(X.loc[i_oob])).to(device)
    pre = model(oob)[:, 1]
    pre = pre.detach().cpu().numpy()
    # 保存预测分数的索引
    scores = pd.DataFrame(np.zeros(shape=y.shape), index=y.index)
    scores.loc[i_oob, 0] = pre
    # Finally, store the scores assigned by this approach
    output_bag = scores.loc[i_oob, 0].sort_values(axis=0, ascending=False).reset_index().values
    # np.save("./output_bag.npy",output_bag)
    # output_bag = np.load("./output_bag.npy")
    # 去掉列表中的nan的行
    # output_bag = np.array(list(filter(lambda x: not np.isnan(x[1]), output_bag)))
    need_N_index = output_bag[:, 0]
    # 取前面1/3作为可靠负样本
    # index = need_N_index[:int((output_bag.shape[0] / 3))]
    # 取中间1/3作为可靠负样本
    index = need_N_index[int(output_bag.shape[0] / 3):int((output_bag.shape[0] / 3) * 2)]
    # 取后面1/3作为可靠负样本
    # index = need_N_index[int(output_bag.shape[0] / 3)*2:]
    need_neg_samples = samples.loc[index]

    return need_neg_samples.values.tolist()
# SVM得到与正样本等量的可靠的负样本
def pul_SVM_get_the_samples(samples, p_features, d_features):
    # 获得训练的样本特征X以及标签y
    X, y = get_sig_feature(samples, p_features, d_features)
    # 将全部的样本转化pandas类型，便于索引操作
    samples = pd.DataFrame(samples)
    # X则是一个DataFrame类型
    X = pd.DataFrame(X)
    # y为一个索引类型
    y = pd.Series(y)
    # Keep track of the indices of positive and unlabeled data points
    # 跟踪正数据点和未标记数据点的索引
    iP = y[y > 0].index  # 正样本的索引
    iU = y[y <= 0].index  # 未标记样本的索引
    # step 1.随机从未知样本中选取与正样本等量的作为负样本
    iN = np.random.choice(iU, len(iP))

    # 使用该正负样本进行训练
    Xb = X.loc[iN].append(X.loc[iP])
    yb = y.loc[iN].append(y.loc[iP])
    # # use SVM
    from sklearn.svm import SVC
    # from thundersvm import SVC
    estimator = SVC(C=1.0, kernel='rbf', gamma=1, probability=True)
    # from sklearn.ensemble import RandomForestClassifier
    # estimator = RandomForestClassifier(n_estimators=80, max_features=0.2, n_jobs=-1)
    estimator.fit(Xb, yb)

    # 需要标记的样本
    i_oob = list(set(iU) - set(iN))
    # 得到预测分数
    pre = estimator.predict_proba(X.loc[i_oob])[:, 1]

    # 保存预测分数的索引
    scores = pd.DataFrame(np.zeros(shape=y.shape), index=y.index)
    scores.loc[i_oob, 0] = pre
    # Finally, store the scores assigned by this approach
    output_bag = scores.loc[i_oob, 0].sort_values(axis=0, ascending=False).reset_index().values
    # np.save("./output_bag.npy",output_bag)
    # output_bag = np.load("./output_bag.npy")
    # 去掉列表中的nan的行
    # output_bag = np.array(list(filter(lambda x: not np.isnan(x[1]), output_bag)))
    need_N_index = output_bag[:, 0]
    # 取前面1/3作为可靠负样本
    #     index = need_N_index[:int((output_bag.shape[0] / 3))]
    # 取中间1/3作为可靠负样本
    # index = need_N_index[int(output_bag.shape[0] / 3):int((output_bag.shape[0] / 3) * 2)]
    # 取后面1/3作为可靠负样本
    index = need_N_index[int(output_bag.shape[0] / 3)*2:]
    need_neg_samples = samples.loc[index]

    return need_neg_samples.values.tolist()

import copy

def cotraining(X1, X2, y, clf, U_=7500):
    clf1 = clf
    clf2 = copy.copy(clf)
    # 将数据转化为DataFrame数据便于索引操作
    X1 = pd.DataFrame(X1)
    X2 = pd.DataFrame(X2)
    y = pd.Series(y)
    # a) 分别用L训练出一个模型 F1和F2
    iL = y[y > -1].index  # 标记数据的索引
    iU = y[y <= -1].index # 未标记数据的索引
    # b) 分别用模型F1和F2去预测U (给U打标签)，每次训练打U_个
    D1, L1 = X1.loc[iL], y.loc[iL]
    D2, L2 = X2.loc[iL], y.loc[iL]
    while len(iU) != 0:
        # 使用标记好的数据集训练好两个模型
        F1 = clf1.fit(D1, L1.astype('int'))
        F2 = clf2.fit(D2, L2.astype('int'))
        iU_= np.random.choice(iU, min(U_,len(iU)))  # 从未标记的索引中随机取出U_个
        y_pre1 = F1.predict_proba(X1.loc[iU_])[:, 1]  # 模型F1的预测正样本的值
        y_pre2 = F2.predict_proba(X2.loc[iU_])[:, 1]  # 模型F2的预测正样本的值
        # 将预测好的L2标签放入X1中
        y_pre2[y_pre2 > 0.5] = int(1)
        y_pre2[y_pre2 < 0.5] = int(0)
        y2 = y.copy()
        y2.loc[iU_] = y_pre2
        # 将预测好的X2的特征和标签添加到D1中
        D1 = pd.concat([D1, X2.loc[iU_]], axis=0)
        L1 = pd.concat([L1, pd.Series(y2.loc[iU_])], axis=0)
        # 将预测好的L1标签放入X2中
        y_pre1[y_pre1 > 0.5] = int(1)
        y_pre1[y_pre1 < 0.5] = int(0)
        y1 = y.copy()
        y1.loc[iU_] = y_pre1
        # 将预测好的X2的特征和标签添加到D1中
        D2 = pd.concat([D2, X1.loc[iU_]], axis=0)
        L2 = pd.concat([L2, pd.Series(y1.loc[iU_])], axis=0)
        # 如果某一个模型的预测值较大，则作为最后的预测值
        y_pre = []  # 保存最后的预测概率
        for i in tqdm(range(min(U_,len(iU))),desc="co-training"):
        # for i in range(min(U_, len(iU))):
            if y_pre1[i] > y_pre2[i]:
                y_pre.append(y_pre1[i])
            else:
                y_pre.append(y_pre2[i])
        # 将预测完的未知标签的样本打上标签
        y_pre = np.array(y_pre)
        y_pre[y_pre > 0.5] = int(1)
        y_pre[y_pre < 0.5] = int(0)
        y.loc[iU_] = y_pre

        # 将预测完的标签从未知标签中删除
        iU = list(set(iU) - set(iU_))

    # 返回最后全部打好标签的标签值y
    return y

def cotraining_mean(X1, X2, y, clf, U_=7500):
    clf1 = clf
    clf2 = copy.copy(clf)
    # 将数据转化为DataFrame数据便于索引操作
    X1 = pd.DataFrame(X1)
    X2 = pd.DataFrame(X2)
    y = pd.Series(y)
    # a) 分别用L训练出一个模型 F1和F2
    iL = y[y > -1].index  # 标记数据的索引
    iU = y[y <= -1].index # 未标记数据的索引
    # b) 分别用模型F1和F2去预测U (给U打标签)，每次训练打U_个
    D1, L1 = X1.loc[iL], y.loc[iL]
    D2, L2 = X2.loc[iL], y.loc[iL]
    while len(iU) != 0:
        # 使用标记好的数据集训练好两个模型
        F1 = clf1.fit(D1, L1.astype('int'))
        F2 = clf2.fit(D2, L2.astype('int'))
        iU_= np.random.choice(iU, min(U_,len(iU)))  # 从未标记的索引中随机取出U_个
        y_pre1 = F1.predict_proba(X1.loc[iU_])[:, 1]  # 模型F1的预测正样本的值
        y_pre2 = F2.predict_proba(X2.loc[iU_])[:, 1]  # 模型F2的预测正样本的值
        # 将预测好的L2标签放入X1中
        y_pre2[y_pre2 > 0.5] = int(1)
        y_pre2[y_pre2 < 0.5] = int(0)
        y2 = y.copy()
        y2.loc[iU_] = y_pre2
        # 将预测好的X2的特征和标签添加到D1中
        D1 = pd.concat([D1, X2.loc[iU_]], axis=0)
        L1 = pd.concat([L1, pd.Series(y2.loc[iU_])], axis=0)
        # 将预测好的L1标签放入X2中
        y_pre1[y_pre1 > 0.5] = int(1)
        y_pre1[y_pre1 < 0.5] = int(0)
        y1 = y.copy()
        y1.loc[iU_] = y_pre1
        # 将预测好的X2的特征和标签添加到D1中
        D2 = pd.concat([D2, X1.loc[iU_]], axis=0)
        L2 = pd.concat([L2, pd.Series(y1.loc[iU_])], axis=0)
        # 如果某一个模型的预测值较大，则作为最后的预测值
        y_pre = []  # 保存最后的预测概率
        for i in tqdm(range(min(U_,len(iU))),desc="co-training"):
        # for i in range(min(U_, len(iU))):
        # 这里我们求两者的平均值
            y_pre.append(0.5*(y_pre1[i]+y_pre2[i]))
        # 将预测完的未知标签的样本打上标签
        y_pre = np.array(y_pre)
        y_pre[y_pre > 0.5] = int(1)
        y_pre[y_pre < 0.5] = int(0)
        y.loc[iU_] = y_pre

        # 将预测完的标签从未知标签中删除
        iU = list(set(iU) - set(iU_))

    # 返回最后全部打好标签的标签值y
    return y


# 将数据进行融合
#W is the matrix which needs to be normalized
# 对应论文中的公式（1）
def new_normalization (w):
    m = w.shape[0]
    p = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            if i == j:
                p[i][j] = 1/2
            elif np.sum(w[i,:])-w[i,i]>0:
                p[i][j] = w[i,j]/(2*(np.sum(w[i,:])-w[i,i]))
    return p
# get the KNN kernel, k is the number if first nearest neibors

# 欧式距离相似性
def euclidean_dist(x, y=None):
    """
    Compute all pairwise distances between vectors in X and Y matrices.
    :param x: numpy array, with size of (d, m)
    :param y: numpy array, with size of (d, n)
    :return: EDM:   numpy array, with size of (m,n).
                    Each entry in EDM_{i,j} represents the distance between row i in x and row j in y.
    """
    if y is None:
        y = x

    # calculate Gram matrices
    G_x = np.matmul(x, x.T)
    G_y = np.matmul(y, y.T)

    # convert diagonal matrix into column vector
    diag_Gx = np.reshape(np.diag(G_x), (-1, 1))
    diag_Gy = np.reshape(np.diag(G_y), (-1, 1))

    # Compute Euclidean distance matrix
    EDM = diag_Gx + diag_Gy.T - 2*np.matmul(x, y.T) # broadcasting

    return EDM

# pearson similarity
def pearson_similarity(Asso_RNA_Dis):
    # the number of row
    nc = Asso_RNA_Dis.shape[0]
    #initate a matrix as result matrix
    matrix = np.zeros((nc, nc))
    for i in tqdm(range(nc),desc="pearson"):
        for j in range(nc):
            x_ = Asso_RNA_Dis[i,:] - np.mean(Asso_RNA_Dis[i,:])
            y_ = Asso_RNA_Dis[j,:] - np.mean(Asso_RNA_Dis[j,:])
            cov_product = np.dot(x_,y_)
            temp_up = np.linalg.norm(Asso_RNA_Dis[i,:])*np.linalg.norm(Asso_RNA_Dis[j,:])
            if cov_product == 0:
                matrix[i][j] = 0
            elif i==j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = cov_product / temp_up
    return matrix



# 使用矩阵分解获得线性特征表示
def objective_function(W, A, U, V, lam):
    m, n = A.shape
    sum_obj = 0
    for i in range(m):
        for j in range(n):
            #print("the shape of Ui", U[i,:].shape, V[:,j].shape)
            sum_obj = sum_obj + W[i,j]*(A[i,j] - U[i,:].dot(V[:,j]))+ lam*(np.linalg.norm(U[i, :], ord=1,keepdims= False) + np.linalg.norm(V[:, j], ord = 1, keepdims = False))
    return  sum_obj
def updating_U (W, A, U, V, lam):
    m, n = U.shape
    fenzi = (W*A).dot(V.T)
    fenmu = (W*(U.dot(V))).dot((V.T)) + (lam/2) *(np.ones([m, n]))

    # fenmu = (W*(U.dot(V))).dot((V.T)) + lam*U
    U_new = U
    for i in range(m):
    # for i in tqdm(range(m),desc="get_low_u"):
        for j in range(n):
            U_new[i,j] = U[i, j]*(fenzi[i,j]/fenmu[i, j])
    return U_new
def updating_V (W, A, U, V, lam):
        m,n = V.shape
        fenzi = (U.T).dot(W*A)
        fenmu = (U.T).dot(W*(U.dot(V)))+(lam/2)*(np.ones([m,n]))
        # fenmu = (U.T).dot(W*(U.dot(V)))+lam*V
        V_new = V
        for i in range(m):
        # for i in tqdm(range(m),desc="get_low_v"):
            for j in range(n):
                V_new[i,j] = V[i, j]*(fenzi[i,j]/fenmu[i,j])
        return V_new
def get_low_feature(k,lam, th, A):#k is the number elements in the features, lam is the parameter for adjusting, th is the threshold for coverage state
    m, n = A.shape
    arr1=np.random.randint(0,100,size=(m,k))
    U = arr1/100#miRNA
    arr2=np.random.randint(0,100,size=(k,n))
    V = arr2/100#disease
    obj_value = objective_function(A, A, U, V, lam)
    obj_value1 = obj_value + 1
    i = 0
    diff = abs(obj_value1 - obj_value)
    while i < 1000:
        i =i + 1
        U = updating_U(A, A, U, V, lam)
        V = updating_V(A, A, U, V, lam)
        # obj_value1 = obj_value
        # obj_value = objective_function(A, A, U, V, lam)
        # diff = abs(obj_value1 - obj_value)
        # print("ite ", i, diff, obj_value1)
        #print("iter", i)
    #print(U)
    #print(V)
    return U, V.transpose()