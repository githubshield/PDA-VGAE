# 各种评价的指标函数
# 求AUC和AUPR值
import numpy as np
from scipy import interp
from sklearn.metrics import roc_auc_score, average_precision_score
# 求AUC和PR曲线所用到的值
from sklearn.metrics import roc_curve, precision_recall_curve,auc
# 求precision，Recall， F1
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
import matplotlib.pyplot as plt



AUC = []
AUPR = []
mean_fpr = np.linspace(0, 1, 20000)
tpr = []



Acc = []
Pre = []
Rec = []
F1 = []
AUC = []
AUPR = []
import pandas as pd
# for i in range(5):
#     test_label = np.load("./result/VGAE/VGAE10_test_label_{0}.npy".format(i + 1))
#     y_pred = np.load("./result/VGAE/VGAE10_y_pred_{0}.npy".format(i + 1))
#     test_samples = np.load("./piRNA/test_samples_{0}.npy".format(i+1))
#     pd.DataFrame(test_label).to_csv("./case_study/test_label_{0}.csv".format(i+1))
#     pd.DataFrame(y_pred).to_csv("./case_study/y_pred_{0}.csv".format(i + 1))
#     pd.DataFrame(test_samples).to_csv("./case_study/test_samples_{0}.csv".format(i + 1))


# 将每个疾病所在的关联列置为0（去掉要验证的关联）
# for i in [0, 6, 7]:
#     # 遍历所有的疾病（取关联矩阵的前列）
#     test_label = np.load("./case_study/disease/test_samples_{0}.npy".format(i))
#     y_pred = np.load("./case_study/disease/disease_y_pred_{0}.npy".format(i))
#
#     pd.DataFrame(test_label).to_csv("./case_study/disease/test_samples_{0}.csv".format(i))
#     pd.DataFrame(y_pred).to_csv("./case_study/disease/disease_y_pred_{0}.csv".format(i))





# 画ROC曲线，画其他的可以把这一部分注释掉
# lw = 2
# plt.figure(figsize=(5, 5))
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC curves based on 5-cv')


# VGAE和VAE进行对比
# 读取每一测试集和标签的值
# test_label = np.load("./result/VAE/VAE2_test_label_{0}.npy".format(i + 1))
# y_pred = np.load("./result/VAE/VAE2_y_pred_{0}.npy".format(i + 1))


# 单个数据的roc曲线
def plot_auc_curves(fprs, tprs, auc, directory, name):
    mean_fpr = np.linspace(0, 1, 20000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, linestyle='--', label='Fold %d AUC: %.4f' % (i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    plt.plot(mean_fpr, mean_tpr, color='BlueViolet', alpha=0.9, label='Mean AUC: %.4f $\pm$ %.4f' % (mean_auc, auc_std))

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)

    # std_tpr = np.std(tpr, axis=0)
    # tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='LightSkyBlue', alpha=0.3, label='$\pm$ 1 std.dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.savefig(directory+'/%s.jpg' % name, dpi=1200, bbox_inches='tight')
    plt.show()

# 单个数据的PR曲线
def plot_prc_curves(precisions, recalls, prc, directory, name):
    mean_recall = np.linspace(0, 1, 20000)
    precision = []

    for i in range(len(recalls)):
        precision.append(interp(1-mean_recall, 1-recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.4, linestyle='--', label='Fold %d AUPR: %.4f' % (i + 1, prc[i]))

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    # mean_prc = metrics.auc(mean_recall, mean_precision)
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)
    plt.plot(mean_recall, mean_precision, color='BlueViolet', alpha=0.9,
             label='Mean AP: %.4f $\pm$ %.4f' % (mean_prc, prc_std))  # AP: Average Precision

    plt.plot([1, 0], [0, 1], linestyle='--', color='black', alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('P-R curve')
    plt.legend(loc='lower left')
    plt.savefig(directory + '/%s.jpg' % name, dpi=1200, bbox_inches='tight')
    plt.show()

# 两个数据的roc曲线
def plot_auc_curves2(A_fprs, A_tprs, A_auc, B_fprs, B_tprs, B_auc,directory, name):
    B_mean_fpr = np.linspace(0, 1, 20000)
    B_tpr = []

    for i in range(len(B_fprs)):
        B_tpr.append(interp(B_mean_fpr, B_fprs[i], B_tprs[i]))
        B_tpr[-1][0] = 0.0
        plt.plot(B_fprs[i], B_tprs[i], alpha=0.4, linestyle='--', label='MCVAE-Fold %d AUC: %.4f' % (i + 1, B_auc[i]))

    B_mean_tpr = np.mean(B_tpr, axis=0)
    B_mean_tpr[-1] = 1.0
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    B_mean_auc = np.mean(B_auc)
    B_auc_std = np.std(B_auc)
    plt.plot(B_mean_fpr, B_mean_tpr, color='BlueViolet', alpha=0.9, label='MCVAE Mean AUC: %.4f $\pm$ %.4f' % (B_mean_auc, B_auc_std))

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)

    A_mean_fpr = np.linspace(0, 1, 20000)
    A_tpr = []

    for i in range(len(A_fprs)):
        A_tpr.append(interp(A_mean_fpr, A_fprs[i], A_tprs[i]))
        A_tpr[-1][0] = 0.0
        # plt.plot(A_fprs[i], A_tprs[i], alpha=0.4, linestyle='--', label='Fold %d AUC: %.4f' % (i + 1, A_auc[i]))

    A_mean_tpr = np.mean(A_tpr, axis=0)
    A_mean_tpr[-1] = 1.0
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    A_mean_auc = np.mean(A_auc)
    A_auc_std = np.std(A_auc)
    plt.plot(A_mean_fpr,A_mean_tpr, color='red', alpha=0.9, label='MCVAGE-Mean AUC: %.4f $\pm$ %.4f' % (A_mean_auc, A_auc_std))

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)


    # std_tpr = np.std(tpr, axis=0)
    # tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='LightSkyBlue', alpha=0.3, label='$\pm$ 1 std.dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.savefig(directory+'/%s.jpg' % name, dpi=1200, bbox_inches='tight')
    plt.show()

# 两个数据的PR曲线
def plot_prc_curves2(A_precisions, A_recalls, A_prc, B_precisions, B_recalls, B_prc, directory, name):
    B_mean_recall = np.linspace(0, 1, 20000)
    B_precision = []

    for i in range(len(B_recalls)):
        B_precision.append(interp(1 - B_mean_recall, 1 - B_recalls[i], B_precisions[i]))
        B_precision[-1][0] = 1.0
        plt.plot(B_recalls[i], B_precisions[i], alpha=0.4, linestyle='--', label='VAE-Fold %d AUPR: %.4f' % (i + 1, B_prc[i]))

    B_mean_precision = np.mean(B_precision, axis=0)
    B_mean_precision[-1] = 0
    # mean_prc = metrics.auc(mean_recall, mean_precision)
    B_mean_prc = np.mean(B_prc)
    B_prc_std = np.std(B_prc)
    plt.plot(B_mean_recall, B_mean_precision, color='BlueViolet', alpha=0.9,
             label='VAE-Mean AP: %.4f $\pm$ %.4f' % (B_mean_prc, B_prc_std))  # AP: Average Precision


    A_mean_recall = np.linspace(0, 1, 20000)
    A_precision = []

    for i in range(len(A_recalls)):
        A_precision.append(interp(1-A_mean_recall, 1-A_recalls[i], A_precisions[i]))
        A_precision[-1][0] = 1.0
        # plt.plot(A_recalls[i], A_precisions[i], alpha=0.4, linestyle='--', label='Fold %d AP: %.4f' % (i + 1, prc[i]))

    A_mean_precision = np.mean(A_precision, axis=0)
    A_mean_precision[-1] = 0
    # mean_prc = metrics.auc(mean_recall, mean_precision)
    A_mean_prc = np.mean(A_prc)
    A_prc_std = np.std(A_prc)
    plt.plot(A_mean_recall, A_mean_precision, color='red', alpha=0.9,
             label='VGAE-Mean AUPP: %.4f $\pm$ %.4f' % (A_mean_prc, A_prc_std))  # AP: Average Precision

    plt.plot([1, 0], [0, 1], linestyle='--', color='black', alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('P-R curve')
    plt.legend(loc='lower left')
    plt.savefig(directory + '/%s.jpg' % name, dpi=1200, bbox_inches='tight')
    plt.show()


# 画roc曲线
fprs = []
tprs = []
for i in range(5):
    # 读取每一折的测试集和标签的值
    test_label = np.load("./result/VGAE/VGAE10_test_label_{0}.npy".format(i+1))
    y_pred = np.load("./result/VGAE/VGAE10_y_pred_{0}.npy".format(i+1))

    # 得到AUC值
    Auc = roc_auc_score(test_label,y_pred)
    # 求AUC曲线所用到的值
    fpr, tpr, thr = roc_curve(test_label, y_pred)
    fprs.append(fpr)
    tprs.append(tpr)


    # 画出ROC曲线 ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot(fpr, tpr, lw=lw, label='ROC curve fold-{}(AUC ={:.4f})'.format(i+1, Auc))

    AUC.append(Auc)

plot_auc_curves(fprs,tprs,AUC,"roc_pr_curve","VGAE_test_auc")
print("random_auc:", np.around(np.array(AUC),4))
print("random_auc_mean:", np.around(sum(np.array(AUC)) / 5,4))

# 画P-R曲线
pres = []
recs = []
for i in range(5):
    # 读取每一折的测试集和标签的值
    test_label = np.load("./result/VGAE/VGAE10_test_label_{0}.npy".format(i+1))
    y_pred = np.load("./result/VGAE/VGAE10_y_pred_{0}.npy".format(i+1))
    # 得到AUPR值
    aupr = average_precision_score(test_label, y_pred)
    # 求AUPR曲线所用到的值
    pre, rec, thr = precision_recall_curve(test_label, y_pred)
    pres.append(pre)
    recs.append(rec)
    # 画出ROC曲线 ###假正率为横坐标，真正率为纵坐标做曲线
    AUPR.append(aupr)

plot_prc_curves(pres, recs, AUPR,"roc_pr_curve","VGAE_test_aupr")
print("random_aupr:", np.around(np.array(AUPR),4))
print("random_aupr_mean:", np.around(sum(np.array(AUPR)) / 5,4))

# 画roc曲线
VAEfprs = []
VAEtprs = []
VAEAUC = []
for i in range(5):
    # 读取每一折的测试集和标签的值
    test_label = np.load("./result/VAE/VAE1_test_label_{0}.npy".format(i+1))
    y_pred = np.load("./result/VAE/VAE1_y_pred_{0}.npy".format(i+1))

    # 得到AUC值
    Auc = roc_auc_score(test_label,y_pred)
    # 求AUC曲线所用到的值
    fpr, tpr, thr = roc_curve(test_label, y_pred)
    VAEfprs.append(fpr)
    VAEtprs.append(tpr)


    # 画出ROC曲线 ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot(fpr, tpr, lw=lw, label='ROC curve fold-{}(AUC ={:.4f})'.format(i+1, Auc))

    VAEAUC.append(Auc)


plot_auc_curves2(fprs,tprs,AUC,VAEfprs,VAEtprs,VAEAUC,"roc_pr_curve","VAE_test_auc")
print("random_VAEAUC:", np.around(np.array(VAEAUC),4))
print("random_VAEAUC_mean:", np.around(sum(np.array(VAEAUC)) / 5,4))
# 画P-R曲线
VAEpres = []
VAErecs = []
VAEAUPR = []
for i in range(5):
    # 读取每一折的测试集和标签的值
    test_label = np.load("./result/VAE/VAE1_test_label_{0}.npy".format(i+1))
    y_pred = np.load("./result/VAE/VAE1_y_pred_{0}.npy".format(i+1))
    # 得到AUPR值
    VAEaupr = average_precision_score(test_label, y_pred)
    # 求AUPR曲线所用到的值
    VAEpre, VAErec, VAEthr = precision_recall_curve(test_label, y_pred)
    VAEpres.append(VAEpre)
    VAErecs.append(VAErec)
    # 画出ROC曲线 ###假正率为横坐标，真正率为纵坐标做曲线
    VAEAUPR.append(VAEaupr)

plot_prc_curves2(pres, recs, AUPR,VAEpres, VAErecs, VAEAUPR, "roc_pr_curve", "VAE_test_aupr")
print("random_aupr:", np.around(np.array(AUPR),4))
print("random_aupr_mean:", np.around(sum(np.array(AUPR)) / 5,4))





# 计算precision， recall， f1-socre的值
for i in range(5):
    # 读取每一折的测试集和标签的值
    # test_label = np.load("./result/GCN/GCN3_test_label_{0}.npy".format(i+1))
    # y_pred = np.load("./result/GCN/GCN3_y_pred_{0}.npy".format(i+1))

    test_label = np.load("./result/MCVGAE/MCVGAE_test_label_{0}.npy".format(i + 1))
    y_pred = np.load("./result/MCVGAE/MCVGAE_y_pred_{0}.npy".format(i + 1))
    # 得到AUC值
    Auc = roc_auc_score(test_label.reshape(-1,1), y_pred)
    # 得到AUPR值
    Aupr = average_precision_score(test_label.reshape(-1,1), y_pred)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    # 得到accuracy值
    acc = accuracy_score(test_label.reshape(-1,1), y_pred)
    # 得到precision值
    pre = precision_score(test_label.reshape(-1,1), y_pred)
    # 得到recall值
    rec = recall_score(test_label.reshape(-1,1), y_pred)
    # 得到F1值
    f1 = f1_score(test_label.reshape(-1,1), y_pred)


    Acc.append(acc)
    Pre.append(pre)
    Rec.append(rec)
    F1.append(f1)
    AUC.append(Auc)
    AUPR.append(Aupr)



print("accuracy:{}".format(np.around(np.array(Acc),4)))
print("accuracy_mean:{}".format(np.around(sum(np.array(Acc)) / 5,4)))

print("precison:{}".format(np.around(np.array(Pre),4)))
print("precision_mean:{}".format(np.around(sum(np.array(Pre)) / 5,4)))

print("Recall:{}".format(np.around(np.array(Rec),4)))
print("Recall_mean:{}".format(np.around(sum(np.array(Rec)) / 5,4)))

print("F1:{}".format(np.around(np.array(F1),4)))
print("F1_mean:{}".format(np.around(sum(np.array(F1)) / 5,4)))

print("AUC:{}".format(np.around(np.array(AUC),4)))
print("AUC_mean:{}".format(np.around(sum(np.array(AUC)) / 5,4)))

print("aupr:{}".format(np.around(np.array(AUPR),4)))
print("AUPR_mean:{}".format(np.around(sum(np.array(AUPR)) / 5,4)))





VAEAcc = []
VAEPre = []
VAERec = []
VAEF1 = []
for i in range(5):
    # 读取每一折的测试集和标签的值
    test_label = np.load("./result/VAE/VAE1_test_label_{0}.npy".format(i+1))
    y_pred = np.load("./result/VAE/VAE1_y_pred_{0}.npy".format(i+1))
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    # 得到accuracy值
    acc = accuracy_score(test_label.reshape(-1,1), y_pred)
    # 得到precision值
    pre = precision_score(test_label.reshape(-1,1), y_pred)
    # 得到recall值
    rec = recall_score(test_label.reshape(-1,1), y_pred)
    # 得到F1值
    f1 = f1_score(test_label.reshape(-1,1), y_pred)

    VAEAcc.append(acc)
    VAEPre.append(pre)
    VAERec.append(rec)
    VAEF1.append(f1)



# print("accuracy:{}".format(np.around(np.array(Acc),4)))
print("VAEaccuracy_mean:{}".format(np.around(sum(np.array(VAEAcc)) / 5,4)))

# print("precison:{}".format(np.around(np.array(Pre),4)))
print("VAEprecision_mean:{}".format(np.around(sum(np.array(VAEPre)) / 5,4)))

# print("Recall:{}".format(np.around(np.array(Rec),4)))
print("VAERecall_mean:{}".format(np.around(sum(np.array(VAERec)) / 5,4)))

# print("F1:{}".format(np.around(np.array(F1),4)))
print("VAEF1_mean:{}".format(np.around(sum(np.array(F1)) / 5,4)))








