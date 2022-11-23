import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, random
import numpy as np
import copy
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from tools import *

from dgl.nn.pytorch import SGConv,GraphConv,GATConv,GINConv

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')


class VGAEModel(nn.Module):
	def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout):
		super(VGAEModel, self).__init__()
		self.hidden_dim2 = hidden_dim2
		self.hidden1 = GraphConv(input_dim, hidden_dim1, weight=True, bias=True, activation=F.relu, allow_zero_in_degree=True)
		self.z_mean = GraphConv(hidden_dim1, hidden_dim2, weight=True, bias=True,activation=lambda x: x, allow_zero_in_degree=True)
		self.z_log_std = GraphConv(hidden_dim1, hidden_dim2, weight=True, bias=True,activation=lambda x: x, allow_zero_in_degree=True)
		self.dc = InnerProductDecoder(dropout, act=lambda x: x)

	def forward(self, g, features):
		h = self.hidden1(g, features)  # 第一层得到输出h
		self.mean = self.z_mean(g, h)  # 第二层求均值
		self.log_std = self.z_log_std(g, h)  # 第二层求方差
		gaussian_noise = torch.randn(features.size(0), self.hidden_dim2).to(device)  # 标准高斯分布采样，大小是features_size*hidden2_dim
		represrntation = self.mean + gaussian_noise * torch.exp(self.log_std).to(device) # 这里其实是reparameterization trick，具体看公式1和代码如何对应
		pre = torch.sigmoid(self.dc(represrntation)) # 得到最后的预测值

		return pre,represrntation

class VAEModel(nn.Module):
	def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout):
		super(VAEModel, self).__init__()
		self.hidden_dim2 = hidden_dim2
		self.hidden1 = nn.Linear(input_dim, hidden_dim1, bias=True)
		self.z_mean = nn.Linear(hidden_dim1, hidden_dim2,bias=True)
		self.z_log_std = nn.Linear(hidden_dim1, hidden_dim2, bias=True)
		self.dc = InnerProductDecoder(dropout, act=lambda x: x)

	def forward(self,features):
		features = torch.Tensor(features).to(device)
		h = self.hidden1(features)# 第一层得到输出h
		self.mean = self.z_mean(h)  # 第二层求均值
		self.log_std = self.z_log_std(h)  # 第二层求方差
		gaussian_noise = torch.randn(features.size(0), self.hidden_dim2).to(device)  # 标准高斯分布采样，大小是features_size*hidden2_dim
		represrntation = self.mean + gaussian_noise * torch.exp(self.log_std).to(device) # 这里其实是reparameterization trick，具体看公式1和代码如何对应
		pre = torch.sigmoid(self.dc(represrntation)) # 得到最后的预测值

		return pre,represrntation

class GCN(nn.Module):
	def __init__(self,
				 input_dim,  # 输入维度
				 hidden_dim1,  # 隐藏层维度
				 out_put,  # 输出层维度
				 feat_drop  # 特征层的丢弃率
				 ):
		super(GCN, self).__init__()
		self.conv1 = GraphConv(input_dim, hidden_dim1, weight=True, bias=True, activation=F.relu, allow_zero_in_degree=True)
		self.conv2 = GraphConv(hidden_dim1, out_put, weight=True, bias=True, activation=F.relu, allow_zero_in_degree=True)
		# 点积层还原链接预测
		self.dc = InnerProductDecoder(feat_drop, act=lambda x: x)

	def forward(self, g, inputs):
		h = self.conv1(g, inputs)
		h = F.relu(h)
		self.mean = self.conv2(g, h)  # 第二层求均值
		self.log_std = self.conv2(g, h)  # 第二层求方差
		h = self.conv2(g, h)
		return self.dc(h), h

class GAT(nn.Module):
	def __init__(self,
				 num_layers,  # 层数
				 in_dim,  # 输入维度
				 num_hidden,  # 隐藏层维度
				 out_put,  # 输出层维度
				 heads,  # 多头注意力的计算次数
				 feat_drop,  # 特征层的丢弃率
				 attn_drop,  # 注意力分数的丢弃率
				 negative_slope,  # LeakyReLU激活函数的负向参数
		):
		super(GAT, self).__init__()
		self.num_layers = num_layers
		self.gat_layers = nn.ModuleList()
		self.activation = F.relu
		self.gat_layers.append(GATConv(in_feats=in_dim, out_feats=num_hidden,num_heads=heads[0],
									   feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope,
									  activation=self.activation))
		# 定义隐藏层
		for l in range(1, num_layers):
			# 多头注意力 the in_dim = num_hidden * num_heads
			self.gat_layers.append(GATConv(
				num_hidden * heads[l - 1], num_hidden, heads[l],
				feat_drop, attn_drop, negative_slope,  self.activation))
		# 输出层
		self.gat_layers.append(GATConv(
			num_hidden * heads[-2], out_put, heads[-1],
			feat_drop, attn_drop, negative_slope,None))

		# 点积层还原链接预测
		self.dc = InnerProductDecoder(feat_drop, act=lambda x: x)

	def forward(self, g, inputs):
		h = inputs
		for l in range(self.num_layers):  # 隐藏层
			h = self.gat_layers[l](g, h).flatten(1)
		# 输出层
		logits = self.gat_layers[-1](g, h).mean(1)
		return self.dc(logits), logits



class InnerProductDecoder(nn.Module):
	"""Decoder for using inner product for prediction."""
	def __init__(self, dropout, act=torch.sigmoid):
		super(InnerProductDecoder, self).__init__()
		self.dropout = dropout
		self.act = act

	def forward(self, z):
		z = F.dropout(z, self.dropout, training=self.training)
		adj = self.act(torch.mm(z, z.t()))
		return adj

class MLPModel(nn.Module):
	def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
		super(MLPModel, self).__init__()
		self.MLP_hidden_layer1 = nn.Linear(input_dim, hidden_dim1)
		self.MLP_hidden_layer2 = nn.Linear(hidden_dim1, hidden_dim2)
		self.MLP_output_layer = nn.Linear(hidden_dim2, output_dim)

	def fit(self, x):
		h = self.MLP_hidden_layer1(x)  # 第一层神经网络
		h = nn.ReLU()(h)  # 激活函数
		h = self.MLP_hidden_layer2(h)  # 第二层神经网络
		h = nn.ReLU()(h)  # 激活函数
		output_emb = self.MLP_output_layer(h)  # 输出层神经网络
		output_emb = torch.sigmoid(output_emb)
		return output_emb

	def predict_proba(self, test_x):
		test_x = torch.Tensor(np.array(test_x.cpu())).to(device)  # 防止存在pandas的类型的数据
		pre = self.fit(test_x)  # 将训练好的模型用来预测最后的值
		# 输出正样本的概率值
		pre = pre.detach().cpu().numpy()  # 将最终的值转化为CPU，且为numpy的格式
		return pre

class MLPClf(nn.Module):
	def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, epoch, lr=1e-3):
		super(MLPClf, self).__init__()
		self.model = MLPModel(input_dim=input_dim,
						 hidden_dim1=hidden_dim1,
						 hidden_dim2=hidden_dim2,
						# hidden_dim3=hidden_dim3,
						 output_dim=output_dim).to(device)
		self.MLP_hidden_layer1 = nn.Linear(input_dim, hidden_dim1)
		self.MLP_hidden_layer2 = nn.Linear(hidden_dim1, hidden_dim2)
		# self.MLP_hidden_layer3 = nn.Linear(hidden_dim2, hidden_dim3)
		self.MLP_output_layer = nn.Linear(hidden_dim2, output_dim)
		self.epoch = epoch
		self.lr = lr
		self.input_dim = input_dim
		self.hidden1 = hidden_dim1
		self.hidden2 = hidden_dim2
		# self.hidden3 = hidden_dim3
		self.output_dim = output_dim

	def fit(self,train_x,train_y):
		train_x = torch.Tensor(np.array(train_x)).to(device)  # 防止存在pandas的类型的数据
		train_y = torch.Tensor(np.array(train_y)).to(device)  # 防止存在pandas的类型的数据
		# 初始化优化函数
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		# 初始化损失函数
		loss_function = torch.nn.BCEWithLogitsLoss()
		# 训练网络
		for i in range(self.epoch):
			t = time.time()
			pre = self.model.fit(train_x)
			loss = loss_function(pre.view(-1), train_y.view(-1))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# 打印损失
			# print("Epoch:", '%04d' % (i + 1), "train_loss=", "{:.5f}".format(loss.item()),
			# 	  "time=", "{:.5f}".format(time.time() - t))
		return self

	def predict(self,test_x):
		pre_x = torch.Tensor(np.array(test_x)).to(device)  # 防止存在pandas的类型的数据
		pre = self.model.predict_proba(pre_x)[:,1]
		pre[pre > 0.5] = 1
		pre[pre < 0.5] = 0
		return pre.tolist()

	def predict_proba(self, test_x):
		pre_x = torch.Tensor(np.array(test_x)).to(device)  # 防止存在pandas的类型的数据
		pre = self.model.predict_proba(pre_x)
		return pre



