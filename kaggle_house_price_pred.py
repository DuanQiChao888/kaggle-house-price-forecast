import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from d2l import torch as d2l
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / '01_Data' / '02_DataSet_Kaggle_House'

train_path = DATA_DIR / 'kaggle_house_pred_train.csv'
test_path = DATA_DIR / 'kaggle_house_pred_test.csv'

if not train_path.exists() or not test_path.exists():
    train_path = DATA_DIR / 'kaggle_house_pred' / 'train.csv'
    test_path = DATA_DIR / 'kaggle_house_pred' / 'test.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
print(train_data.shape)

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
#若无法获得测试数据集，可根据训练数据计算特征的均值和标准差，并使用这些统计数据来标准化训练数据。
number_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[number_features] = all_features[number_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)
#在标准化之后，缺失值被替换为0。
all_features = all_features.fillna(0)

#ummy_na = true将na(缺失值)视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True).astype(np.float32)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

loss = nn.MSELoss()
in_features = train_features.shape[1]

class FeatureAttentionNet(nn.Module):
    def __init__(self, num_features, hidden_dim=128):
        super().__init__()
        self.feature_embed = nn.Linear(1, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, X):
        # X: (batch_size, num_features)
        tokens = self.feature_embed(X.unsqueeze(-1))  # (batch_size, num_features, hidden_dim)
        K = self.key(tokens)  # (batch_size, num_features, hidden_dim)
        V = self.value(tokens)  # (batch_size, num_features, hidden_dim)
        q = self.query.unsqueeze(0).unsqueeze(1)  # (1, 1, hidden_dim)
        scores = torch.matmul(q, K.transpose(1, 2)) / math.sqrt(K.shape[-1])
        attn = torch.softmax(scores, dim=-1)  # (batch_size, 1, num_features)
        weighted = torch.matmul(attn, V).squeeze(1)  # (batch_size, hidden_dim)
        weighted = self.norm(weighted)
        return self.output(weighted)

    def get_feature_importance(self, X):
        tokens = self.feature_embed(X.unsqueeze(-1))
        K = self.key(tokens)
        q = self.query.unsqueeze(0).unsqueeze(1)
        scores = torch.matmul(q, K.transpose(1, 2)) / math.sqrt(K.shape[-1])
        return torch.softmax(scores, dim=-1).squeeze(1)


def get_net():
    return FeatureAttentionNet(in_features, hidden_dim=128)

def log_rmse(net, features, labels):
    #为了在取对数后得到更大的惩罚，使用clamp函数将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()

# 在 GPU 上训练模型，如果有多个 GPU，则使用 DataParallel 包装模型。
if torch.cuda.device_count() > 1:
    print(f'使用 {torch.cuda.device_count()} 个GPU进行训练')
    net = nn.DataParallel(get_net())
else:
    net = get_net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
train_features, train_labels, test_features = (
    train_features.to(device), train_labels.to(device), test_features.to(device)
)
def train(net, train_features, train_labels, test_features, test_labels,
            num_epochs, learning_rate, weight_decay, batch_size):
        net = net.to(device)
        train_ls, test_ls = [], []
        dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                     weight_decay=weight_decay)
        for epoch in range(num_epochs):
            for X, y in train_iter:
                optimizer.zero_grad()
                l = loss(net(X), y)
                l.backward()
                optimizer.step()
            train_ls.append(log_rmse(net, train_features, train_labels))
            if test_labels is not None:
                test_ls.append(log_rmse(net, test_features, test_labels))
        return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                    weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'])
        print(f'折{i + 1}，训练 rmse {float(train_ls[-1]):f}, '
              f'验证 rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 5, 1000, 5, 0, 128
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs,
                        lr, weight_decay, batch_size)
print(f'{k}-折交叉验证: 平均训练 rmse: {float(train_l):f},'
       f'平均验证 rmse: {float(valid_l):f}')

#训练模型并预测
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(list(range(1, num_epochs + 1)), [train_ls], xlabel='epoch',
             ylabel='rmse', xlim=[1, num_epochs], legend=['train'])
    plt.show()
    print(f'训练 rmse: {float(train_ls[-1]):f}')
    net.eval()
    preds = net(test_features).cpu().detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = test_data[['Id', 'SalePrice']]
    submission.to_csv('submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
