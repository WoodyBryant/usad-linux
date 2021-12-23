# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 10:48:27 2021

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from utils import *
from usad import *

##选择设备，如果GPU可用，则用GPU，否则用CPU
device = get_default_device()

##读取数据，丢掉时间戳、正常/异常标签
##修改这里的文件读取目录
normal = pd.read_csv(r"E:\github_desktop\usad-linux\usad-linux\input\SWaT_Dataset_Normal_v1.csv")
normal = normal.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
print("正常数据的维度:{0}".format(normal.shape))


# Transform all columns into float64
# 这里每个具体的i是指normal中的一列
#将数据中的,替换为.
for i in list(normal): 
    normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
normal = normal.astype(float)

#标准化模板
#Normalization
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
#这两种方法得到的结果一样
# x = normal.values
# x_scaled = min_max_scaler.fit_transform(x)
x_scaled = min_max_scaler.fit_transform(normal)
normal = pd.DataFrame(x_scaled)

print("正常数据的前两行:\n{0}".format(normal.head(2)))


#Attack
#Read data
#标签设置:正常为0，异常为1
#丢掉时间戳和Normal/Attack标签
##读取数据，丢掉时间戳、正常/异常标签
##修改这里的文件读取目录
attack = pd.read_csv(r"E:\github_desktop\usad-linux\usad-linux\input\SWaT_Dataset_Attack_v0.csv",sep=";")#, nrows=1000)
labels = [ float(label!= 'Normal' ) for label  in attack["Normal/Attack"]]
# labels = [ float(label!= 'Normal' ) for label  in attack["Normal/Attack"].values]
attack = attack.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
print("攻击数据的维度:{0}".format(attack.shape))
# attack.to_csv(r"E:\github_desktop\usad-linux\usad-linux\input\test_attack.csv")
# Transform all columns into float64
for i in list(attack):
    attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
attack = attack.astype(float)

#Normalization
# x = attack.values 
x_scaled = min_max_scaler.transform(attack)
attack = pd.DataFrame(x_scaled)
print("异常数据的前两行:\n{0}".format(attack.head(2)))

#Windows
#np.arrange()返回一个有固定步长的排列
#np.arange(window_size)[None, :]返回一个二维数组，只有一行
#np.arange(normal.shape[0]-window_size)[:, None]，返回一个二维数组，只有一列
#np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None]，相加时，
#分别将第一个的行和第二个的列广播扩充，两个加起来就起到了滑动窗口的效果，结果中的每一行有12个数据的索引，就是一个滑动窗口
##它这里原始的窗口数算错了，应该是数据维度-窗口维度+1
##正常窗口总的维度(494989，12，51)
window_size=12
windows_normal=normal.values[np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None]]
# print(np.arange(window_size)[None, :])
# print(np.arange(normal.shape[0]-window_size)[:, None])
# print(np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None])
print("正常窗口的总维度:{0}".format(windows_normal.shape))
windows_attack=attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None]]
print("攻击窗口的总维度:{0}".format(windows_attack.shape))



#Training
import torch.utils.data as data_utils

BATCH_SIZE =  7919
N_EPOCHS = 100
hidden_size = 100
##w_size是一个滑动窗口中的数据的总维度(窗口长度*特征数)，也是输入的维度
##z_size是？？
w_size=windows_normal.shape[1]*windows_normal.shape[2]
z_size=windows_normal.shape[1]*hidden_size

##划分正常数据的训练集和验证集,比例为0.8:0.2，训练集的维度是(395991，12，51)
windows_normal_train = windows_normal[:int(np.floor(.8 *  windows_normal.shape[0]))]
windows_normal_val = windows_normal[int(np.floor(.8 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

##torch.utils.data.DataLoader该接口主要用来将自定义的数据读取接口的输出或者
#PyTorch已有的数据读取接口的输入按照batch size封装成Tensor，后续只需要再包装成Variable即可作为模型的输入
#torch.from_numpy是将数据从numpy转换为张量
#view是改变张量的维度，正常数据训练集维度从(395991，12，51)变为(395991,12*51),即将一个窗口中的矩阵拉直。
train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = UsadModel(w_size, z_size)
##检查模型的格式
model = to_device(model,device)
##得到历次迭代中的验证集的loss1和loss2
history = training(N_EPOCHS,model,train_loader,val_loader)
plot_history(history)
##保存模型等参数，想恢复某一个阶段的训练时，就可以读取之前保存的网络模型参数
torch.save({
            'encoder': model.encoder.state_dict(),
            'decoder1': model.decoder1.state_dict(),
            'decoder2': model.decoder2.state_dict()
            }, "model.pth")

#Testing
#读取之前保存的模型参数
checkpoint = torch.load("model.pth")

model.encoder.load_state_dict(checkpoint['encoder'])
model.decoder1.load_state_dict(checkpoint['decoder1'])
model.decoder2.load_state_dict(checkpoint['decoder2'])
##异常分数
results=testing(model,test_loader)
##画ROC曲线
windows_labels=[]
for i in range(len(labels)-window_size):
    windows_labels.append(list(np.int_(labels[i:i+window_size])))
##窗口内有一个异常就是窗口
##stack沿着一个新维度对张量进行拼接
y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels ]
y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                              results[-1].flatten().detach().cpu().numpy()])
threshold=ROC(y_test,y_pred)
##打印异常分数
print(y_pred)