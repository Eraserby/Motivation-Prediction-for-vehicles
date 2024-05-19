'''
multi layer LSTM进行轨迹预测训练
'''
import traceback
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import random
import time
import os
import sys

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # 标准化库
import os
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 进行数据集的划分（训练/验证/测试）
ROOT_DIR = "D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/Motivation_Data"
TRAIN = ROOT_DIR + "/train_1/"
VAL = ROOT_DIR + "/val/"
TEST = ROOT_DIR + "/test/"
DATASET_PATH = {
        "train" : TRAIN,
        "val" : VAL,
        "test" : TEST
        }

class TrajectoryDataset(Dataset):
    
    def __init__(self, root_dir, mode):
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.sequences = [(self.root_dir / x).absolute() for x in os.listdir(self.root_dir)]
        self.obs_len = 50
        '''
        self.obs_len =  步长 
        '''
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = pd.read_csv(self.sequences[idx]) # 读取第{idx}个csv文件
        veh_id = sequence['id']
        array_elements = list(set(veh_id))
        trajectories_his = []  
        trajectories_gt = []    # 存储多条轨迹的列表
        # 使用torch.randperm()生成10个随机排列的索引
        random_index = torch.randperm(len(array_elements))[0]
        # # 使用随机索引获取随机元素
        # andom_element = array_elements[random_index]

        # for num in random_index: # 使用列表进行遍历
        #     random_element = array_elements[num]
        random_element = array_elements[random_index]
        x_vehID = sequence[sequence['id']== random_element]
        # 第id辆车的信息
        agent_x = x_vehID['x']
        agent_y = x_vehID['y']
        agent_traj = np.column_stack((agent_x, agent_y)).astype(np.float32)
        agent_traj_his = agent_traj[:self.obs_len]
        agent_traj_gt = agent_traj[self.obs_len:self.obs_len+50]
        '''
        对轨迹数据进行标准化
        '''
        scaler = StandardScaler() # 标准化
        agent_traj_his = scaler.fit_transform(agent_traj_his) # 一步达成结果
        agent_traj_gt = scaler.fit_transform(agent_traj_gt)
        # trajectories_his.append(agent_traj_his)
        # trajectories_gt.append(agent_traj_gt)
        '''
        np.column_stack((agent_x, agent_y)) 将x和y坐标向量相加为两列

        '''
        # 将numpy数组转换为PyTorch张量
        agent_traj_his = torch.tensor(agent_traj_his, dtype=torch.float)
        agent_traj_gt = torch.tensor(agent_traj_gt, dtype=torch.float)

        return agent_traj_his,agent_traj_gt

        '''返回输入轨迹以及预测轨迹:
        agent_traj[:self.obs_len]为观察窗口内的代理人轨迹信息
        agent_traj[self.obs_len:self.obs_len+50]为从第 self.obs_len 行开始后50个步长的部分,
        表示了预测窗口内的代理人轨迹信息
        '''
               

def get_dataset(modes):
    return (TrajectoryDataset(DATASET_PATH[mode], mode) for mode in modes)
    # 根据 train_data, val_data, test_data = get_dataset(["train", "val", "test"])
    # 循环遍历数据


BATCH_SIZE = 10 # 处理的是轨迹条数
train_data, val_data, test_data = get_dataset(["train", "val", "test"])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
print(len(train_data))

'''
shuffle=True 表示在每个 epoch 开始时是否对数据进行洗牌，
即打乱每个批次内样本的顺序，而不是打乱单个样本内数据点的顺序。

这有助于模型更好地学习数据的分布，并且可以提高模型的泛化能力。
num_workers=6 用于数据加载的子进程数量。
设置多个子进程可以加速数据加载过程，特别是当数据集很大时。通常建议将其设置为计算机 CPU 核心数量的 1 到 4 倍。

'''


dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 使用GPU或CPU加速

# # 定义 RMSE 函数
# def rmse(predictions, targets):
#     return torch.sqrt(nn.MSELoss()(predictions, targets))

"""
    期望输入为(batch_size, 50,2)
    50:输入序列长度
    2:输入特征的维度(x和y)
    输出形状:(batch_size, 50 * 2)

    多层LSTM,dropout=0.2,隐藏层为128个单元

    """
class MultiLayerLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=100, output_size=2, num_layers=4):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        # 创建多层LSTM
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers)

        # 添加最后一层的线性层
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # 初始化隐藏状态和记忆单元
        h_0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size)
        c_0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size)
        hidden = (h_0, c_0)

        lstm_out, hidden = self.lstm(input_seq.view(len(input_seq), 1, -1), hidden)

        # 取最后一层 LSTM 的输出
        lstm_out_last = lstm_out[-1]

        predictions = self.linear(lstm_out_last.view(1, -1))
        return predictions
# 单层LSTM
class LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=100, output_size=2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.elu = nn.ELU()
        self.linear = nn.Linear(hidden_layer_size, output_size)  # 添加线性层
        # # 初始化隐藏单元
        # self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
        #                     torch.zeros(1, 1, self.hidden_layer_size))
        
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        lstm_out = self.elu(lstm_out)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# 模型实例化

model = LSTM()
model_1 = MultiLayerLSTM()
print(model)
print(model_1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 使用Adam优化方法 学习率为0.001
loss_fn = nn.MSELoss()
model.to(dev)

epoches = 20 
epoch_times = []  # 用于存储每个epoch的训练时间
# 存储每个epoch的loss和rmseloss
epoch_losses = []
RMSE_epoch_losses = []

print("start training...")
for epoch in range(epoches):
    start_epoch = time.time()  # 记录每个epoch的开始时间
    start = time.time() # 记录每个batch的开始时间
    model.train()
    epoch_loss = 0.0
    RMSE_epoch_loss = 0.0
    for i ,(xb,yb) in enumerate(tqdm(train_loader)):
        print(xb)
        print(yb)

        xb = xb.to(dev)
        yb = yb.to(dev)
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        yb_pred = model(xb)

        
        loss = loss_fn(yb_pred, yb) # 这句代码是将对应元素作差平方再取平均
        loss_RSME = torch.sqrt(loss)

        # 反向传播 & 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录每个batch的loss/RMSEloss
        epoch_loss += loss.item()
        RMSE_epoch_loss += loss_RSME.item()
        
        if i % 20 == 0:
            print("epoch {}, round {}/{} train loss: {:.4f}".format(epoch, i, len(train_loader), loss.item()))
    
    # 计算每个epoch的平均loss/RMSEloss
    epoch_loss /= len(train_loader)
    RMSE_epoch_loss /= len(train_loader)

    # 记录每个epoch的loss/RMSEloss
    epoch_losses.append(epoch_loss)      
    RMSE_epoch_losses.append(RMSE_epoch_loss)  

    epoch_time = time.time() - start_epoch  # 计算每个epoch的训练时间
    epoch_times.append(epoch_time)  # 将每个epoch的训练时间记录在列表中
    print("Epoch {} training time: {:.2f} seconds".format(epoch, epoch_time)) 


    model.eval() # 将模型设置为评估模式
    model_dir = "saved_model/Prediction_LSTM_single_150"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + "/LSTM_epoch{}".format(epoch))

    # print("start validating...")
    # with torch.no_grad():
    #     # 假设此时yb是未知的，不需要传入

    #     # 但是应该输入一个全零张量，以免报错
    #     dummy_yb = torch.zeros_like(xb) 

    #     val_loss = sum(loss_fn(
    #         my_seq2seq(xb, dummy_yb, teacher_forcing_ratio=0), yb.to(dev)) for xb, yb in val_loader)
    # print("epoch {}, val loss: {:.4f}, time spend: {}s".format(
    #         epoch, val_loss / len(val_loader), time.time() - start))
# 绘制loss随epoch的变化图

print(f'Train MSE Loss最小为{min(epoch_losses)}')
print(f'Train RMSE Loss最小为{min(RMSE_epoch_losses)}')

plt.rcParams.update({'font.size': 12})  # 设置图例字体大小
plt.subplot( 1 , 2, 1 )
plt.plot(range(1, epoches+1), epoch_losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Train MSELoss(m^2)')
plt.title('Single LSTM Training Loss vs. Epoch')
plt.subplot( 1 , 2, 2 )
plt.plot(range(1, epoches+1), RMSE_epoch_losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Train RMSELoss(m)')
plt.title('Single LSTM Training Loss vs. Epoch')

plt.grid(True)
plt.show()