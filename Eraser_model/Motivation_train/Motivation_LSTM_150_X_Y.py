'''
进行LSTM进行意图识别————训练
X_Y_150
'''
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import time
import os
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # 标准化库
from sklearn.preprocessing import MinMaxScaler # 导入归一化库


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

# 定义轨迹数据集类
class TrajectoryDataset(Dataset):
    def __init__(self, root_dir, mode):
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.sequences = [(self.root_dir / x).absolute() for x in os.listdir(self.root_dir)]
        self.obs_len = 150
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = pd.read_csv(self.sequences[idx]) # 读取第{idx}个csv文件
        veh_id = sequence['id']
        array_elements = list(set(veh_id))
        # # 使用torch.randperm()生成一个随机排列的索引
        # random_index = torch.randperm(len(array_elements))[0]
        # # 使用随机索引获取随机元素
        # andom_element = array_elements[random_index]
        # x_vehID = sequence[sequence['id']== andom_element]
        for num in array_elements: # 使用列表进行遍历
            if ((sequence['id']==num) ).any():

                x_vehID = sequence[sequence['id']== num]
                # 第id辆车的信息
                agent_x = x_vehID['x']
                agent_y = x_vehID['y']
                agent_Turn_Left = x_vehID['Turn_Left'].unique()
                agent_Turn_Right = x_vehID['Turn_Right'].unique()
                agent_Straight = x_vehID['Straight'].unique()

                motivation = [agent_Straight , agent_Turn_Left , agent_Turn_Right]

                # 转置后，每个子列表将成为一个列向量
                motivation_1 = np.transpose(motivation)
                # motivation 为一行三列的数据
                agent_traj = np.column_stack((agent_x, agent_y))
                traj = agent_traj[:self.obs_len] # 取前150行数据
                '''
                对轨迹数据进行标准化
                '''
                scaler = StandardScaler() # 标准化
                traj = scaler.fit_transform(traj) # 一步达成结果
                
                # 将numpy数组转换为PyTorch张量
                traj = torch.tensor(traj, dtype=torch.float)
                # '''
                # 对轨迹数据进行标归一化
                # '''
                # scaler = MinMaxScaler() # 归一化
                # agent_traj = scaler.fit_transform(agent_traj) # 一步达成结果

                return traj , motivation_1
            
def get_dataset(modes):
    return (TrajectoryDataset(DATASET_PATH[mode], mode) for mode in modes)

# 定义LSTM网络模型
class TimeSeriesClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim=256, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)  # output_size classes

    def forward(self, x):
        x, _ = self.lstm(x)  # LSTM层
        x = x[:, -1, :]  # 只取LSTM输出中的最后一个时间步
        x = self.fc(x)  # 通过一个全连接层
        return x
    
# 加载数据集    
BATCH_SIZE = 10 # 处理的是轨迹条数
train_data, val_data, test_data = get_dataset(["train", "val", "test"])
print(len(train_data))
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# 设备设置
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 使用GPU或CPU

# 模型实例化
num_class = 3
n_features = 2  # 根据你的特征数量进行调整
model = TimeSeriesClassifier(n_features=n_features, output_size=num_class)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(dev)

# 学习率和优化策略
learning_rate = 0.001 # 学习率
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=5e-4)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)  # 设置学习率下降策略

# 设置训练参数
epochs = 100  # 训练轮数，根据需要进行调整
# ----------------------------------------------------#
#   训练
# ----------------------------------------------------#
# 训练模型
print("start training...")
best_acc = 0.0
# 存储每个epoch的loss
epoch_losses = []
for epoch in range(epochs):
    model.train()  # 将模型设置为训练模式
    train_epoch_loss = []
    train_epoch_accuracy = []
    epoch_loss = 0.0
    pbar = tqdm(train_loader, total=len(train_loader))
    for index, (inputs, labels) in enumerate(pbar, start=1):
        # 获取输入数据和目标，并将它们转移到GPU（如果可用）
        inputs = inputs.to(dev)
        labels = labels.float()
        labels = labels.to(dev)

        
        # 前向传播
        outputs = model(inputs)
        # pred = outputs.unsqueeze(1) # 将outputs转为labels一样的维度
        labels_1 = labels.squeeze(1) # 将labels转为outputs一样的维度

        # print(outputs)
        # # print(pred)
        # print(labels)
        # print(labels_1)
        # break

        # 计算损失及反向传播
        loss = criterion(outputs, labels_1)
        # print(loss)
        

        # 清零梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录每个epoch的loss
        epoch_loss += loss.item()

        if index%5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

    # 计算每个epoch的平均loss
    epoch_loss /= len(train_loader)
    # 记录每个epoch的loss
    epoch_losses.append(epoch_loss) 
    model.eval() # 将模型设置为评估模式
    model_dir = "saved_model/Motivation_LSTM_150_X_Y"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_dir + "/LSTM_epoch{}".format(epoch+1))        
        
# 绘图
print(f'Train CE Loss最小为{min(epoch_losses)}')

plt.rcParams.update({'font.size': 12})  # 设置图例字体大小
plt.plot(range(1, epochs+1), epoch_losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Train Cross Entropy')
plt.title('LSTM Training Loss vs. Epoch frame=150')

plt.grid(True)
plt.show()