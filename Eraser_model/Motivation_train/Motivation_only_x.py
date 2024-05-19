'''
进行意图识别检测 — — MLP

只观测 x 方向

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
                # agent_x = x_vehID['x']
                agent_y = x_vehID['y']
                agent_Turn_Left = x_vehID['Turn_Left'].unique()
                agent_Turn_Right = x_vehID['Turn_Right'].unique()
                agent_Straight = x_vehID['Straight'].unique()

                motivation = [agent_Straight , agent_Turn_Left , agent_Turn_Right]

                # 转置后，每个子列表将成为一个列向量
                motivation_1 = np.transpose(motivation)
                # motivation 为一行三列的数据
                agent_traj = np.column_stack((agent_y, agent_y)).astype(np.float32) # 相同列合并
                # agent_traj = agent_y.astype(np.float32) # 只包含x方向

                '''
                对轨迹数据进行标准化
                '''
                scaler = StandardScaler() # 标准化
                agent_traj = scaler.fit_transform(agent_traj) # 一步达成结果

                # '''
                # 对轨迹数据进行标归一化
                # '''
                # scaler = MinMaxScaler() # 归一化
                # agent_traj = scaler.fit_transform(agent_traj) # 一步达成结果

                return agent_traj[:self.obs_len] , motivation_1
            
def get_dataset(modes):
    return (TrajectoryDataset(DATASET_PATH[mode], mode) for mode in modes)
    # 根据 train_data, val_data, test_data = get_dataset(["train", "val", "test"])
    # 循环遍历数据

# 搭建全连接神经网络
class MLP(nn.Module):
    def __init__(self , num_input , num_hidden,num_class) :
        super(MLP,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_input, num_hidden),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, num_hidden*2),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden*2, num_hidden*2),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden*2, num_class)
            )
      
    
    def forward(self,x):
        x=self.layers(x)
        return x

BATCH_SIZE = 5 # 处理的是轨迹条数
train_data, val_data, test_data = get_dataset(["train", "val", "test"])
print(len(train_data))
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 使用GPU或CPU

learning_rate = 0.001 # 学习率
num_input = 150*2
num_hidden = 128
num_class = 3

epochs = 100
# 初始化函数
model = MLP(num_input,num_hidden,num_class)

# 定义损失函数(交叉熵)
lossF = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 输入和标签的维度
for batch_index , (inputs, labels) in enumerate((train_loader)):
    print(inputs.shape)
    print(labels.shape)
    break

# 存储每个epoch的loss
epoch_losses = []

# 训练模型
print("start training...")
for epoch in range(epochs):
    # model.train()
    epoch_loss = 0.0
    for batch_index , (inputs, labels) in enumerate(tqdm(train_loader)):
        # 把维度进行变换
        inputs = inputs.reshape(-1,num_input).to(dev) # batch_size x150x1 ---> batch_size x 150
        labels = labels.view(labels.size(0), -1)
        labels = labels.float()
        labels = labels.to(dev)

        outputs = model(inputs)

        # 计算损失
        loss = lossF(outputs , labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录每个epoch的loss
        epoch_loss += loss.item()

        if batch_index%10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

    # 计算每个epoch的平均loss/RMSEloss
    epoch_loss /= len(train_loader)
    # 记录每个epoch的loss/RMSEloss
    epoch_losses.append(epoch_loss) 
    model.eval() # 将模型设置为评估模式
    model_dir = "saved_model/Motivation_MLP_only_x"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_dir + "/MLP_epoch{}".format(epoch+1))


# 绘图
print(f'Train CE Loss最小为{min(epoch_losses)}')

plt.rcParams.update({'font.size': 12})  # 设置图例字体大小
plt.plot(range(1, epochs+1), epoch_losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Train Cross Entropy')
plt.title('single MLP Training Loss vs. Epoch frame=100')

plt.grid(True)
plt.show()