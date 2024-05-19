'''
进行意图识别测试 — — CNN
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
        # for num in array_elements: # 使用列表进行遍历
        #     if ((sequence['id']==num) ).any():
        # 使用torch.randperm()生成一个随机排列的索引
        random_index = torch.randperm(len(array_elements))[0]
        # 使用随机索引获取随机元素
        andom_element = array_elements[random_index]

        x_vehID = sequence[sequence['id']== andom_element]
        # 第id辆车的信息
        agent_x = x_vehID['x']
        agent_y = x_vehID['y']
        agent_Turn_Left = x_vehID['Turn_Left'].unique()
        agent_Turn_Right = x_vehID['Turn_Right'].unique()
        agent_Straight = x_vehID['Straight'].unique()

        motivation = [agent_Straight , agent_Turn_Left , agent_Turn_Right]

        # motivation_labels = torch.tensor(
        #     motivation.values, dtype=torch.float32)

        # 转置后，每个子列表将成为一个列向量
        motivation_1 = np.transpose(motivation)
        # motivation 为一行三列的数据

        input_data = torch.tensor(agent_x[:self.obs_len].values, dtype=torch.float32)
        input_data = input_data.unsqueeze(1)  # 添加维度以匹配卷积层期望的输入形状

        input_data_2 = torch.tensor(agent_y[:self.obs_len].values, dtype=torch.float32)
        input_data_2 = input_data_2.unsqueeze(1)  # 添加维度以匹配卷积层期望的输入形状

        # 合并两列数据，以便作为模型的输入
        tensor_data = torch.cat((input_data, input_data_2), dim=1)

        # agent_traj = np.column_stack((agent_x, agent_y)).astype(np.float32)
        '''
        对轨迹数据进行标准化
        '''
        # scaler = StandardScaler() # 标准化
        # tensor_data = scaler.fit_transform(tensor_data) # 一步达成结果

        # traj = agent_traj[:self.obs_len] # 轨迹数据
        # 将数据转换为张量
        # traj_input = torch.tensor(traj.values, dtype=torch.float32)

        # '''
        # 对轨迹数据进行标归一化
        # '''
        # scaler = MinMaxScaler() # 归一化
        # agent_traj = scaler.fit_transform(agent_traj) # 一步达成结果

        return  tensor_data , motivation_1
            
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

# 搭建卷积神经网络
class CNN(nn.Module):
    def __init__(self) :
        super(CNN,self).__init__()
        self.conv = nn.Conv1d(in_channels=150, out_channels=3, kernel_size=2)  
        # 2 输入通道，3 输出通道，卷积核大小为 3
        
        

    def forward(self , x):
        x = self.conv(x)  # 在第 0 维添加一个 batch 维度
        return x.view(-1, 3)  # 将输出张量 reshape 成一行三列的形状




BATCH_SIZE = 1 # 处理的是轨迹条数
train_data, val_data, test_data = get_dataset(["train", "val", "test"])
print(len(train_data))
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 使用GPU或CPU

num_input = 150*2
num_hidden = 128
num_class = 3

epochs = 100
# 初始化函数
model = CNN().to(dev)

# 定义损失函数(交叉熵)
lossF = nn.CrossEntropyLoss()

# 存储每个epoch的loss
epoch_losses = []

# 将模型切换到评估模式，这会关闭诸如Dropout这样的训练时行为
model.eval()
loaded_params = torch.load(
    'D:\\Eraser_python\\pre_trajectory_for_vehicles\\saved_model\\Motivation_CNN_150frame_x_x\\CNN_epoch20')  

# 选择性加载参数
model.load_state_dict(loaded_params, strict=False)
# 测试模型
print("start testing...")
with torch.no_grad():
    correct_num = 0
    total_num = 0
    for inputs , labels in train_loader:
        inputs = inputs.to(dev)
        labels = labels.view(labels.size(0), -1)
        labels = labels.float()
        labels = labels.to(dev)

        outputs = model(inputs) # 生成一个一维的概率分布
        outputs = torch.softmax(outputs, dim=-1)
        # 找到每行概率最高的索引
        max_indices = torch.argmax(outputs, dim=-1)

        # 创建一个与outputs相同大小的零张量
        outputs = torch.zeros_like(outputs)

        # 将每行最高概率对应的位置置为1
        outputs.scatter_(1, max_indices.view(-1, 1), 1)

        true_count = torch.sum(outputs == labels).item()
        if true_count == 3:
            true_count = 1
        else:
            true_count = 0
        
        correct_num += true_count
        print(correct_num)
        
        total_num += (outputs.size(0))
        print(total_num)
        
        # break

print(f'测试精度为：{correct_num/total_num*100}%')


 