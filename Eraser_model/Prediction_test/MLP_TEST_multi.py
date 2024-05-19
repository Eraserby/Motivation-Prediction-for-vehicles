'''
Multi layer MLP进行轨迹预测测试
'''
import traceback
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as npMotivation_Data
import random
import time
import os
import sys

import matplotlib.pyplot as plt

import os
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 进行数据集的划分（训练/验证/测试）

ROOT_DIR = "D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/Motivation_Data"
TRAIN = ROOT_DIR + "/train/"
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
        # 使用torch.randperm()生成一个随机排列的索引
        random_index = torch.randperm(len(array_elements))[0]
        # 使用随机索引获取随机元素
        andom_element = array_elements[random_index]

        x_vehID = sequence[sequence['id']== andom_element]
        # 第id辆车的信息
        agent_x = x_vehID['x']
        agent_y = x_vehID['y']
        agent_traj = np.column_stack((agent_x, agent_y)).astype(np.float32)
        '''
        np.column_stack((agent_x, agent_y)) 将x和y坐标向量相加为两列

        '''
        obs_traj = agent_traj[:self.obs_len]  # 观测窗口内的轨迹
        pred_traj_gt = agent_traj[self.obs_len:self.obs_len + 50]  # 真实未来轨迹

        return obs_traj, pred_traj_gt

        '''返回输入轨迹以及预测轨迹:
        agent_traj[:self.obs_len]为观察窗口内的代理人轨迹信息
        agent_traj[self.obs_len:self.obs_len+50]为从第 self.obs_len 行开始后50个步长的部分,
        表示了预测窗口内的代理人轨迹信息
        '''
               

def get_dataset(modes):
    return (TrajectoryDataset(DATASET_PATH[mode], mode) for mode in modes)
    # 根据 train_data, val_data, test_data = get_dataset(["train", "val", "test"])
    # 循环遍历数据


dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 使用GPU或CPU加速
BATCH_SIZE = 1 # 处理的是轨迹条数
train_data, val_data, test_data = get_dataset(["train", "val", "test"])
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

class MLP(nn.Module):
    """
    期望输入为(batch_size, 50,2)
    50:输入序列长度
    2:输入特征的维度(x和y)
    输出形状:(batch_size, 50 * 2)

    三层MLP,dropout=0.2,隐藏层为128/256/128个单元

    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(50 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 50 * 2)
            )
    
    def forward(self, x):
        # 将 (batch_size, 50, 2) 转换为 (batch_size, 50 * 2)
        x = x.view(x.size(0), -1)

        '''
        x.size(0) 表示取张量 x 在第一个维度【通常是批量大小batch size】上的元素数量。
        -1 是一个特殊的值,告诉PyTorch自动计算该维度的大小,以便让整个张量的元素总数保持不变。
        '''
        x = self.layers(x) # 将执行搭好的网络层，执行前向传播后赋值给x
        return x


# 加载模型参数
model = MLP()


loaded_params = torch.load(
    'D:\\Eraser_python\\pre_trajectory_for_vehicles\\saved_model\\MLP_multi\\MLP_epoch99')  
# 将模型切换到评估模式，这会关闭诸如Dropout这样的训练时行为
model.eval()
# 选择性加载参数
model.load_state_dict(loaded_params, strict=False)
# 当strict=False时，模型允许存在未加载的参数或多余的参数，
# 这对于迁移学习特别有用，您可以在新模型中保留或忽略某些层的参数。

model.to(dev)
loss_fn = nn.MSELoss()

print("start testing...")


# 初始化测试损失和RMSE损失的累积变量
test_loss = 0.0
RMSE_test_loss = 0.0

def visualize_trajectories(predictions, ground_truths, ids=None):
    """
    可视化预测轨迹与实际轨迹
    """
    plt.figure(figsize=(15, 10))
    for his_gt, pred, gt, idx in zip(history_truths, predictions, ground_truths, ids):

        plt.plot(pred[:, 0], pred[:, 1], '-o', label=f'Predicted Traj {idx}', alpha=0.5)
        plt.plot(gt[:, 0], gt[:, 1], '-x', label=f'Actual Traj {idx}', alpha=0.9)
        plt.plot(his_gt[:, 0], his_gt[:, 1], '-x', label=f'Actual history Traj ', alpha=0.9 )
        break
    plt.legend()
    plt.title('Predicted vs Actual Trajectories')
    plt.xlabel('X coordinate (m)')
    plt.ylabel('Y coordinate (m)')
    plt.grid(True)
    plt.show()

# 遍历测试集中的每一个batch

# 在测试循环外添加以下代码进行可视化
if __name__ == "__main__":
    with torch.no_grad():  # 确保在这个上下文中不计算梯度，节省内存并加快速度
        history_truths , predictions, ground_truths, ids = [], [], [], []
        for i, (xb, yb) in enumerate(tqdm(test_loader)):
            # 将数据移动到与模型相同的设备上
            xb = xb.to(dev)
            yb = yb.to(dev).view(yb.size(0), -1)
            
            # 通过模型进行预测
            yb_pred = model(xb)

            # 调整形状并转到CPU上
            yb_pred_1 = yb_pred.view(-1, 2).cpu().numpy()  

            # 获取实际轨迹
            xb_true = xb.view(-1, 2).cpu().numpy() # 历史轨迹
            yb_true = yb.view(-1, 2).cpu().numpy() # 未来真实轨迹

            # 累积数据以供可视化
            predictions.append(yb_pred_1)
            ground_truths.append(yb_true)
            history_truths.append(xb_true)
            

            # 假设每个batch只有一辆车，实际情况可能需要根据batch_size调整
            ids.extend([f'vehicle_{i}' for _ in range(xb.shape[0])])

            # 计算损失
            loss = loss_fn(yb_pred, yb)
            loss_RMSE = torch.sqrt(loss)

            
            # 累积损失
            test_loss += loss.item()
            RMSE_test_loss += loss_RMSE.item()

        # 计算整个测试集的平均损失
        avg_test_loss = test_loss / len(test_loader)
        avg_RMSE_test_loss = RMSE_test_loss / len(test_loader)
        
        print(f"Test MSE Loss: {avg_test_loss:.4f}")
        print(f"Test RMSE Loss: {avg_RMSE_test_loss:.4f}")
        # 可视化
        
        visualize_trajectories(predictions, ground_truths, ids)
        

