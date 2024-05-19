'''
single layer LSTM进行轨迹预测训练
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
from matplotlib.pyplot import figure

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

        # for num in array_elements: # 使用列表进行遍历
        #     if (sequence['id']==num).any():

        x_vehID = sequence[sequence['id']== andom_element]
        # 第id辆车的信息
        agent_x = x_vehID['x']
        agent_y = x_vehID['y']
        agent_traj = np.column_stack((agent_x, agent_y)).astype(np.float32)
        '''
        np.column_stack((agent_x, agent_y)) 将x和y坐标向量相加为两列

        '''

        return [agent_traj[:self.obs_len], agent_traj[self.obs_len:self.obs_len+50]]

        '''返回输入轨迹以及预测轨迹:
        agent_traj[:self.obs_len]为观察窗口内的代理人轨迹信息
        agent_traj[self.obs_len:self.obs_len+50]为从第 self.obs_len 行开始后50个步长的部分,
        表示了预测窗口内的代理人轨迹信息
        '''
               

def get_dataset(modes):
    return (TrajectoryDataset(DATASET_PATH[mode], mode) for mode in modes)
    # 根据 train_data, val_data, test_data = get_dataset(["train", "val", "test"])
    # 循环遍历数据


BATCH_SIZE = 1 # 处理的是轨迹条数
train_data, val_data, test_data = get_dataset(["train", "val", "test"])
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

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

    单层LSTM,dropout=0,隐藏层为128个单元

    """
class Encoder(nn.Module):
    def __init__(self,
                 input_size = 2,
                 embedding_size = 64,
                 hidden_size = 128,
                 n_layers = 1,
                 dropout = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.linear = nn.Linear(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers,
                           dropout = dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x:输入批次数据,size:[序列长度，批次大小，特征大小］
        for the highD trajectory data, size(x) is [50, batch size, 2]
        """
        # embedded: [序列长度, batch size, embedding size]
        embedded = self.dropout(F.relu(self.linear(x)))
        # you can checkout https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM
        # for details of the return tensor
        # 简而言之，输出包含每个时间步的最后一层的输出
        # 隐藏和单元包含每个层最后一个时间步的隐藏和单元状态
        # 我们只将隐藏和单元作为输入解码器的上下文

        output, (hidden, cell) = self.lstm(embedded)

        # hidden = [n layers * n directions, batch size, hidden size]
        # cell = [n layers * n directions, batch size, hidden size]
        # 由于我们不使用双向 RNN，所以 n 方向为 1
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self,
                 output_size = 2,
                 embedding_size = 64,
                 hidden_size = 128,
                 n_layers = 1,
                 dropout = 0):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Linear(output_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, dropout = dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        """
        x : 批量输入数据,size(x): [批次大小，特征大小］
        注意 x 只有两个维度，因为输入是批次数据
        观察到的轨迹的最后一个坐标
        因此序列长度已被删除。
        """
        # 为 x 添加序列维度，以便使用 nn.LSTM
        # 之后，size(x) 将是 [1，批次大小，特征大小］
        x = x.unsqueeze(0)
        # 在张量 x 的维度之间插入一个新的维度，这个新维度的大小为 1

        # embedded = [1, batch size, embedding size]
        embedded = self.dropout(F.relu(self.embedding(x)))

        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #序列长度 and n 个方向在解码器中始终为 1, therefore:
        #output = [1, batch size, hidden size]
        #hidden = [n layers, batch size, hidden size]
        #cell = [n layers, batch size, hidden size]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        # prediction = [batch size, output size]
        prediction = self.linear(output.squeeze(0))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        # 确保encoder和decoder隐藏状态和层数一致

    def forward(self, x, y, teacher_forcing_ratio = 0.5):
        """
        x = [观测序列长度、批次大小、特征大小］
        y = [目标序列长度、批次大小、特征大小］
        对于我们的 highD 运动预测数据集
        观测序列长度为 50,目标序列长度为 50
        特征大小暂定为 2(x 和 y)

        teacher_forcing_ratio 是使用教师强迫的概率
        例如，如果 teacher_forcing_ratio 为 0.75,则 75% 的时间都会使用地面实况输入
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        """

        batch_size = x.shape[1]
        target_len = y.shape[0]

        
        # 张量，用于存储每个时间步的解码器输出
        outputs = torch.zeros(y.shape).to(self.device)
        
        # 编码器的最后一个隐藏状态被用作解码器的初始隐藏状态
        hidden, cell = self.encoder(x)

        # 解码器的第一个输入是 x 的最后一个坐标
        decoder_input = x[-1, :, :]
        
        for i in range(target_len):
            # 运行解码一个时间步
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)

            # 将预测结果放入一个张量中，其中包含每个时间步的预测结果
            outputs[i] = output

            # 决定是否使用教师强制手段
            teacher_forcing = random.random() < teacher_forcing_ratio

            # 输出与输入形状相同，[批量大小，特征大小］
            # 因此，我们可以直接将输出作为输入，也可以使用 true 标记，这取决于
            # 教师强制是否为真
            decoder_input = y[i] if teacher_forcing else output

        return outputs


# 进行encode+decode+seq2seq的类堆叠

my_encoder = Encoder().to(dev)
my_decoder = Decoder().to(dev)
my_seq2seq = Seq2Seq(my_encoder , my_decoder, dev).to(dev)

# 将模型切换到评估模式，这会关闭诸如Dropout这样的训练时行为
my_seq2seq.eval()
loaded_params = torch.load(
    'D:\\Eraser_python\\pre_trajectory_for_vehicles\\saved_model\\LSTM_single\\LSTM_epoch199')  
my_seq2seq.load_state_dict(loaded_params, strict=False)



loss_fn = nn.MSELoss()
my_seq2seq.to(dev)


print("start testing...")
# 初始化测试损失和RMSE损失的累积变量
test_loss = 0.0
RMSE_test_loss = 0.0
def visualize_trajectories(predictions, ground_truths, ids=None):
    """
    可视化预测轨迹与实际轨迹
    """
    plt.figure(figsize=(15, 10)).canvas.toolbar.pan()
    for his_gt, pred, gt, idx in zip(history_truths, predictions, ground_truths, ids):
        if (abs(max(gt[:,1]) - min(his_gt[:,1])) >  1.5).any():
            # 如果不出换道的图，这个if请注释掉
            # 并且将轨迹输出为从0开始
            plt.plot(pred[:, 0], pred[:, 1], '-o', label=f'Predicted Traj {idx}', alpha=0.5)
            plt.plot(gt[:, 0], gt[:, 1], '-x', label=f'Actual Traj {idx}', alpha=0.9)
            plt.plot(his_gt[:, 0], his_gt[:, 1], '-s', label=f'Actual history Traj ', alpha=0.9 )
            # break
            # 若是只要一条轨迹使用break
    plt.legend()
    plt.title('Predicted vs Actual Trajectories')
    plt.xlabel('X coordinate (m)')
    plt.ylabel('Y coordinate (m)')
    plt.grid(True)
    
    plt.show()

# 在测试循环外添加以下代码进行可视化
if __name__ == "__main__":
    with torch.no_grad():  # 确保在这个上下文中不计算梯度，节省内存并加快速度
        history_truths , predictions, ground_truths, ids = [], [], [], []
        for i, (xb, yb) in enumerate(tqdm(test_loader)):
            
            xb = xb.to(dev)
            yb = yb.to(dev)
            
            # 通过模型进行预测
            yb_pred = my_seq2seq(xb, xb, teacher_forcing_ratio=0.5)


            # 获取实际轨迹
            xb_true = xb.view(-1, 2).cpu().numpy() # 历史轨迹
            yb_true = yb.view(-1, 2).cpu().numpy() # 未来真实轨迹
            yb_pred_1 = yb_pred.view(-1, 2).cpu().numpy() # 预测轨迹

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


