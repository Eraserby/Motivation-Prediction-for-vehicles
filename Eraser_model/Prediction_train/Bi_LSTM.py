# %%
# code by Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
        input_batch = agent_traj[:self.obs_len]
        target_batch = agent_traj[self.obs_len:self.obs_len+50]
        return input_batch , target_batch

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

class BiLSTM(nn.Module):
    def __init__(self,
                 input_size = 2,
                 embedding_size = 128,
                 hidden_size = 256,
                 n_layers = 4,
                 dropout = 0.2):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.linear = nn.Linear(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers,
                           dropout = dropout,bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        # bidirectional=True表示创建一个双向 LSTM

        # hidden_size=n_hidden LSTM 层中隐藏状态的维度大小，也就是 LSTM 单元的数量
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, 
                            bidirectional=True)
        self.W = nn.Linear(n_hidden * 2, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        input = X.transpose(0, 1)  # input : [n_step, batch_size, n_class]

        hidden_state = torch.zeros(1*2, len(X), n_hidden)   # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.zeros(1*2, len(X), n_hidden)     # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model

if __name__ == '__main__':
    n_hidden = 5 # LSTM 层中隐藏状态的维度大小，也就是 LSTM 单元的数量

    sentence = (
        'Lorem ipsum dolor sit amet consectetur adipisicing elit '
        'sed do eiusmod tempor incididunt ut labore et dolore magna '
        'aliqua Ut enim ad minim veniam quis nostrud exercitation'
    )

    word_dict = {w: i for i, w in enumerate(list(set(sentence.split())))}
    number_dict = {i: w for i, w in enumerate(list(set(sentence.split())))}
    n_class = len(word_dict)
    max_len = len(sentence.split())

    model = BiLSTM()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    

    input_batch = torch.FloatTensor(input_batch)
    # torch.FloatTensor(input_batch)：
    #这句代码将名为 input_batch 的输入数据转换为 PyTorch 的 FloatTensor (浮点数)
    target_batch = torch.FloatTensor(target_batch)

    # Training
    for epoch in range(10000):
        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print(sentence)
    print([number_dict[n.item()] for n in predict.squeeze()])