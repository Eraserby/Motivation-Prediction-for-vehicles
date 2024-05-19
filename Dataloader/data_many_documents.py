import os
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 进行数据集的划分（训练/验证/测试）
ROOT_DIR = "E:/毕业设计/data set/Argoverse/"
TRAIN = ROOT_DIR + "train/data"
VAL = ROOT_DIR + "val/data"
TEST = ROOT_DIR + "test/data/"
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
        self.obs_len = 20
        '''
        self.obs_len = 20 步长为20 
        '''
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = pd.read_csv(self.sequences[idx])
        agent_x = sequence[sequence["OBJECT_TYPE"] == "AGENT"]["X"]
        agent_y = sequence[sequence["OBJECT_TYPE"] == "AGENT"]["Y"]
        agent_traj = np.column_stack((agent_x, agent_y)).astype(np.float32)
        '''
        np.column_stack((agent_x, agent_y)) 将x和y坐标向量相加为两列

        '''

        return [agent_traj[:self.obs_len], agent_traj[self.obs_len:]]
    
        '''返回输入轨迹以及预测轨迹:
        agent_traj[:self.obs_len]为观察窗口内的代理人轨迹信息
        agent_traj[self.obs_len:]为从第 self.obs_len 行开始到最后一行的部分，
        表示了预测窗口内的代理人轨迹信息
        '''
               

def get_dataset(modes):
    return (TrajectoryDataset(DATASET_PATH[mode], mode) for mode in modes)
