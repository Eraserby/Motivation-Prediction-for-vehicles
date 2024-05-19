import os
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 进行数据集的划分（训练/验证/测试）
ROOT_DIR = "D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/Prediction_Data"
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
        for num in array_elements: # 使用列表进行遍历
            if (sequence['id']==num).any():
                x_vehID = sequence[sequence['id']==num]
                # 第id辆车的信息
                agent_x = x_vehID['x']
                agent_y = x_vehID['y']
                agent_traj = np.column_stack((agent_x, agent_y)).astype(np.float32)
                '''
                np.column_stack((agent_x, agent_y)) 将x和y坐标向量相加为两列

                '''

                return [agent_traj[:self.obs_len], agent_traj[self.obs_len:self.obs_len+25]]

                '''返回输入轨迹以及预测轨迹:
                agent_traj[:self.obs_len]为观察窗口内的代理人轨迹信息
                agent_traj[self.obs_len:self.obs_len+25]为从第 self.obs_len 行开始后25个步长的部分,
                表示了预测窗口内的代理人轨迹信息
                '''
               

def get_dataset(modes):
    return (TrajectoryDataset(DATASET_PATH[mode], mode) for mode in modes)
    # 根据 train_data, val_data, test_data = get_dataset(["train", "val", "test"])
    # 循环遍历数据
