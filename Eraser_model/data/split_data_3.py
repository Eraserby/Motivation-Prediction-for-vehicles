'''
为了做意图识别
将数据集中的<左转/右转/直行>
单独拿出来放在一个csv文件中
'''
import pandas as pd
import os
import shutil
import math
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
import matplotlib.pyplot as plt

# 文件夹路径
source_folder = 'D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/Motivation_Data/train'

train_folder = 'D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/Motivation_Data/train_1'
index = 60 
file_name = f'{index}.csv'
source_path = os.path.join(source_folder, file_name)
# files = sorted(os.listdir(train_folder))  # 对文件列表进行排序

# # 遍历文件，并按数字顺序重命名并复制到目标文件夹
# for idx, file_name in enumerate(files):
        
# if file_name.endswith('.csv'):  # 对所有.csv文件进行列名添加
#     source_path = os.path.join(train_folder, file_name)  # 获取该文件地址

data = pd.read_csv(source_path)


veh_id = data['id']

array_elements = list(set(veh_id))
agent_Turn_Left = data['Turn_Left']
agent_Turn_Right = data['Turn_Right']
agent_Straight = data['Straight']

'''
下方代码将不同状态分为不同csv文件
'''

if (agent_Straight == 1).any():  # 直行数据集
    STRAIGHT = data[data['Straight'] == 1]
    
    new_file_name = f'{3*index-2}.csv'  # 按数字顺序重命名

    # 不同数据集路径        
    train_path = os.path.join(train_folder, new_file_name)
    
    STRAIGHT.to_csv(train_path, index=False)  # index=False 防止写入索引列

if (agent_Turn_Left == 1).any():  # 左转数据集
    LEFT = data[data['Turn_Left'] == 1]
    
    new_file_name = f'{3*index-1}.csv'  # 按数字顺序重命名

    # 不同数据集路径        
    train_path = os.path.join(train_folder, new_file_name)
    
    LEFT.to_csv(train_path, index=False)  # index=False 防止写入索引列

if (agent_Turn_Right == 1).any():  # 直行数据集
    RIGHT = data[data['Turn_Right'] == 1]
    
    new_file_name = f'{index*3}.csv'  # 按数字顺序重命名

    # 不同数据集路径        
    train_path = os.path.join(train_folder, new_file_name)
    
    RIGHT.to_csv(train_path, index=False)  # index=False 防止写入索引列