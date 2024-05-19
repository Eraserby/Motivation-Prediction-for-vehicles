'''
此段代码是将帧数 > 100的数据进行划分<train、val、test>
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
import tqdm

PATH = 'D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/Prediction_Data/test'

# 原始文件夹路径
source_folder = PATH

# 目标文件夹路径
train_folder = 'D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/Motivation_Data/train'

val_folder = 'D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/Motivation_Data/val'

test_folder = 'D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/Motivation_Data/test'


# 获取原始文件夹中的所有文件列表
files = sorted(os.listdir(source_folder))  # 对文件列表进行排序
index = 1
# 遍历文件，并按数字顺序重命名并复制到目标文件夹
for idx, file_name in enumerate(files):
        
    if file_name.endswith('.csv'): # 如果文件名对应，则读取该文件内的数据进行划分
        source_path = os.path.join(source_folder, file_name)  # 获取该文件地址
        data = pd.read_csv(source_path)

        # 从数据中提取出laneId和id列
        lane_id = data['laneId']
        veh_id = data['id']

        array_elements = list(set(veh_id)) 
        # 使用集合的不可重复性，将每个id进行循环遍历

        train_num = math.ceil(veh_id.nunique()*0.8) # 训练集轨迹数,向上取整
        val_num = math.ceil(veh_id.nunique()*0.1) # 测试集轨迹数,向上取整
        test_num = (veh_id.nunique()-train_num-val_num) # 测试集轨迹数

        # 获取训练集的id列表
        train_ids = array_elements[index + 1 : index + 1 + train_num]
        # 训练集按照一个轨迹向后移动划分

        # 选取train_num个独立id区域内容给TRAIN_VEHICLE
        TRAIN_VEHICLE = data[data['id'].isin(train_ids)]

        # 获取验证集的id列表
        val_ids = array_elements[index + 2 + train_num : index + 2 + train_num + val_num]

        VAL_VEHICLE = data[data['id'].isin(val_ids)]

        # 获取测试集的id列表
        # 获取前index项
        test_ids_1 = array_elements[ : index]
        # 获取后前index + 2 + train_num + val_num项
        test_ids_2 = array_elements[ index + 3 + train_num + val_num : ]
        # 合并两个列表
        test_ids = test_ids_1 + test_ids_2

        TEST_VEHICLE = data[data['id'].isin(test_ids)]

        

        new_file_name = f'{index}.csv'  # 按数字顺序重命名

        # 不同数据集路径        
        train_path = os.path.join(train_folder, new_file_name)
        val_path = os.path.join(val_folder, new_file_name)
        test_path = os.path.join(test_folder, new_file_name)

        TRAIN_VEHICLE.to_csv(train_path, index=False)  # index=False 防止写入索引列
        VAL_VEHICLE.to_csv(val_path, index=False)  # index=False 防止写入索引列
        TEST_VEHICLE.to_csv(test_path, index=False)  # index=False 防止写入索引列

        index +=1