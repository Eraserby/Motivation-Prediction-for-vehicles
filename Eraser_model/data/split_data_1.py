'''
轨迹预测所使用代码:
本段代码主要是将highD数据集进行train、val、test划分
并且分别储存起来供prediction_data.py使用
其中训练集为任意一整段轨迹数据 <——> train:val:test = 8:1:1

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

PATH = 'D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/data'

# 原始文件夹路径
source_folder = PATH

# 目标文件夹路径
train_folder = 'D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/Prediction_Data/train'

val_folder = 'D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/Prediction_Data/val'

test_folder = 'D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/Prediction_Data/test'


# 获取原始文件夹中的所有文件列表
files = sorted(os.listdir(source_folder))  # 对文件列表进行排序
index = 1
# 遍历文件，并按数字顺序重命名并复制到目标文件夹
for idx, file_name in enumerate(files):
        
    if file_name == f'{index:02}_tracks.csv': # 如果文件名对应，则读取该文件内的数据进行划分
        source_path = os.path.join(source_folder, file_name)  # 获取该文件地址
        data = pd.read_csv(source_path)
        lane_id = data['laneId']
        veh_id = data['id']
        train_num = math.ceil(veh_id.nunique()*0.8) # 训练集轨迹数,向上取整
        val_num = math.ceil(veh_id.nunique()*0.1) # 测试集轨迹数,向上取整
        test_num = (veh_id.nunique()-train_num-val_num) # 测试集轨迹数

        TRAIN_VEHICLE = data[(index<data['id']) & (data['id']<=(train_num+index))] # 训练集按照一个轨迹向后移动划分



        VAL_VEHICLE = data[((train_num+index)<data['id']) & (data['id']<=(train_num+val_num+index))]
        TEST_VEHICLE = data[(data['id']<=index) | (data['id']>(train_num+val_num+index))]

        new_file_name = f'{index}.csv'  # 按数字顺序重命名

        # 不同数据集路径        
        train_path = os.path.join(train_folder, new_file_name)
        val_path = os.path.join(val_folder, new_file_name)
        test_path = os.path.join(test_folder, new_file_name)

        TRAIN_VEHICLE.to_csv(train_path, index=False)  # index=False 防止写入索引列
        VAL_VEHICLE.to_csv(val_path, index=False)  # index=False 防止写入索引列
        TEST_VEHICLE.to_csv(test_path, index=False)  # index=False 防止写入索引列

        index +=1
        

        
    
#     # 检查是否是文件，避免处理目录
#     if os.path.isfile(source_path):
#         new_file_name = f'{idx + 1}.txt'  # 按数字顺序重命名
#         destination_path = os.path.join(destination_folder, new_file_name)  # 目标文件的路径
#         # 复制文件
#         shutil.copy2(source_path, destination_path)

# print(f"所有文件已复制并按数字顺序重命名到文件夹: {destination_folder}")



