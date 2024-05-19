'''
此段代码将数据集帧数>100的数据筛选出来
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

# select_folder = 'D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/select_data'

# train_folder = 'D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/Motivation_Data/train'

# val_folder = 'D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/Motivation_Data/val'

test_folder = 'D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/Motivation_Data/test'
# 获取原始文件夹中的所有文件列表
files = sorted(os.listdir(source_folder))  # 对文件列表进行排序
index = 1
# 遍历文件，并按数字顺序重命名并复制到目标文件夹
for idx, file_name in enumerate(files):
        
    if file_name == f'{idx+1}.csv': # 如果文件名对应，则读取该文件内的数据进行划分
        source_path = os.path.join(source_folder, file_name)  # 获取该文件地址
        data = pd.read_csv(source_path)
        
        # 计算每个 id 的最大帧数和最小帧数之间的差值
        frame_diff = data.groupby('id')['frame'].apply(lambda x: x.max() - x.min())
        
        # 筛选出差值大于 100 的 id
        valid_ids = frame_diff[frame_diff >= 100].index
        
        # 根据筛选的 id，保留对应的数据行
        filtered_data = data[data['id'].isin(valid_ids)]

        # 保存过滤后的数据到目标文件夹，并按数字顺序重命名
        target_file_name = f"{index}.csv"
        target_path = os.path.join(test_folder, target_file_name)
        filtered_data.to_csv(target_path, index=False)       

        index +=1



    


# train_id = TRAIN_VEHICLE['id']
#         array_train = list(set(train_id))

#         for train_num in array_train:
#             num_train = train_id[train_id['id']==train_num]
#             train = len(num_train['frame'])
#             if train >= 100:
#                 TRAIN_VEHICLE_updata = 