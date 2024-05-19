'''
意图识别数据集创建及划分所使用代码:
本段代码主要是将highD数据集中某一id下的轨迹进行straight、left、right
意图one-hot编码
而后进行train、val、test划分
并且分别储存起来供Prediction_data.py使用
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

# 文件夹路径
train_folder = 'D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/Motivation_Data/train'

val_folder = 'D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/Motivation_Data/val'

test_folder = 'D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/Motivation_Data/test'


'''
进行数据集创建：
1. 增加<Straight/Left/Right>到后三列中
2. 借鉴Prediction_detect.py中首尾不同道和直行进行
'''


# 获取train/test/val文件夹中的所有文件列表,更改以下所有folder再运行即可
files = sorted(os.listdir(test_folder))  # 对文件列表进行排序
index = 1
# 遍历文件，并按数字顺序重命名并复制到目标文件夹
for idx, file_name in enumerate(files):
        
    if file_name.endswith('.csv'):  # 对所有.csv文件进行列名添加
        source_path = os.path.join(test_folder, file_name)  # 获取该文件地址
        data = pd.read_csv(source_path)
        

        veh_id = data['id']
        array_elements = list(set(veh_id))

        '''
        下方注释代码是添加列名所使用的
        '''
        # 定义要添加的列名
        new_columns = ['Straight', 'Turn_Left', 'Turn_Right']

        


        # 检查是否已经存在相同的列名
        existing_columns = set(data.columns)
        columns_to_add = [col for col in new_columns if col not in existing_columns]

        if columns_to_add:  # 如果有需要添加的列名
            # 创建新的 DataFrame 包含要添加的列名
            new_data = pd.concat([data, pd.DataFrame(columns=columns_to_add)], axis=1)

           
            new_file_name = f'{index}.csv'  # 按数字顺序重命名

            # 不同数据集路径        
            train_path = os.path.join(train_folder, new_file_name)
            val_path = os.path.join(val_folder, new_file_name)
            test_path = os.path.join(test_folder, new_file_name)

            new_data.to_csv(test_folder, index=False)  # index=False 防止写入索引列


            index +=1



        # # 创建新的 DataFrame 来存储更改
        # updated_data = pd.DataFrame()


        # for num in array_elements: # 使用列表进行遍历
        #     if (data['id']==num).any():
        #         x_vehID = data[data['id']==num]
        #         NUM_LANE = x_vehID['laneId'].nunique() # 计算出该车所经过的车道数
        #         sorted_data = x_vehID.sort_values(by='frame') # 按照‘frame’进行排序
        #         last_row_data = sorted_data['laneId'].iloc[-1]
        #         first_row_data = sorted_data['laneId'].iloc[0]

        #         # 添加判断条件：当判断出其中一种情况时置1，另外两项置0

        #         if last_row_data > first_row_data:
        #             x_vehID['Turn_Left'] = 1
        #             x_vehID['Turn_Right'] = 0
        #             x_vehID['Straight'] = 0
        #         elif last_row_data < first_row_data:
        #             x_vehID['Turn_Left'] = 0
        #             x_vehID['Turn_Right'] = 1
        #             x_vehID['Straight'] = 0
        #         else:
        #             x_vehID['Turn_Left'] = 0
        #             x_vehID['Turn_Right'] = 0
        #             x_vehID['Straight'] = 1

        #         # 将更改后的数据追加到 updated_data
        #         updated_data = pd.concat([updated_data, x_vehID])

        # new_file_name = f'{index}.csv'  # 按数字顺序重命名

        # # 不同数据集路径        
        # train_path = os.path.join(train_folder, new_file_name)
        # val_path = os.path.join(val_folder, new_file_name)
        # test_path = os.path.join(test_folder, new_file_name)

        # updated_data.to_csv(test_folder, index=False)  # index=False 防止写入索引列


        # index +=1

        



