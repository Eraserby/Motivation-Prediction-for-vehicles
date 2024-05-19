'''
这段代码是为了检测highD数据集中<轨迹数、左右转数、超车回正数>
highD
'''
# 导入所需的包
import math
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
import matplotlib.pyplot as plt
import os
# # 读取数据
# path = 'D:\\Eraser_python\\pre_trajectory_for_vehicles\\Dataset\\highd-dataset-v1.0\\select_data\\1_tracks_filtered.csv'
# data = pd.read_csv(path)
# lane_id = data['laneId']
# veh_id = data['id']

CHANGE_LANE = 0
UNCHANGE_LANE = 0
turn_right = 0
turn_left = 0
turn_straight = 0
veh_num = []

num_traj = []

# # 提取车道数据
# for i in range(3000):
#     if (data['id']==(i+1)).any():
#         x_vehID = data[data['id']==i+1]
#         NUM_LANE = x_vehID['laneId'].nunique() # 计算出该车所经过的车道数
#         sorted_data = x_vehID.sort_values(by='frame') # 按照‘frame’进行排序
#         last_row_data = sorted_data['laneId'].iloc[-1]
#         first_row_data = sorted_data['laneId'].iloc[0]

#         # 记录轨迹帧数
#         num_traj.append(len(x_vehID['frame']))
 

#         if last_row_data > first_row_data:
#             turn_left += 1
#         elif last_row_data < first_row_data:
#             turn_right += 1

#         if  NUM_LANE == 1:
#             UNCHANGE_LANE += 1
        
#         else:
#             CHANGE_LANE += 1
#         lane_array = list(set(sorted_data['laneId']))
#         if len(lane_array)>2:
#             print(len(lane_array))

# turn_straight = CHANGE_LANE-(turn_left+turn_right)
        
'''
数出单条轨迹最多帧数
'''       
PATH = 'D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/select_data'
# 原始文件夹路径
source_folder = PATH
# 获取原始文件夹中的所有文件列表
files = sorted(os.listdir(source_folder))  # 对文件列表进行排序
index = 1
# 遍历文件，并按数字顺序重命名并复制到目标文件夹
for idx, file_name in enumerate(files):
    if  file_name.endswith('.csv'): # 如果文件名对应，则读取该文件内的数据进行划分 file_name == f'{index}_tracks_filtered.csv':
        source_path = os.path.join(source_folder, file_name)  # 获取该文件地址
        data = pd.read_csv(source_path)

        veh_id = data['id']
        veh_num.append(veh_id.unique())
        array_elements = list(set(veh_id)) 
        index += 1
        # 提取车道数据
        for num in array_elements:
            if (data['id']==(num)).any():
                x_vehID = data[data['id']==num]
                NUM_LANE = x_vehID['laneId'].nunique() # 计算出该车所经过的车道数
                sorted_data = x_vehID.sort_values(by='frame') # 按照‘frame’进行排序
                last_row_data = sorted_data['laneId'].iloc[-1]
                first_row_data = sorted_data['laneId'].iloc[0]

                # 记录轨迹帧数
                num_traj.append(len(x_vehID['frame']))
        

                if last_row_data > first_row_data:
                    turn_left += 1
                elif last_row_data < first_row_data:
                    turn_right += 1

                if  NUM_LANE == 1:
                    UNCHANGE_LANE += 1
                
                else:
                    CHANGE_LANE += 1
                lane_array = list(set(sorted_data['laneId']))
                # if len(lane_array)>2:
                #     print(len(lane_array))

print(f'换道轨迹数为{CHANGE_LANE}条')
print(f'不换道轨迹数为{UNCHANGE_LANE}条')
print(f'左转轨迹有{turn_left}条')
print(f'右转轨迹有{turn_right}条')
print(f'转向后回正有{turn_straight}条')
# print(f'一共有{max(veh_num)}条轨迹')
print(f'单条轨迹的最多帧数是{max(num_traj)}')
plt.plot(range(num_traj),num_traj)



# 创建新数据表格
# 包含数据内容为<id / >

# print(x_vehID['laneId'].nunique())
