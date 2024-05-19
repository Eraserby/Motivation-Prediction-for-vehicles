# '''
# Mirror-Traffic—清华
# '''
# # 导入所需的包
# import math
# import pandas as pd
# import numpy as np
# from scipy.signal import savgol_filter
# import matplotlib
# import matplotlib.pyplot as plt

# # 读取数据
# path = 'D:\\Eraser_python\\pre_trajectory_for_vehicles\\Dataset\\Mirror-Traffic—清华\\Expressway-merge-in\\Expressway-merge-in\\Trajectory.csv'
# data = pd.read_csv(path)
# lane_id = data['laneId']
# veh_id = data['trackId']

# '''多项式拟合'''
# plt.rcParams.update({'font.size': 10})  # 设置图例字体大小


# # # Savitzky-Golay 滤波器
# # y_smooth = savgol_filter(local_y, window_length=23, polyorder=7)
# # plt.plot(local_x ,y_smooth , color='green' ,label='S-G filter')
# # # 多项式拟合
# # coefficients = np.polyfit(local_x, local_y, 6)  # 数字表示n次多项式拟合
# # p_poly = np.poly1d(coefficients)
# # y_poly = p_poly(local_x)
# # plt.scatter(local_x,local_y, s=100  , label='original data')
# # plt.plot(local_x , y_poly , color='red' ,label='6th order polynomial fit')




# CHANGE_LANE = 0
# UNCHANGE_LANE = 0

# # 提取车道数据
# plt.subplot( 1 , 2, 1 )
# for i in range(3000):
#     if (data['trackId']==(i+1)).any():
#         x_vehID = data[data['trackId']==i+1]
#         NUM_LANE = x_vehID['laneId'].nunique()
#         sorted_data = x_vehID.sort_values(by='frameId')
#         plt.title(f'UNCHANGED LANE TRAJECTORY')
#         plt.ylabel('Lateral distance(m)')
#         plt.xlabel('Longitudinal distance(m)')
#         if  NUM_LANE == 1:
#             UNCHANGE_LANE += 1
#             plt.plot(sorted_data['localY'],sorted_data['localX'])
#         else:
#             CHANGE_LANE += 1

# plt.subplot( 1 , 2, 2 )
# for i in range(3000):
#     if (data['trackId']==(i+1)).any():
#         x_vehID = data[data['trackId']==i+1]
#         NUM_LANE = x_vehID['laneId'].nunique()
#         sorted_data = x_vehID.sort_values(by='frameId')
#         if  NUM_LANE != 1:
#             plt.ylabel('Lateral distance(m)')
#             plt.xlabel('Longitudinal distance(m)')
#             plt.title(f'CHANGED LANE TRAJECTORY')
#             plt.plot(sorted_data['localY'],sorted_data['localX'])

# print(f'换道轨迹数为{CHANGE_LANE}条')
# print(f'不换道轨迹数为{UNCHANGE_LANE}条')
# print(f'一共有{veh_id.nunique()}条轨迹')
# # print(x_vehID['laneId'].nunique())

# plt.tight_layout()  # 自动调整子图布局，避免重叠   
# plt.show()


'''
highD
'''
# 导入所需的包
import math
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
import matplotlib.pyplot as plt

# 读取数据
path = 'D:\\Eraser_python\\pre_trajectory_for_vehicles\\Dataset\\highd-dataset-v1.0\\Motivation_Data\\train\\60.csv'
data = pd.read_csv(path)
lane_id = data['laneId']
veh_id = data['id']

'''多项式拟合'''
plt.rcParams.update({'font.size': 14})  # 设置图例字体大小


# # Savitzky-Golay 滤波器
# y_smooth = savgol_filter(local_y, window_length=23, polyorder=7)
# plt.plot(local_x ,y_smooth , color='green' ,label='S-G filter')
# # 多项式拟合
# coefficients = np.polyfit(local_x, local_y, 6)  # 数字表示n次多项式拟合
# p_poly = np.poly1d(coefficients)
# y_poly = p_poly(local_x)
# plt.scatter(local_x,local_y, s=100  , label='original data')
# plt.plot(local_x , y_poly , color='red' ,label='6th order polynomial fit')

CHANGE_LANE = 0
UNCHANGE_LANE = 0
array_elements = list(set(veh_id)) # 使用集合的不可重复性，将每个id进行循环遍历
# 提取车道数据
plt.subplot( 1 , 2, 1 )
for i in array_elements:
    if (data['id']==i).any():
        x_vehID = data[data['id']==i]
        NUM_LANE = x_vehID['laneId'].nunique() # 计算出该车所经过的车道数
        sorted_data = x_vehID.sort_values(by='frame') # 按照‘frame’进行排序

        plt.title(f'UNCHANGED LANE TRAJECTORY')
        plt.ylabel('Lateral distance(m)',fontdict={'fontsize': 14})
        plt.xlabel('Longitudinal distance(m)',fontdict={'fontsize': 14})

        if  NUM_LANE == 1:
            UNCHANGE_LANE += 1
            x_data_1 = sorted_data['x']
            y_data_1 = sorted_data['y']
            plt.plot(x_data_1 , y_data_1)
        else:
            CHANGE_LANE += 1
        

plt.subplot( 1 , 2, 2 )
for i in range(3000):
    if (data['id']==(i+1)).any():
        x_vehID = data[data['id']==i+1]
        NUM_LANE = x_vehID['laneId'].nunique()
        sorted_data = x_vehID.sort_values(by='frame')
        if  NUM_LANE != 1:
            plt.ylabel('Lateral distance(m)',fontdict={'fontsize': 14})
            plt.xlabel('Longitudinal distance(m)',fontdict={'fontsize': 14})
            plt.title(f'CHANGED LANE TRAJECTORY')
            x_data_2 = sorted_data['x']
            y_data_2 = sorted_data['y']
            plt.plot(x_data_2 , y_data_2)

print(f'换道轨迹数为{CHANGE_LANE}条')
print(f'不换道轨迹数为{UNCHANGE_LANE}条')
print(f'一共有{veh_id.nunique()}条轨迹')
print(f'一共有{len(array_elements)}条轨迹')
# print(x_vehID['laneId'].nunique())

plt.tight_layout()  # 自动调整子图布局，避免重叠   
plt.show()




# '''
# Argoverse
# '''
# # 导入所需的包
# import math
# import pandas as pd
# import numpy as np
# from scipy.signal import savgol_filter
# import matplotlib
# import matplotlib.pyplot as plt

# # 读取数据
# path = 'E:\\毕业设计\\data set\\Argoverse\\train\\data\\2.csv'
# data = pd.read_csv(path)
# OBJECT = data['OBJECT_TYPE']
# OBJECT_TYPE = data[data['OBJECT_TYPE'] == 'AGENT']

# # 选择一种类型进行绘图<AV AGENT OTHERS>


# '''多项式拟合'''
# plt.rcParams.update({'font.size': 14})  # 设置图例字体大小
# local_x = OBJECT_TYPE['X']
# local_y = OBJECT_TYPE['Y']

# # Savitzky-Golay 滤波器
# y_smooth = savgol_filter(local_y, window_length=23, polyorder=7)
# plt.plot(local_x ,y_smooth , color='green' ,label='S-G filter')
# # 多项式拟合
# coefficients = np.polyfit(local_x, local_y, 6)  # 数字表示n次多项式拟合
# p_poly = np.poly1d(coefficients)
# y_poly = p_poly(local_x)
# plt.scatter(local_x,local_y, s=100  , label='original data')
# plt.plot(local_x , y_poly , color='red' ,label='6th order polynomial fit')

# plt.xlabel('Lateral distance(m)')
# plt.ylabel('Longitudinal distance(m)')
# plt.title(f'AGENT TRAJECTORY')




# plt.tight_layout()  # 自动调整子图布局，避免重叠   
# plt.show()


# '''
# 东南大学 —— SQM1
# '''
# # 导入所需的包
# import math
# import pandas as pd
# import numpy as np
# from scipy.signal import savgol_filter
# import matplotlib
# import matplotlib.pyplot as plt

# # 读取数据
# path = 'D:\\Eraser_python\\pre_trajectory_for_vehicles\\SQM1\\SQM1.csv'
# data = pd.read_csv(path)
# lane_id = data['LaneID']
# veh_id = data['VehicleID']


# '''Savitzky-Golay 滤波器'''
# plt.rcParams.update({'font.size': 14})  # 设置图例字体大小
# for j in range(4):
#     lanedata = data[data['LaneID']==j+1]
#     # 画n行2列的子图
#     plt.subplot( 2 , 2, j+1 )
    
#     k = 0
#     for i in range(veh_id.nunique()):
#         if (lanedata['VehicleID']==(i+1)).any():
#             x_vehID = lanedata[lanedata['VehicleID']==i+1]
#             local_x = x_vehID['Lateral distance(m)']
#             local_y = x_vehID['Longitudinal distance(m)']
            
#             # Savitzky-Golay 滤波器
#             y_smooth = savgol_filter(local_y, window_length=9, polyorder=7)

#             plt.scatter(local_x,local_y, s=1  , label='original data')
#             plt.plot(local_x ,y_smooth , color='red' ,label='S-G filter')
            
#             plt.xlabel('Lateral distance(m)')
#             plt.ylabel('Longitudinal distance(m)')
            
#             plt.title(f'The {j+1}th lane')
 
            
#             plt.xlim([-3, 3])
#             k +=1
#         if k >=1:
#             break
# plt.tight_layout()  # 自动调整子图布局，避免重叠   
# plt.show()


# '''多项式拟合'''
# plt.rcParams.update({'font.size': 14})  # 设置图例字体大小
# for j in range(4):
#     lanedata = data[data['LaneID']==j+1]
#     # 画n行2列的子图
#     plt.subplot( 2 , 2, j+1 )
    
#     k = 0
#     for i in range(veh_id.nunique()):
#         if (lanedata['VehicleID']==(i+1)).any():
#             x_vehID = lanedata[lanedata['VehicleID']==i+1]
#             local_x = x_vehID['Lateral distance(m)']
#             local_y = x_vehID['Longitudinal distance(m)']
            
#             # 多项式拟合
#             coefficients = np.polyfit(local_x, local_y, 2)  # 3表示三次项，即三次多项式拟合
#             p_poly = np.poly1d(coefficients)
#             y_poly = p_poly(local_x)
#             plt.scatter(local_x,local_y, s=1  , label='original data')
#             plt.plot(local_x , y_poly , color='red' ,label='6th order polynomial fit')
            
#             plt.xlabel('Lateral distance(m)')
#             plt.ylabel('Longitudinal distance(m)')
            
#             plt.title(f'The {j+1}th lane')
 
            
#             plt.xlim([-3, 3])
#             k +=1
#         if k >=1:
#             break
# plt.tight_layout()  # 自动调整子图布局，避免重叠   
# plt.show()



# '''
# 东南大学 —— SQM2
# '''
# # 导入所需的包
# import math
# import pandas as pd
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt

# # 读取数据
# path = 'E:\\毕业设计\\data set\\Ubiquitous Traffic Eyes\\SQM2\\SQM2.csv'
# data = pd.read_csv(path)
# lane_id = data['LaneID']
# veh_id = data['VehicleID']

# # print(data.keys())
# # 提取车道数据
# for j in range(lane_id.nunique()):
#     lanedata = data[data['LaneID']==j+1]
#     # 画n行2列的子图
#     plt.subplot( 2 , math.ceil(lane_id.nunique()/2) , j+1 )
#     for i in range(veh_id.nunique()):
#         if (lanedata['VehicleID']==(i+1)).any():
#             x_vehID = lanedata[lanedata['VehicleID']==i+1]
#             plt.plot(x_vehID['Lateral distance(m)'],x_vehID['Longitudinal distance(m)'])
# plt.tight_layout()  # 自动调整子图布局，避免重叠            
# plt.show()



# '''
# 东南大学 —— YTAvenue3
# '''
# # 导入所需的包
# import math
# import pandas as pd
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt

# # 读取数据
# path = 'E:\\毕业设计\\data set\\Ubiquitous Traffic Eyes\\YTAvenue3\\YTAvenue3.csv'
# data = pd.read_csv(path)
# lane_id = data['LaneID']
# veh_id = data['VehicleID']

# # print(data.keys())
# # 提取车道数据
# for j in range(lane_id.nunique()):
#     lanedata = data[data['LaneID']==j+1]
#     # 画n行2列的子图
#     plt.subplot( 2 , math.ceil(lane_id.nunique()/2) , j+1 )
#     for i in range(veh_id.nunique()):
#         if (lanedata['VehicleID']==(i+1)).any():
#             x_vehID = lanedata[lanedata['VehicleID']==i+1]
#             plt.plot(x_vehID['Lateral distance(m)'],x_vehID['Longitudinal distance(m)'])
# plt.tight_layout()  # 自动调整子图布局，避免重叠            
# plt.show()




# '''
# 东南大学 —— KZM5
# '''
# # 导入所需的包
# import math
# import pandas as pd
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt

# # 读取数据
# path = 'E:\\毕业设计\\data set\\Ubiquitous Traffic Eyes\\KZM5\\KZM5-150m.csv'
# data = pd.read_csv(path)
# lane_id = data['LaneID']
# veh_id = data['VehicleID']

# # print(data.keys())
# # 提取车道数据
# for j in range(lane_id.nunique()):
#     lanedata = data[data['LaneID']==j+1]
#     # 画n行2列的子图
#     plt.subplot( 2 , math.ceil(lane_id.nunique()/2) , j+1 )
#     for i in range(veh_id.nunique()):
#         if (lanedata['VehicleID']==(i+1)).any():
#             x_vehID = lanedata[lanedata['VehicleID']==i+1]
#             plt.plot(x_vehID['Lateral distance(m)'],x_vehID['Longitudinal distance(m)'])
# plt.tight_layout()  # 自动调整子图布局，避免重叠            
# plt.show()




# '''
# 东南大学 —— KZM6
# '''
# # 导入所需的包
# import math
# import pandas as pd
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt

# # 读取数据
# path = 'E:\\毕业设计\\data set\\Ubiquitous Traffic Eyes\\KZM6\\KZM6frenet.csv'
# data = pd.read_csv(path)
# lane_id = data['lane_id']
# veh_id = data['car_id']

# # print(data.keys())
# # 提取车道数据
# for j in range(lane_id.nunique()):
#     lanedata = data[data['lane_id']==j+1]
#     # 画n行2列的子图
#     plt.subplot( 2 , math.ceil(lane_id.nunique()/2) , j+1 )
#     for i in range(veh_id.nunique()):
#         if (lanedata['car_id']==(i+1)).any():
#             x_vehID = lanedata[lanedata['car_id']==i+1]
#             plt.plot(x_vehID['lateral distance<m>'],x_vehID['longitudinal distance<m>'])
# plt.tight_layout()  # 自动调整子图布局，避免重叠            
# plt.show()



# '''
# 东南大学 —— RML7
# '''
# # 导入所需的包
# import math
# import pandas as pd
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt

# # 读取数据
# path = 'E:\\毕业设计\\data set\\Ubiquitous Traffic Eyes\\RML7\\RML7.csv'
# data = pd.read_csv(path)
# lane_id = data['lane_id']
# veh_id = data['car_id']

# # print(data.keys())
# # 提取车道数据
# for j in range(lane_id.nunique()):
#     lanedata = data[data['lane_id']==j+1]
#     # 画n行2列的子图
#     plt.subplot( 2 , math.ceil(lane_id.nunique()/2) , j+1 )
#     for i in range(veh_id.nunique()):
#         if (lanedata['car_id']==(i+1)).any():
#             x_vehID = lanedata[lanedata['car_id']==i+1]
#             plt.plot(x_vehID['latitude'],x_vehID['longitude'])
# plt.tight_layout()  # 自动调整子图布局，避免重叠            
# plt.show()



# '''
# 东南大学 —— PKDD8
# '''
# # 导入所需的包
# import math
# import pandas as pd
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt

# # 读取数据
# path = 'E:\\毕业设计\\data set\\Ubiquitous Traffic Eyes\\PKDD8\\TJU8.csv'
# data = pd.read_csv(path)
# lane_id = data['lane_id']
# veh_id = data['car_id']

# # print(data.keys())
# # 提取车道数据
# for j in range(lane_id.nunique()):
#     lanedata = data[data['lane_id']==j+1]
#     # 画n行2列的子图
#     plt.subplot( 2 , math.ceil(lane_id.nunique()/2) , j+1 )
#     for i in range(veh_id.nunique()):
#         if (lanedata['car_id']==(i+1)).any():
#             x_vehID = lanedata[lanedata['car_id']==i+1]
#             plt.plot(x_vehID['lateral distance<m>'],x_vehID['longitudinal distance<m>'])
# plt.tight_layout()  # 自动调整子图布局，避免重叠            
# plt.show()





# '''
# 东南大学 —— KZM9
# '''
# # 导入所需的包
# import math
# import pandas as pd
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt

# # 读取数据
# path = 'E:\\毕业设计\\data set\\Ubiquitous Traffic Eyes\\KZM9\\KZM9frenet.csv'
# data = pd.read_csv(path)
# lane_id = data['lane']
# veh_id = data['car_id']

# # print(data.keys())
# # 提取车道数据
# for j in range(lane_id.nunique()):
#     lanedata = data[data['lane']==j+1]
#     # 画n行2列的子图
#     plt.subplot( 2 , math.ceil(lane_id.nunique()/2) , j+1 )
#     for i in range(veh_id.nunique()):
#         if (lanedata['car_id']==(i+1)).any():
#             x_vehID = lanedata[lanedata['car_id']==i+1]
#             plt.plot(x_vehID['latitude'],x_vehID['longitude'])
# plt.tight_layout()  # 自动调整子图布局，避免重叠            
# plt.show()







# # 导入字体属性相关的包或类 
# from matplotlib.font_manager import FontProperties
# # 预设字体类型、大小
# font = FontProperties(size=10)
# #设置画布的尺寸
# plt.figure(figsize=(10, 4))



# x_vehID = data[data['car_id']==1]

# plt.scatter(x_vehID['Lateral distance(m)'],x_vehID['Longitudinal distance(m)'])
# # 基本设置及绘图
# plt.show()





