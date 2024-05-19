import pandas as pd
PATH = 'D:/Eraser_python/pre_trajectory_for_vehicles/Dataset/highd-dataset-v1.0/select_data/1_tracks_filtered.csv'
data = pd.read_csv(PATH , index_col=0)
#将第0列作为索引index_col=0
# data.head()#查看数据前五行
data.info() # 查看数据情况