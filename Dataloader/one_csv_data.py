import pandas as pd
from pathlib import Path
datas = pd.read_csv('E:\\毕业设计\\data set\\NGSIM_Data.csv')
 
def main():
    choice = int(input('请输入需要提取数据的路段(1.i-80;2.us-101;3.larkershim;4.peachtree):'))
    if choice==1:
        i_80()
    if choice==2:
        us_101()
    if choice==3:
        larkershim()
    if choice==4:
        peachtree()
 
def i_80():
    """获取i-80道路上的数据"""
    datas_i_80=datas[datas.Location=='i-80']
    """"按车辆类型进行数据读取"""
    answer=int(input('请输出需要提取的车辆类型数据(1.摩托车;2.小型车;3.大型车):'))
    if answer==1:
        datas_i_80_m=datas_i_80[datas_i_80.v_Class==1]
        datas_i_80_m.to_csv('E:\\毕业设计\\data set\\NGSIM_Data\\I-80\\摩托车数据.csv')
        print('i-80路段,摩托车数据已导出完毕')
    if answer==2:
        datas_i_80_c = datas_i_80[datas_i_80.v_Class == 2]
        datas_i_80_c.to_csv('E:\\毕业设计\\data set\\NGSIM_Data\\I-80\\小型车数据.csv')
        if datas_i_80_c.shape[0]>1048765: # 行数大于该数时进行分割数据集
            split_i_80_c()
        print('i-80路段,小型车数据已导出完毕')
    if answer==3:
        datas_i_80_t = datas_i_80[datas_i_80.v_Class == 3]
        datas_i_80_t.to_csv('E:\\毕业设计\\data set\\NGSIM_Data\\I-80\\大型车数据.csv')
        print('i-80路段,大型车数据已导出完毕')
    quertion()
 
def us_101():
    datas_us_101 = datas[datas.Location == 'us-101']
    answer = int(input('请输出需要提取的车辆类型数据(1.摩托车;2.小型车;3.大型车):'))
    if answer == 1:
        datas_us_101_m = datas_us_101[datas_us_101.v_Class == 1]
        datas_us_101_m.to_csv('E:\\毕业设计\\data set\\NGSIM_Data\\US-101\\摩托车数据.csv')
        print('us-101路段,摩托车数据已导出完毕')
    if answer == 2:
        datas_us_101_c = datas_us_101[datas_us_101.v_Class == 2]
        datas_us_101_c.to_csv('E:\\毕业设计\\data set\\NGSIM_Data\\US-101\\小型车数据.csv')
        if datas_us_101_c.shape[0]>1048765:
            split_us_101_c()
        print('us-101路段,小型车数据已导出完毕')
    if answer == 3:
        datas_us_101_t = datas_us_101[datas_us_101.v_Class == 3]
        datas_us_101_t.to_csv('E:\\毕业设计\\data set\\NGSIM_Data\\US-101\\大型车数据.csv')
        print('us-101路段,大型车数据已导出完毕')
    quertion()
 
def larkershim():
    datas_lankershim = datas[datas.Location == 'lankershim']
    answer = int(input('请输出需要提取的车辆类型数据(1.摩托车;2.小型车;3.大型车):'))
    if answer == 1:
        datas_lankershim_m = datas_lankershim[datas_lankershim.v_Class == 1]
        datas_lankershim_m.to_csv('E:\\毕业设计\\data set\\NGSIM_Data\\lankershim\\摩托车数据.csv')
        print('lankershim路段,摩托车数据已导出完毕')
    if answer == 2:
        datas_lankershim_c = datas_lankershim[datas_lankershim.v_Class == 2]
        datas_lankershim_c.to_csv('E:\\毕业设计\\data set\\NGSIM_Data\\lankershim\\小型车数据.csv')
        if datas_lankershim_c.shape[0]>1048765:
            split_lankershim_c()
        print('lankershim路段,小型车数据已导出完毕')
    if answer == 3:
        datas_lankershim_t = datas_lankershim[datas_lankershim.v_Class == 3]
        datas_lankershim_t.to_csv('E:\\毕业设计\\data set\\NGSIM_Data\\lankershim\\大型车数据.csv')
        print('lankershim路段,大型车数据已导出完毕')
    quertion()
 
def peachtree():
    datas_peachtree = datas[datas.Location == 'peachtree']
    answer = int(input('请输出需要提取的车辆类型数据(1.摩托车;2.小型车;3.大型车):'))
    if answer == 1:
        datas_peachtree_m = datas_peachtree[datas_peachtree.v_Class == 1]
        datas_peachtree_m.to_csv('E:\\毕业设计\\data set\\NGSIM_Data\\peachtree\\摩托车数据.csv')
        print('peachtree路段,摩托车数据已导出完毕')
    if answer == 2:
        datas_peachtree_c = datas_peachtree[datas_peachtree.v_Class == 2]
        datas_peachtree_c.to_csv('E:\\毕业设计\\data set\\NGSIM_Data\\peachtree\\小型车数据.csv')
        if datas_peachtree_c.shape[0]>1048765:
            split_peachtree_c()
        print('peachtree路段,小型车数据已导出完毕')
    if answer == 3:
        datas_lankershim_t = datas_peachtree[datas_peachtree.v_Class == 3]
        datas_lankershim_t.to_csv('E:\\毕业设计\\data set\\NGSIM_Data\\peachtree\\大型车数据.csv')
        print('peachtree路段,大型车数据已导出完毕')
    quertion()
 
def split_lankershim_c():
    res_file_path = Path(r'E:\\毕业设计\\data set\\NGSIM_Data\\lankershim\\小型车数据.csv')  # 仅修改待分割文件路径即可
    split_size = 1000000
    tar_dir = res_file_path.parent / ("split_" + res_file_path.name.split(".")[0])
    if not tar_dir.exists():
        tar_dir.mkdir()
        print("创建文件夹\t" + str(tar_dir))
    print("目标路径：\t" + str(tar_dir))
    print("分割文件：\t" + str(res_file_path))
    print("分割大小：\t" + "{:,}".format(split_size))
    tmp = pd.read_csv(res_file_path, nrows=10)
    columns = tmp.columns.to_list()
    idx = 0
    while (len(tmp) > 0):
        start = 1 + (idx * split_size)
        tmp = pd.read_csv(res_file_path,
                          header=None,
                          names=columns,
                          skiprows=start,
                          nrows=split_size)
        if len(tmp) <= 0:
            break
        file_name = res_file_path.name.split(".")[0] + "_{}_{}".format(start, start + len(tmp)) + ".csv"
        file_path = tar_dir / file_name
        tmp.to_csv(file_path, index=False)
        idx += 1
        print(file_name + "\t切割成功")
 
def split_i_80_c():
    res_file_path = Path(r'E:\\毕业设计\\data set\\NGSIM_Data\\I-80\\小型车数据.csv')  # 仅修改待分割文件路径即可
    split_size = 1000000
    tar_dir = res_file_path.parent / ("split_" + res_file_path.name.split(".")[0])
    '''
    res_file_path.parent 返回 res_file_path 的父文件夹路径。
    res_file_path.name 返回 res_file_path 的文件名，包括扩展名。
    / 是路径连接符，用于将父文件夹路径和新的文件夹名拼接在一起，得到最终的目标文件夹路径。
    '''
    if not tar_dir.exists():
        tar_dir.mkdir()
        print("创建文件夹\t" + str(tar_dir))
    print("目标路径：\t" + str(tar_dir))
    print("分割文件：\t" + str(res_file_path))
    print("分割大小：\t" + "{:,}".format(split_size))

    tmp = pd.read_csv(res_file_path, nrows=10)
    columns = tmp.columns.to_list()
    '''
    这两行代码用于读取一个 CSV 文件的前 10 行数据，并获取数据的列名（列标签）。
    '''
    idx = 0
    while (len(tmp) > 0):
        start = 1 + (idx * split_size)
        tmp = pd.read_csv(res_file_path,
                          header=None,
                          names=columns,
                          skiprows=start,
                          nrows=split_size)
        if len(tmp) <= 0:
            break
        file_name = res_file_path.name.split(".")[0] + "_{}_{}".format(start, start + len(tmp)) + ".csv"
        file_path = tar_dir / file_name
        tmp.to_csv(file_path, index=False)
        idx += 1
        print(file_name + "\t切割成功")
 
def split_peachtree_c():
    res_file_path = Path(r'E:\\毕业设计\\data set\\NGSIM_Data\\peachtree\\小型车数据.csv')  # 仅修改待分割文件路径即可
    split_size = 1000000
    tar_dir = res_file_path.parent / ("split_" + res_file_path.name.split(".")[0])
    if not tar_dir.exists():
        tar_dir.mkdir()
        print("创建文件夹\t" + str(tar_dir))
    print("目标路径：\t" + str(tar_dir))
    print("分割文件：\t" + str(res_file_path))
    print("分割大小：\t" + "{:,}".format(split_size))
    tmp = pd.read_csv(res_file_path, nrows=10)
    columns = tmp.columns.to_list()
    idx = 0
    while (len(tmp) > 0):
        start = 1 + (idx * split_size)
        tmp = pd.read_csv(res_file_path,
                          header=None,
                          names=columns,
                          skiprows=start,
                          nrows=split_size)
        if len(tmp) <= 0:
            break
        file_name = res_file_path.name.split(".")[0] + "_{}_{}".format(start, start + len(tmp)) + ".csv"
        file_path = tar_dir / file_name
        tmp.to_csv(file_path, index=False)
        idx += 1
        print(file_name + "\t切割成功")
 
def split_us_101_c():
    res_file_path = Path(r'E:\\毕业设计\\data set\\NGSIM_Data\\US-101\\小型车数据.csv')  # 仅修改待分割文件路径即可
    split_size = 1000000
    tar_dir = res_file_path.parent / ("split_" + res_file_path.name.split(".")[0])
    if not tar_dir.exists():
        tar_dir.mkdir()
        print("创建文件夹\t" + str(tar_dir))
    print("目标路径：\t" + str(tar_dir))
    print("分割文件：\t" + str(res_file_path))
    print("分割大小：\t" + "{:,}".format(split_size))
    tmp = pd.read_csv(res_file_path, nrows=10)
    columns = tmp.columns.to_list()
    idx = 0
    while (len(tmp) > 0):
        start = 1 + (idx * split_size)
        tmp = pd.read_csv(res_file_path,
                          header=None,
                          names=columns,
                          skiprows=start,
                          nrows=split_size)
        if len(tmp) <= 0:
            break
        file_name = res_file_path.name.split(".")[0] + "_{}_{}".format(start, start + len(tmp)) + ".csv"
        file_path = tar_dir / file_name
        tmp.to_csv(file_path, index=False)
        idx += 1
        print(file_name + "\t切割成功")
 
def quertion():
        question=int(input('是否继续导出数据？(1.是;2.否）：'))
        if question==1:
            main()
        if question==2:
            print('程序关闭')
 
 
if __name__ == '__main__':
    main()
 
 
 
