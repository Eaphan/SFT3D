import os
import numpy as np

# 定义目录路径
directory = 'datasets/semantickitti/assets/segments_gridsample/00'

# 用于统计所有 instance 的总数量
total_instance_count = 0

# 遍历目录下的所有文件
for filename in os.listdir(directory):
    # 获取文件的完整路径
    file_path = os.path.join(directory, filename)
    
    # 确保是文件（忽略子目录）
    if os.path.isfile(file_path):
        try:
            # 从文件中读取数据
            instance = np.fromfile(file_path, dtype=np.int16).astype(np.int32)
            
            # import pdb;pdb.set_trace()

            # 统计该文件中的 instance 数量
            total_instance_count += np.unique(instance).shape[0] - 1
        
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

# 输出统计结果
print(f"Avg number of instances: {total_instance_count/len(os.listdir(directory))}")






