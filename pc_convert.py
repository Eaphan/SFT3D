import numpy as np
import laspy
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

pc_file_path='datasets/semantickitti/sequences/00/velodyne/000009.bin'
label_file_path='datasets/semantickitti/assets/segments_gridsample/00/000009_0.seg'

# 1. 读取点云和实例标签
points = np.fromfile(pc_file_path, dtype=np.float32).reshape(-1, 4)
points = points[:, :3]  # 只取XYZ，忽略其他通道
instance_labels = np.fromfile(label_file_path, dtype=np.int16).astype(np.int32)

# 2. 获取tab20色系
tab20 = plt.cm.get_cmap('tab20').colors  # 直接获取tab20的RGB颜色列表
unique_instances = np.unique(instance_labels)
num_instances = len(unique_instances)

# 3. 为每个实例分配颜色
# 如果实例数量超过20个，循环使用tab20颜色
instance_color_map = {inst_id: tab20[i % 20] for i, inst_id in enumerate(unique_instances)}

# 为每个点分配颜色
colors = np.array([instance_color_map[inst] for inst in instance_labels]) * 255
colors = colors.astype(np.uint8)

# 4. 创建并写入LAS文件
header = laspy.LasHeader(point_format=7)  # 格式7支持RGB颜色
header.scales = [0.001, 0.001, 0.001]  # 设置合理的缩放比例

las = laspy.LasData(header)
las.x = points[:, 0]
las.y = points[:, 1]
las.z = points[:, 2]

# LAS文件要求RGB范围是0-65535（16bit）
las.red = colors[:, 0] * 256    # 0-255 -> 0-65535
las.green = colors[:, 1] * 256
las.blue = colors[:, 2] * 256

# 4. 保存文件
las.write("colored_instances.las")

print(f"已保存LAS文件，包含{len(unique_instances)}个实例的tab20配色")
