'''
样例: 从gt_database随机选择，然后copy paste到某一帧的ground points上;
'''

import os
import torch
import numpy as np
import random
import laspy
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from pcdet.ops.iou3d_nms import iou3d_nms_utils

def get_expanded_bbox(points, padding=0.1):
    """计算点集的扩展包络框
    Args:
        points: N×2 或 N×3 的坐标数组
        padding: 边界扩展距离（单位：米）
    Returns:
        expanded_min: 扩展后的最小边界点 [x_min, y_min, (z_min)]
        expanded_max: 扩展后的最大边界点 [x_max, y_max, (z_max)]
    """
    # import pdb;pdb.set_trace()
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    expanded_min = min_vals - padding
    expanded_max = max_vals + padding
    return np.concatenate([expanded_min, expanded_max]) # return [x_min, y_min, x_max, y_max]


def boxes_iou(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 4) [x1, y1, x2, y2]
        boxes_b: (M, 4) [x1, y1, x2, y2]

    Returns:
        iou: (N, M) IoU矩阵
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 4
    
    # 计算交集的坐标
    x_min = np.maximum(boxes_a[:, 0, None], boxes_b[None, :, 0])  # (N, M)
    x_max = np.minimum(boxes_a[:, 2, None], boxes_b[None, :, 2])  # (N, M)
    y_min = np.maximum(boxes_a[:, 1, None], boxes_b[None, :, 1])  # (N, M)
    y_max = np.minimum(boxes_a[:, 3, None], boxes_b[None, :, 3])  # (N, M)
    
    # 计算交集区域的宽高（无重叠时为0）
    x_len = np.maximum(x_max - x_min, 0)  # (N, M)
    y_len = np.maximum(y_max - y_min, 0)  # (N, M)
    
    # 计算各自面积
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])  # (N,)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])  # (M,)
    
    # 计算交集和并集面积
    a_intersect_b = x_len * y_len  # (N, M)
    union = np.maximum(area_a[:, None] + area_b[None, :] - a_intersect_b, 1e-6)  # (N, M)
    
    # 计算IoU
    iou = a_intersect_b / union
    return iou

def boxes_iou_normal(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 4) [x1, y1, x2, y2]
        boxes_b: (M, 4) [x1, y1, x2, y2]

    Returns:

    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 4
    x_min = torch.max(boxes_a[:, 0, None], boxes_b[None, :, 0])
    x_max = torch.min(boxes_a[:, 2, None], boxes_b[None, :, 2])
    y_min = torch.max(boxes_a[:, 1, None], boxes_b[None, :, 1])
    y_max = torch.min(boxes_a[:, 3, None], boxes_b[None, :, 3])
    x_len = torch.clamp_min(x_max - x_min, min=0)
    y_len = torch.clamp_min(y_max - y_min, min=0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    a_intersect_b = x_len * y_len
    iou = a_intersect_b / torch.clamp_min(area_a[:, None] + area_b[None, :] - a_intersect_b, min=1e-6)
    return iou

def points_in_boxes(points, boxes):
    """
    判断点是否位于矩形框内
    
    Args:
        points: (N, 2) 数组，表示N个点的[x, y]坐标
        boxes: (M, 4) 数组，每个框为[x1, y1, x2, y2]（左上和右下坐标）
    
    Returns:
        mask: (N, M) 布尔数组，True表示点位于对应框内
    """
    # 将输入转换为numpy数组（如果尚未是）
    points = np.asarray(points)
    boxes = np.asarray(boxes)
    
    # 扩展维度以便广播计算 (N,1,2) 和 (1,M,4)
    points = points[:, np.newaxis, :]  # (N,1,2)
    x1, y1, x2, y2 = np.split(boxes, 4, axis=1)  # 每个形状 (M,1)
    
    # 判断条件：x1 <= x <= x2 且 y1 <= y <= y2
    mask = (points[..., 0] >= x1.T) & (points[..., 0] <= x2.T) & \
           (points[..., 1] >= y1.T) & (points[..., 1] <= y2.T)
    
    return mask  # (N,M)

# box_extra_len = 0.2 # 

# datasets/semantickitti/assets/segments_gridsample/00/000109_2.seg
# datasets/semantickitti/assets/segments_gridsample/00/000209_5.seg
# datasets/semantickitti/assets/segments_gridsample/00/000309_7.seg
# datasets/semantickitti/assets/segments_gridsample/00/000409_10.seg
# datasets/semantickitti/assets/segments_gridsample/00/000509_12.seg

# pc_file_path='datasets/semantickitti/sequences/00/velodyne/000009.bin'
# label_file_path='datasets/semantickitti/assets/segments_gridsample/00/000009_0.seg'
pc_file_path='datasets/semantickitti/sequences/00/velodyne/000609.bin'
label_file_path='datasets/semantickitti/assets/segments_gridsample/00/000609_15.seg'

#  00/000609_15.seg   00/002609_65.seg
# #  00/000709_17.seg   00/002709_67.seg
# #  00/000809_20.seg   00/002809_70.seg
# #  00/000909_22.seg   00/002909_72.seg
# #  00/001009_25.seg   00/003009_75.seg
# #  00/001109_27.seg   00/003109_77.seg
# #  00/001209_30.seg   00/003209_80.seg
# #  00/001309_32.seg   00/003309_82.seg
# #  00/001409_35.seg   00/003409_85.seg
# #  00/001509_37.seg   00/003509_87.seg
# #  00/001609_40.seg   00/003609_90.seg
# #  00/001709_42.seg   00/003709_92.seg
# #  00/001809_45.seg   00/003809_95.seg
# #  00/001909_47.seg   00/003909_97.seg
# #  00/002009_50.seg   00/004009_100.seg
# #  00/002109_52.seg   00/004109_102.seg
# #  00/002209_55.seg   00/004209_105.seg
# #  00/002309_57.seg   00/004309_107.seg
# #  00/002409_60.seg   00/004409_110.seg
# #  00/002509_62.seg   00/004509_112.seg


# 1. 读取点云和实例标签
points = np.fromfile(pc_file_path, dtype=np.float32).reshape(-1, 4)
points = points[:, :3]  # 只取XYZ，忽略其他通道
instance_labels = np.fromfile(label_file_path, dtype=np.int16).astype(np.int32)

# 筛选地面点
ground_points = points[instance_labels==0]
ground_labels = np.zeros(len(ground_points))

gt_database_dir = 'datasets/semantickitti/assets/gt_database'
gt_object_file_list = [f'{gt_database_dir}/{fname}' for fname in os.listdir(gt_database_dir)]

existed_boxes = np.empty([0,4])
total_valid_sampled_points = np.empty([0, 4])
total_sampled_instance_ids = np.array([])

valid_id_cum = 1

for s_id in range(60):
    sampled_segment_files = random.sample(gt_object_file_list, 50)  # 抽取10个不重复的元素
    segments_list = [np.fromfile(x, dtype=np.float64).reshape(-1, 5)[:, :4] for x in sampled_segment_files]

    # compute the convex hull box of segments
    sampled_boxes = np.array([get_expanded_bbox(x[:, :2]) for x in segments_list])

    overlap_mask = points_in_boxes(ground_points[:, :2], sampled_boxes)
    segments_valid_mask = overlap_mask.max(axis=0) == 1
    if segments_valid_mask.sum()==0: 
        # print("iteration:", s_id, "invalid sampled_boxes")
        continue
    # import pdb;pdb.set_trace()
    sampled_boxes = sampled_boxes[segments_valid_mask]
    segments_list = [segments_list[x] for x in range(len(segments_list)) if segments_valid_mask[x]]
    # boxes_iou(segemnts_boxes, segemnts_boxes)

    iou1 = boxes_iou(sampled_boxes, existed_boxes)
    iou2 = boxes_iou(sampled_boxes, sampled_boxes)
    iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
    iou1 = iou1 if iou1.shape[1] > 0 else iou2
    valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0)
    valid_mask = valid_mask.nonzero()[0]
    # print("iteration:", s_id, "valid_mask", valid_mask)

    if len(valid_mask)==0: continue

    valid_sampled_points = np.concatenate([segments_list[x] for x in valid_mask])
    valid_sampled_boxes = sampled_boxes[valid_mask]

    # import pdb;pdb.set_trace()

    existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0)
    # total_valid_sampled_points.extend(valid_sampled_points)
    total_valid_sampled_points = np.concatenate([total_valid_sampled_points, valid_sampled_points])

    sampled_instance_ids = np.concatenate([np.full(len(valid_sampled_points), valid_id_cum+x) for x in range(len(valid_mask))])
    total_sampled_instance_ids = np.concatenate([total_sampled_instance_ids, sampled_instance_ids])
    valid_id_cum += len(valid_mask)

points = np.concatenate([ground_points, total_valid_sampled_points[:, :3]])
labels = np.concatenate([ground_labels, total_sampled_instance_ids])

# import pdb;pdb.set_trace()
# print(np.unique(labels).shape);exit()
instance_labels = labels

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

# 4. 创建并写入 LAS 文件
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


