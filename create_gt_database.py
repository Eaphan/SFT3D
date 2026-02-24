
import re
import os
from os.path import basename, splitext
import numpy as np
import pypatchworkpp as pwpp
from torch.utils.data import Dataset
from preprocess.pcd_preprocess import clusterize_pcd, grid_sample, order_segments, apply_transform

# for visualization
import laspy
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from tqdm import tqdm

class KITTIPreprocessor(Dataset):
    def __init__(self, data_dir, scan_window, split, downsampling_resolution, ground_method):
        super().__init__()
        self.data_dir = data_dir
        self.augmented_dir = 'segments_gridsample'

        self.downsampling_resolution = downsampling_resolution
        self.scan_window = scan_window
        self.sampling_window = scan_window

        self.split = split
        if split == 'train':
            self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        elif split == 'val':
            self.seqs = ['08']
        elif split == 'trainval':
            self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        self.ground_method = ground_method

        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath_pretrain()
        self.nr_data = len(self.points_datapath)
        print('The size of %s data is %d' % (self.split, self.nr_data))

    def datapath_pretrain(self):
        self.points_datapath = []

        for seq in self.seqs:
            point_seq_path = os.path.join(self.data_dir, 'sequences', seq, 'velodyne')
            point_seq_bin = os.listdir(point_seq_path)
            point_seq_bin.sort()

            for file_num in range(0, len(point_seq_bin), self.sampling_window):
                end_file = file_num + self.scan_window if len(point_seq_bin) - file_num > self.scan_window else len(point_seq_bin)
                self.points_datapath.append([os.path.join(point_seq_path, point_file) for point_file in point_seq_bin[file_num:end_file]])
                if end_file == len(point_seq_bin):
                    break

    def __getitem__(self, index):
        seq_num = self.points_datapath[index][0].split('/')[-3]
        # if seg_num == '08':
        if seq_num in ['00', '01', '02', '03', '04', '05', '06', '07', '08']:
            return
        fname = self.points_datapath[index][0].split('/')[-1].split('.')[0]

        # cluster the aggregated pcd and save the result
        cluster_path = os.path.join(self.data_dir, 'assets', self.augmented_dir, seq_num)
        points_agg, num_points, distance = self.aggregate_pcds(self.points_datapath[index])

        segments_list = []
        for fname, num in zip(self.points_datapath[index], num_points):
            fname = splitext(basename(fname))[0]

            # labels via cluster
            lname = os.path.join(cluster_path, fname + f'_{index}.seg')
            segments = np.fromfile(lname, dtype=np.int16).astype(np.int32) # instance

            # labels via inference
            # lname = f'datasets/semantickitti/assets/sk_temporal_official/{seq_num}/{fname}.seg'
            # segments = np.fromfile(lname, dtype=np.uint32)

            segments_list.append(segments)
        # import pdb;pdb.set_trace()

        # To save the segments in 2 frames
        frame_len = len(num_points)
        for f_id in range(frame_len-1):
            if f_id%5!=0: continue
            segments_curr = segments_list[f_id]
            segments_next = segments_list[f_id+1]
            instances_curr = np.unique(segments_curr)
            instances_next = np.unique(segments_next)
            share_ids = [x for x in instances_curr if x in instances_next]

            for _sid in share_ids:
                if _sid in [-1, 0]: continue
                # import pdb;pdb.set_trace()
                points_s_curr = points_agg[points_agg[:, -1]==f_id][segments_curr==_sid]
                points_s_next = points_agg[points_agg[:, -1]==f_id+1][segments_next==_sid]
                points_s = np.concatenate([points_s_curr, points_s_next])

                print(points_s.min(0), points_s.max(0))
                import pdb;pdb.set_trace()
                save_path = f'datasets/semantickitti/assets/gt_database/{seq_num}_{index}_{f_id}_{_sid}.bin'
                # points_s.tofile(save_path)

    def __len__(self):
        return self.nr_data

    def aggregate_pcds(self, data_batch):
        # load empty pcd point cloud to aggregate
        points_set = np.empty((0, 5))
        num_points = []
        distance = []

        fname = data_batch[0].split('/')[-1].split('.')[0]

        # load poses
        datapath = data_batch[0].split('velodyne')[0]
        poses = self.load_poses(os.path.join(datapath, 'calib.txt'), os.path.join(datapath, 'poses.txt'))

        for t in range(len(data_batch)):
            fname = data_batch[t].split('/')[-1].split('.')[0]
            # load the next t scan, apply pose and aggregate
            p_set = np.fromfile(data_batch[t], dtype=np.float32)
            p_set = p_set.reshape((-1, 4))

            new_col = np.full((p_set.shape[0], 1), t)
            p_set = np.hstack([p_set, new_col])

            # p_set[:, 3] = t
            dist_ref = p_set[:, :2] - [0.5,0.]
            dist_ref[:, 0] *= 0.75
            distance.append(np.linalg.norm(dist_ref, axis=1))
            # distance.append(np.linalg.norm(p_set[:, :2] - [0.5,0.], axis=1))
            num_points.append(len(p_set))
            pose_idx = int(fname)
            p_set[:, :3] = apply_transform(p_set[:, :3], poses[pose_idx])
            points_set = np.vstack([points_set, p_set])

        return points_set, num_points, np.concatenate(distance)

    def load_poses(self, calib_fname, poses_fname):
        calibration = self.parse_calibration(calib_fname)
        poses_file = open(poses_fname)

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        poses = []

        for line in poses_file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses
    
    @staticmethod
    def parse_calibration(filename):
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

if __name__ == "__main__":
    SCAN_WINDOW = 40  # Number of frames to aggregate in the clusterization
    DOWNSAMPLING_RESOLUTION = [0.05,0.05,0.05,5]
    GROUND_METHOD = "patchworkpp"
    ds = KITTIPreprocessor(data_dir="datasets/semantickitti/",
                           scan_window=SCAN_WINDOW,
                           split='trainval',
                           downsampling_resolution=DOWNSAMPLING_RESOLUTION,
                           ground_method=GROUND_METHOD,
                           )
    with tqdm(total=len(ds)) as pbar:
        for i in tqdm(range(len(ds))):
            if i%5!=0: continue
            data = ds[i]  # 获取第 i 个样本
            pbar.update(5)

