import os, sys
from collections import defaultdict


class SevenScenesRelPoseDataset(object):
    def __init__(self, cfg, img_path, pair_path='assets/data',split='train', transforms=None):
        self.cfg = cfg
        self.transforms = transforms
        self.scenes_dict = defaultdict(str)
        for i, scene in enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']):
            self.scenes_dict[i] = scene

        file_path = '{}/db_all_med_hard_{}.txt'.format(pair_path, split)
        with open(file_path, 'r') as f:
            self.file = f.readlines()


    def __getitem__(self, idx):

        line_info = self.file[idx].split(' ')
        pairname = [line_info[0], line_info[1]]
        scene_id = int(line_info[2])
        translation_gt = [line_info[3], line_info[4], line_info[5]]
        quaternion_gt = [line_info[7], line_info[8], line_info[9], line_info[10]]



