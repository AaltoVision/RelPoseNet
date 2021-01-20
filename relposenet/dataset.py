import os, sys
from collections import defaultdict
from PIL import Image
import random

class SevenScenesRelPoseDataset(object):
    def __init__(self, cfg, img_path, pair_path='assets/data',split='train', transforms=None):
        self.cfg = cfg
        self.img_path = img_path
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

        img1_tmp = Image.open(os.path.join(self.img_path, pairname[0]))
        img2_tmp = Image.open(os.path.join(self.img_path, pairname[1]))

        flip = random.sample[[0,1],1][0]

        if flip:
            img1 = img2_tmp
            img2 = img1_tmp

            quat = [quaternion_gt[1], quaternion_gt[2], -quaternion_gt[3], -quaternion_gt[4]]
            trans = -translation_gt
        else:
            img1 = img1_tmp
            img2 = img2_tmp

            quat = quaternion_gt
            trans = translation_gt

        # transform


        return {'data' = }

