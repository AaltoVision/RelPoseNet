from os import path as osp
from collections import defaultdict
from PIL import Image
import torch


class SevenScenesRelPoseDataset(object):
    def __init__(self, cfg, split='train', transforms=None):
        self.cfg = cfg
        self.split = split
        self.transforms = transforms
        self.scenes_dict = defaultdict(str)
        for i, scene in enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']):
            self.scenes_dict[i] = scene

        self.fnames1, self.fnames2, self.t_gt, self.q_gt = self._read_pairs_txt()

    def _read_pairs_txt(self):
        fnames1, fnames2, t_gt, q_gt = [], [], [], []

        data_params = self.cfg.data_params

        pairs_txt = data_params.train_pairs_fname if self.split == 'train' else data_params.val_pairs_fname
        with open(pairs_txt, 'r') as f:
            for line in f:
                chunks = line.rstrip().split(' ')
                scene_id = int(chunks[2])
                fnames1.append(osp.join(data_params.img_dir, self.scenes_dict[scene_id], chunks[0][1:]))
                fnames2.append(osp.join(data_params.img_dir, self.scenes_dict[scene_id], chunks[1][1:]))

                t_gt.append(torch.FloatTensor([float(chunks[3]), float(chunks[4]), float(chunks[5])]))
                q_gt.append(torch.FloatTensor([float(chunks[6]),
                                               float(chunks[7]),
                                               float(chunks[8]),
                                               float(chunks[9])]))

        return fnames1, fnames2, t_gt, q_gt

    def __getitem__(self, item):
        img1 = Image.open(self.fnames1[item]).convert('RGB')
        img2 = Image.open(self.fnames2[item]).convert('RGB')
        t_gt = self.t_gt[item]
        q_gt = self.q_gt[item]

        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        return {'img1': img1,
                'img2': img2,
                't_gt': t_gt,
                'q_gt': q_gt}

    def __len__(self):
        return len(self.fnames1)
