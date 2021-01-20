from collections import defaultdict


class SevenScenesRelPoseDataset(object):
    def __init__(self, cfg, transforms):
        self.cfg = cfg
        self.transforms = transforms
        self.scenes_dict = defaultdict(str)
        for i, scene in enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']):
            self.scenes_dict[i] = scene
