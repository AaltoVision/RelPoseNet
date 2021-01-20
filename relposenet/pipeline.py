from relposenet.model import RelPoseNet


class Pipeline(object):
    def __init__(self, cfg):
        self.cfg = cfg
        cfg_model = self.cfg.model_params

        self.model = RelPoseNet(cfg_model)

    def run(self):
        pass
