import torch


class Benchmark(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self):
        raise NotImplementedError
