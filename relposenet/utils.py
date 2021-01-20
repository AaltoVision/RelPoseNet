import numpy as np
import random
import torch


def set_seed(seed):
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
