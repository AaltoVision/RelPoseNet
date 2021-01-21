import os
from os import path as osp
from tqdm import tqdm
import torch
from experiments.service.benchmark_base import Benchmark
from relposenet.dataset import SevenScenesTestDataset
from relposenet.augmentations import get_augmentations
from relposenet.model import RelPoseNet


class SevenScenesBenchmark(Benchmark):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.dataloader = self._init_dataloader()
        self.model = self._load_model_relposenet().to(self.device)

    def _init_dataloader(self):
        experiment_cfg = self.cfg.experiment.experiment_params

        # define test augmentations
        _, eval_aug = get_augmentations()

        # test dataset
        dataset = SevenScenesTestDataset(experiment_cfg, eval_aug)

        # define a dataloader
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=experiment_cfg.bs,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=experiment_cfg.n_workers,
                                                 drop_last=False)

        return dataloader

    def _load_model_relposenet(self):
        print(f'Loading RelPoseNet model...')
        model_params_cfg = self.cfg.model.model_params
        model = RelPoseNet(model_params_cfg)

        data_dict = torch.load(model_params_cfg.snapshot)
        model.load_state_dict(data_dict['state_dict'])
        print(f'Loading RelPoseNet model... Done!')
        return model.eval()

    def evaluate(self):
        q_est_all, t_est_all = [], []
        print(f'Evaluate on the dataset...')
        with torch.no_grad():
            for data_batch in tqdm(self.dataloader):
                q_est, t_est = self.model(data_batch['img1'].to(self.device),
                                          data_batch['img2'].to(self.device))

                q_est_all.append(q_est)
                t_est_all.append(t_est)

        q_est_all = torch.cat(q_est_all).cpu().numpy()
        t_est_all = torch.cat(t_est_all).cpu().numpy()

        print(f'Write the estimates to a text file')
        experiment_cfg = self.cfg.experiment.experiment_params

        if not osp.exists(experiment_cfg.output.home_dir):
            os.makedirs(experiment_cfg.output.home_dir)

        with open(experiment_cfg.output.res_txt_fname, 'w') as f:
            for q_est, t_est in zip(q_est_all, t_est_all):
                f.write(f"{q_est[0]} {q_est[1]} {q_est[2]} {q_est[3]} {t_est[0]} {t_est[1]} {t_est[2]}\n")

        print(f'Done')
