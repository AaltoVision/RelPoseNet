import os
from os import path as osp
import time
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from relposenet.model import RelPoseNet
from relposenet.dataset import SevenScenesRelPoseDataset
from relposenet.augmentations import get_augmentations
from relposenet.criterion import RelPoseCriterion
from relposenet.utils import cycle, set_seed


class Pipeline(object):
    def __init__(self, cfg):
        self.cfg = cfg
        cfg_model = self.cfg.model_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.cfg.seed)

        # initialize dataloaders
        self.train_loader, self.val_loader = self._init_dataloaders()
        self.train_loader_iterator = iter(cycle(self.train_loader))

        self.model = RelPoseNet(cfg_model).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.cfg.train_params.lr)

        # Scheduler
        cfg_scheduler = self.cfg.train_params.scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=cfg_scheduler.lrate_decay_steps,
                                                         gamma=cfg_scheduler.lrate_decay_factor)

        # Criterion
        self.criterion = RelPoseCriterion(self.cfg.train_params.alpha).to(self.device)

        # create writer (logger)
        self.writer = SummaryWriter(self.cfg.output_params.logger_dir)

        self.start_step = 0
        self.val_total_loss = 1e6
        if self.cfg.model_params.resume_snapshot:
            self._load_model(self.cfg.model_params.resume_snapshot)

    def _init_dataloaders(self):
        cfg_data = self.cfg.data_params
        cfg_train = self.cfg.train_params

        # get image augmentations
        train_augs, val_augs = get_augmentations()

        train_dataset = SevenScenesRelPoseDataset(cfg=self.cfg, split='train', transforms=train_augs)

        val_dataset = SevenScenesRelPoseDataset(cfg=self.cfg, split='val', transforms=val_augs)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=cfg_train.bs,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=cfg_train.n_workers,
                                                   drop_last=True)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=cfg_train.bs,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=cfg_train.n_workers,
                                                 drop_last=True)
        return train_loader, val_loader

    def _predict_cam_pose(self, mini_batch):
        q_est, t_est = self.model.forward(mini_batch['img1'].to(self.device),
                                          mini_batch['img2'].to(self.device))
        return q_est, t_est

    def _save_model(self, step, loss_val, best_val=False):
        if not osp.exists(self.cfg.output_params.snapshot_dir):
            os.makedirs(self.cfg.output_params.snapshot_dir)

        fname_out = 'best_val.pth' if best_val else 'snapshot{:06d}.pth'.format(step)
        save_path = osp.join(self.cfg.output_params.snapshot_dir, fname_out)
        model_state = self.model.state_dict()
        torch.save({'step': step,
                    'state_dict': model_state,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'val_loss': loss_val,
                    },
                   save_path)

    def _load_model(self, snapshot):
        data_dict = torch.load(snapshot)
        self.model.load_state_dict(data_dict['state_dict'])
        self.optimizer.load_state_dict(data_dict['optimizer'])
        self.scheduler.load_state_dict(data_dict['scheduler'])
        self.start_step = data_dict['step']
        if 'val_loss' in data_dict:
            self.val_total_loss = data_dict['val_loss']

    def _train_batch(self):
        train_sample = next(self.train_loader_iterator)
        q_est, t_est = self._predict_cam_pose(train_sample)

        self.optimizer.zero_grad()

        # compute loss
        loss, t_loss_val, q_loss_val = self.criterion(train_sample['q_gt'].to(self.device),
                                                      train_sample['t_gt'].to(self.device),
                                                      q_est,
                                                      t_est)
        loss.backward()

        # update the optimizer
        self.optimizer.step()

        # update the scheduler
        self.scheduler.step()
        return loss.item(), t_loss_val, q_loss_val

    def _validate(self):
        self.model.eval()
        loss_total, t_loss_total, q_loss_total = 0., 0., 0.

        with torch.no_grad():
            for val_sample in tqdm(self.val_loader):
                q_est, t_est = self._predict_cam_pose(val_sample)
                # compute loss
                loss, t_loss_val, q_loss_val = self.criterion(val_sample['q_gt'].to(self.device),
                                                              val_sample['t_gt'].to(self.device),
                                                              q_est,
                                                              t_est)
                loss_total += loss.item()
                t_loss_total += t_loss_val
                q_loss_total += q_loss_val

        avg_total_loss = loss_total / len(self.val_loader)
        avg_t_loss = t_loss_total / len(self.val_loader)
        avg_q_loss = q_loss_total / len(self.val_loader)

        self.model.train()

        return avg_total_loss, avg_t_loss, avg_q_loss

    def run(self):
        print('Start training', self.start_step)
        train_start_time = time.time()
        train_log_iter_time = time.time()
        for step in range(self.start_step + 1, self.start_step + self.cfg.train_params.n_train_iters):
            train_loss_batch, _, _ = self._train_batch()

            if step % self.cfg.output_params.log_scalar_interval == 0 and step > 0:
                self.writer.add_scalar('Train_total_loss_batch', train_loss_batch, step)
                print(f'Elapsed time [min] for {self.cfg.output_params.log_scalar_interval} iterations: '
                      f'{(time.time() - train_log_iter_time) / 60.}')
                train_log_iter_time = time.time()
                print(f'Step {step} out of {self.cfg.train_params.n_train_iters} is done. Train loss (per batch): '
                      f'{train_loss_batch}.')

            if step % self.cfg.output_params.validate_interval == 0 and step > 0:
                val_time = time.time()
                best_val = False
                val_total_loss, val_t_loss, val_q_loss = self._validate()
                self.writer.add_scalar('Val_total_loss', val_total_loss, step)
                self.writer.add_scalar('Val_t_loss', val_t_loss, step)
                self.writer.add_scalar('Val_q_loss', val_q_loss, step)
                if val_total_loss < self.val_total_loss:
                    self.val_total_loss = val_total_loss
                    best_val = True
                self._save_model(step, val_total_loss, best_val=best_val)
                print(f'Validation loss: {val_total_loss}, t_loss: {val_t_loss}, q_loss: {val_q_loss}')
                print(f'Elapsed time [min] for validation: {(time.time() - val_time) / 60.}')
                train_log_iter_time = time.time()

        print(f'Elapsed time for training [min] {(time.time() - train_start_time) / 60.}')
        print('Done')
