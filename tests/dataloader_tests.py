import hydra
import torch
from relposenet.dataset import SevenScenesRelPoseDataset
from relposenet.augmentations import train_augmentations
import matplotlib.pyplot as plt


@hydra.main(config_path="../configs", config_name="main")
def main(cfg):
    augs = train_augmentations()
    dataset = SevenScenesRelPoseDataset(cfg,
                                        split='train',
                                        transforms=augs)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=8)

    for idx, mini_batch in enumerate(dataloader):

        print(f'Pair id: {idx}')
        print(f't: {mini_batch["t_gt"]}, q: {mini_batch["q_gt"]}')

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(35, 10))
        axes = axes.flatten()

        axes[0].imshow(mini_batch['img1'].permute(0, 2, 3, 1).squeeze().numpy())
        axes[1].imshow(mini_batch['img2'].permute(0, 2, 3, 1).squeeze().numpy())

        axes[0].axis("off")
        axes[1].axis("off")

        plt.show(block=False)
        plt.pause(3)
        plt.close()


if __name__ == '__main__':
    main()
