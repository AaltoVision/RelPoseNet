import torchvision.transforms as transforms


def get_imagenet_mean_std():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return mean, std


def get_augmentations():
    train_aug = train_augmentations()
    val_aug = eval_augmentations()
    return train_aug, val_aug


def train_augmentations():
    mean, std = get_imagenet_mean_std()
    transform = transforms.Compose([transforms.RandomCrop(size=224),
                                    transforms.ColorJitter(brightness=0.4,
                                                           contrast=0.4,
                                                           saturation=0.4),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                                    ])

    return transform


def eval_augmentations():
    mean, std = get_imagenet_mean_std()
    transform = transforms.Compose([transforms.CenterCrop(size=224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                                    ])

    return transform
