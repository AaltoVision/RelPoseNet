import torchvision.transforms as transforms


def get_imagenet_mean_std():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return mean, std


def net_preprocessing():
    mean, std = get_imagenet_mean_std()
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    return transform


def train_augmentations():
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.RandomCrop(size=224),
                                    transforms.RandomChoice([
                                        transforms.ColorJitter(brightness=0.4,
                                                               contrast=0.4,
                                                               saturation=0.4),
                                        transforms.GaussianBlur(kernel_size=3),
                                        transforms.RandomGrayscale()])
                                    ])

    return transform


def eval_augmentations():
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.CenterCrop(size=224),
                                    ])

    return transform
