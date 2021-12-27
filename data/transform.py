import torchvision as tv
from utilities import constants


def get_transforms(mode):
    if mode == constants.TRAIN:
        return get_train_transform()


def get_train_transform():
    return tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.Resize((128, 128)),
        tv.transforms.RandomHorizontalFlip(p=0.5),
        tv.transforms.RandomVerticalFlip(p=0.5),
        tv.transforms.RandomAffine(20),
        tv.transforms.ColorJitter(brightness=0.2, contrast=0.7, saturation=0.2, hue=0.2),
        tv.transforms.ToTensor(),
    ])
