import torchvision as tv

from data.options import Options
from utilities import constants


def get_transforms(mode, options: Options):
    if mode == constants.TRAIN:
        return get_train_transform(options)


def get_train_transform(options: Options):
    augmentations = options.transform_opts()
    aug_keys = ['pil'] + list(augmentations.keys()) + ['tensor']
    aug_dicts = [{'active': True}] + list(augmentations.values()) + [{'active': True}]
    return tv.transforms.Compose([
        build_transform(d, k) for (d, k) in zip(aug_dicts, aug_keys)
    ])


def build_transform(transform_dict, key):
    if not transform_dict["active"]:
        return
    if key == 'resize':
        return tv.transforms.Resize((transform_dict['height'], transform_dict['width']))

    if key == 'horizontal_flip':
        return tv.transforms.RandomHorizontalFlip(p=transform_dict['probability'])

    if key == 'vertical_flip':
        return tv.transforms.RandomHorizontalFlip(p=transform_dict['probability'])

    if key == 'random_affine':
        return tv.transforms.RandomAffine(degrees=transform_dict['degrees'])

    if key == 'color_jitter':
        brightness = (transform_dict['brightness']['min'], transform_dict['brightness']['max']) if transform_dict['brightness']['active'] else 0
        contrast = (transform_dict['contrast']['min'], transform_dict['contrast']['max']) if transform_dict['contrast']['active'] else 0
        saturation = (transform_dict['saturation']['min'], transform_dict['saturation']['max']) if transform_dict['saturation']['active'] else 0
        hue = (transform_dict['hue']['min'], transform_dict['hue']['max']) if transform_dict['hue']['active'] else 0
        return tv.transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    if key == 'pil':
        return tv.transforms.ToPILImage()

    if key == 'tensor':
        return tv.transforms.ToTensor()

    return None

