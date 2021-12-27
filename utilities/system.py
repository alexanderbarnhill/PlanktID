import torch
import os
from datetime import date


def make_folder_if_not_exist(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def get_device(use_cuda=True):
    return torch.device("cuda") if torch.cuda.is_available() and use_cuda else torch.device("cpu")


def get_training_directory(base_directory, description=None, model="CNN", suffix=None, run_count=False):

    experiment = description.replace(" ", "-") if description is not None else ""
    dir_prefix = f"{model}-{experiment}-{date.today().strftime('%Y-%m-%d')}"

    if not os.path.isdir(base_directory):
        os.makedirs(base_directory)

    dirs = [d for d in os.listdir(base_directory) if
            os.path.isdir(os.path.join(base_directory, d)) and dir_prefix in d]
    dir_suffix = f"_{suffix}" if suffix is not None else ""
    run = f"-run-{len(dirs) + 1}" if run_count else ""
    dir_name = f"{dir_prefix}" + run + dir_suffix
    directory = os.path.join(base_directory, dir_name)
    make_folder_if_not_exist(os.path.join(base_directory, dir_name))
    return directory