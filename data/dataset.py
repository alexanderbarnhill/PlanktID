import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
import numpy as np
import os
import random
import torchvision as tv
from skimage.io import imread

from data.options import Options
from data.transform import get_transforms
from utilities import constants


class CSVDatasetSplit:
    classes = []
    class_dict = {}
    train_files = []
    val_files = []
    test_files = []

    def __init__(self, data_directory, train_split=0.75):
        self.data_directory = data_directory
        self.train_split = train_split
        self.train_file = os.path.join(self.data_directory, constants.TRAIN)
        self.val_file = os.path.join(self.data_directory, constants.VAL)
        self.test_file = os.path.join(self.data_directory, constants.TEST)

    def __call__(self):
        return self.split() if not self.to_split() else self.get_split()

    def get_split(self):
        return self.train_file, self.val_file, self.test_file

    def to_split(self):
        return os.path.isfile(os.path.join(self.data_directory, constants.TRAIN))

    def split(self):
        self.classes = [os.path.join(self.data_directory, folder) for folder in os.listdir(self.data_directory)
                        if os.path.isdir(os.path.join(self.data_directory, folder))]
        for c in self.classes:
            files = [os.path.join(c, file) for file in os.listdir(c) if os.path.isfile(os.path.join(c, file))
                     and file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png")]
            train_count = int(len(files) * self.train_split)
            train_files = random.sample(files, train_count)

            val_test_count = len(files) - train_count
            val_files = random.sample(list(np.setdiff1d(files, train_files)), int(val_test_count // 2))
            test_files = list(np.setdiff1d(files, train_files + val_files))
            self.train_files.append(os.path.join(c, f'{constants.TRAIN}.csv'))
            self.val_files.append(os.path.join(c, f'{constants.VAL}.csv'))
            self.test_files.append(os.path.join(c, f'{constants.TEST}.csv'))

            train_df = pd.DataFrame({'files': train_files})
            train_df.to_csv(os.path.join(c, f'{constants.TRAIN}.csv'), index=False, header=False)

            val_df = pd.DataFrame({'files': val_files})
            val_df.to_csv(os.path.join(c, f'{constants.VAL}.csv'), index=False, header=False)

            test_df = pd.DataFrame({'files': test_files})
            test_df.to_csv(os.path.join(c, f'{constants.TEST}.csv'), index=False, header=False)


        pd.DataFrame({'files': self.train_files}).to_csv(self.train_file, index=False, header=False)
        pd.DataFrame({'files': self.val_files}).to_csv(self.val_file, index=False, header=False)
        pd.DataFrame({'files': self.test_files}).to_csv(self.test_file, index=False, header=False)

        return self.train_file, self.val_file, self.test_file


class CSVDataset(Dataset):
    classes = []
    class_dict = {}

    indicator = None
    data = None

    def __init__(self, options: Options, mode=constants.TRAIN):
        self.data_options = options.data_opts()
        train_file, val_file, test_file = CSVDatasetSplit(
            self.data_options["data_directory"], self.data_options["train_split"])()
        self.indicator = train_file if mode == constants.TRAIN else val_file if mode == constants.VAL else test_file
        self.data = []
        self.transform = get_transforms(mode, options)
        class_dict_idx = 0

        with open(self.indicator, "r") as f:
            self.files = f.readlines()
            for file in sorted(self.files):
                df = pd.read_csv(file.rstrip())
                for _, row in df.iterrows():
                    file_name = row[0]
                    class_name = file_name.split('/')[-2]
                    self.data.append((file_name, class_name))
                    if class_name not in self.class_dict:
                        self.class_dict[class_name] = class_dict_idx
                        class_dict_idx += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        file, class_name = self.data[index]
        return self.transform(imread(file)), self.class_dict[class_name]
