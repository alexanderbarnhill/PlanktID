import math

from torch import optim
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os

import constants
from data.dataset import CSVDataset
from data.options import Options
from metrics import Accuracy, Loss
from model.resnet import ResNet

from trainer import Trainer
from utilities.logging import Logger
from utilities.system import get_training_directory, get_device


def get_model(opts: Options):
    return ResNet(opts)


def get_optimizer(m, opts: Options):
    optimizer_opts = opts.optimizer_opts()
    return Adam(params=filter(lambda p: p.requires_grad, m.parameters()),
                lr=optimizer_opts["learning_rate"],
                betas=(optimizer_opts["beta_1"], optimizer_opts["beta_2"]))


def get_data_loaders(opts: Options):
    data_options = opts.data_opts()
    train_dataset = CSVDataset(opts, constants.TRAIN)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=data_options["batch_size"],
                                   num_workers=data_options["num_workers"],
                                   shuffle=True)

    val_dataset = CSVDataset(opts, constants.VAL)
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=data_options["batch_size"],
                                 num_workers=data_options["num_workers"],
                                 shuffle=True)

    test_dataset = CSVDataset(opts, constants.TEST)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=data_options["batch_size"],
                                  num_workers=data_options["num_workers"],
                                  shuffle=False)

    return train_data_loader, val_data_loader, test_data_loader


def get_logger(directory):
    return Logger(
        name="TRAIN",
        log_dir=directory
    )


if __name__ == '__main__':
    options_file = os.path.join(os.getcwd(), "options.json")
    options = Options(options_file)
    training_dir = get_training_directory(base_directory=options.training_opts()["training_directory"])
    writer = SummaryWriter(log_dir=training_dir, flush_secs=10)

    model = get_model(options)
    optimizer = get_optimizer(model, options)

    patience_lr = math.ceil(
        options.scheduler_opts()["lr_patience_epochs"] / options.scheduler_opts()["epochs_per_eval"])
    patience_lr = int(max(1, patience_lr))

    metrics = {
        'loss': Loss(writer=writer),
        'accuracy': Accuracy(writer=writer)
    }

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=patience_lr,
        factor=options.scheduler_opts()["lr_decay_factor"],
        threshold=1e-3,
        threshold_mode="abs",
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_loaders=get_data_loaders(options),
        training_directory=training_dir,
        criterion=nn.CrossEntropyLoss(),
        scheduler=lr_scheduler,
        n_epochs=options.training_opts()["epochs"],
        patience_early_stopping=options.training_opts()["patience_early_stopping"],
        logger=get_logger(training_dir),
        summary_writer=writer,
        metrics=metrics,
        epoch_save_count=options.training_opts()["epoch_save_count"],
        device=get_device(options.training_opts()["use_cuda"])
    )

    trainer.fit()
