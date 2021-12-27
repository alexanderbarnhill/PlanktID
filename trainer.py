from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

import constants
from metrics import MetricBase


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer,
                 data_loaders: Tuple[DataLoader, DataLoader, DataLoader],
                 training_directory: str,
                 criterion: _Loss,
                 scheduler,
                 n_epochs,
                 patience_early_stopping,
                 logger,
                 summary_writer,
                 device,
                 metrics: Dict[str, MetricBase],
                 epoch_save_count=10
                 ):
        self.model = model
        self.train_data_loader, self.val_data_loader, self.test_data_loader = data_loaders
        self.device = device

        self.epoch_count = n_epochs
        self.training_dir = training_directory
        self.logger = logger
        self.writer = summary_writer
        self.patience = patience_early_stopping
        self.epoch_save_count = epoch_save_count
        self.metrics = metrics
        self.optimizer = optimizer
        self.criterion = criterion

        self.scheduler = scheduler

    def report(self, loss, labels, predictions, phase, epoch):
        self.metrics["loss"].update(loss, phase, epoch)
        self.metrics["accuracy"].update(labels, predictions, phase, epoch)

    def fit(self):
        for epoch in range(self.epoch_count):
            train_loss, train_labels, train_predictions = self.train_epoch()
            self.report(train_loss, train_labels, train_predictions, constants.TRAIN, epoch)

            val_loss, val_labels, val_predictions = self.val_test_epoch(self.val_data_loader)
            self.report(val_loss, val_labels, val_predictions, constants.VAL, epoch)

            if epoch % self.epoch_save_count == 0:
                self.save_model(epoch)

        test_loss, test_labels, test_predictions = self.val_test_epoch(self.test_data_loader)
        self.report(test_loss, test_labels, test_labels, constants.TEST, self.epoch_count)

        return self.model

    def save_model(self, epoch_count):
        pass

    def train_epoch(self):
        self.model.train()
        running_loss = []
        all_predictions = torch.tensor([])
        all_labels = torch.tensor([])
        for data in self.train_data_loader:
            features, labels = data
            features.to(self.device)
            labels.to(self.device)

            loss, outputs = self.train_step(features, labels)
            running_loss.append(loss)

            _, pred = torch.max(outputs, 1)
            all_predictions = torch.cat((all_predictions, pred))
            all_labels = torch.cat((all_labels, labels))

        return np.mean(running_loss), all_labels, all_predictions

    def train_step(self, X, y):
        self.optimizer.zero_grad()
        model_output = self.model(X)
        loss = self.criterion(model_output, y)
        loss.backward()
        self.optimizer.step()

        return loss.item(), torch.nn.Softmax(dim=1)(model_output)

    def val_test_epoch(self, data_loader):
        self.model.eval()
        running_loss = []
        all_predictions = torch.tensor([])
        all_labels = torch.tensor([])
        for data in data_loader:
            features, labels = data
            features.to(self.device)
            labels.to(self.device)

            loss, outputs = self.val_test_step(features, labels)
            running_loss.append(loss)

            _, pred = torch.max(outputs, 1)
            all_predictions = torch.cat((all_predictions, pred))
            all_labels = torch.cat((all_labels, labels))

        self.scheduler.step(np.mean(running_loss))

        return np.mean(running_loss), all_labels, all_predictions

    def val_test_step(self, X, y):
        with torch.no_grad():
            model_output = self.model(X)
            loss = self.criterion(model_output, y)

            return loss.item(), torch.nn.Softmax(dim=1)(model_output)

