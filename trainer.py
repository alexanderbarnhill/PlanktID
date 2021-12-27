from typing import Tuple

import torch
import torch.nn as nn
import numpy as np


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
                 opts: Tuple[FeatureOptions, LabelOptions],
                 epoch_save_count=10,
                 cuda=True
                 ):
        self.model = model
        self.train_data_loader, self.val_data_loader, self.test_data_loader = data_loaders
        self.feature_options, self.label_options = opts
        self.device = get_device(cuda)

        self.epoch_count = n_epochs
        self.training_dir = training_directory
        self.writer = SummaryWriter(log_dir=self.training_dir, flush_secs=10)
        self.patience = patience_early_stopping
        self.epoch_save_count = epoch_save_count

        self.optimizer = optimizer
        self.criterion = criterion

        self.scheduler = scheduler

    def fit(self):
        for epoch in self.epoch_count:
            train_loss, train_predictions = self.train_epoch()
            report(train_loss, train_predictions, self.writer, constants.TRAIN)

            val_loss, val_predictions = self.val_test_epoch(self.val_data_loader)
            report(val_loss, val_predictions, self.writer, constants.VAL)

            if epoch % self.epoch_save_count == 0:
                self.save_model(epoch)

        test_loss, test_predictions = self.val_test_epoch(self.test_data_loader)

        return self.model

    def save_model(self, epoch_count):
        pass

    def train_epoch(self):
        self.model.train()
        running_loss = []
        predictions = []
        for data in self.train_data_loader:
            features, labels = data
            features.to(self.device)
            labels.to(self.device)

            loss, outputs = self.train_step(features, labels)
            running_loss.append(loss)
            predictions.append(outputs)

        return np.mean(running_loss), predictions

    def train_step(self, X, y):
        self.optimizer.zero_grad()
        model_output = self.model(X)
        loss = self.criterion(model_output, y)
        loss.backward()
        self.optimizer.step()

        return loss.item(), torch.nn.Sigmoid()(model_output)

    def val_test_epoch(self, data_loader):
        self.model.eval()
        running_loss = []
        predictions = []
        for data in self.val_data_loader:
            features, labels = data
            features.to(self.device)
            labels.to(self.device)

            loss, outputs = self.val_test_step(features, labels)
            running_loss.append(loss)
            predictions.append(outputs)

        self.scheduler.step(np.mean(running_loss))

        return np.mean(running_loss), predictions

    def val_test_step(self, X, y):
        with torch.no_grad():
            model_output = self.model(X)
            loss = self.criterion(model_output, y)

            return loss.item(), torch.nn.Sigmoid()(model_output)

