import torch

import constants
from utilities.logging import Logger


class MetricBase:

    def __init__(self):
        pass

    def update(self, *args):
        pass


class Scalar(MetricBase):

    def __init__(self, name, logger: Logger = None):
        super(Scalar, self).__init__()
        self.name = name
        self.logger = logger

    def update(self, **kwargs):
        if self.logger is not None:
            identifier = f"{self.name}"
            if 'phase' in kwargs:
                identifier += f"/{kwargs['phase']}"

            self.logger.add_scalar(identifier, kwargs['value'], kwargs['epoch'])


class Accuracy(Scalar):

    def __init__(self, logger: Logger = None):
        super(Accuracy, self).__init__(constants.ACC, logger=logger)

    def update(self, labels, predictions, phase, epoch):
        with torch.no_grad():
            predictions = predictions.type_as(labels)
            is_correct = torch.eq(labels, predictions).float()
            accuracy = (is_correct.sum() / is_correct.numel()).item()
        super().update(value=accuracy, phase=phase, epoch=epoch)


class Loss(Scalar):
    def __init__(self, logger: Logger = None):
        super(Loss, self).__init__(constants.LOSS, logger=logger)
        self.logger = logger

    def update(self, loss, phase, epoch):
        super().update(value=loss, phase=phase, epoch=epoch)


class LearningRate(Scalar):

    def __init__(self, logger: Logger = None):
        super(LearningRate, self).__init__(constants.LR, logger=logger)

    def update(self, lr, phase, epoch):
        super().update(value=lr, epoch=epoch)


class Sample(MetricBase):
    _phase = None
    _epoch = None
    _img_data = None

    def __init__(self, logger: Logger = None):
        super(Sample, self).__init__()
        self.logger = logger

    def update(self, img_data, phase, epoch):
        if phase == self._phase and epoch == self._epoch:
            return
        self._img_data = img_data
        self._phase = phase
        self._epoch = epoch
        if self.logger is not None:
            self.logger.tb.add_images(f'Sample/{phase}', self._img_data, epoch)
