import torch


class MetricBase:

    def __init__(self):
        pass

    def update(self, *args):
        pass


class Accuracy(MetricBase):
    _value = None

    def __init__(self, writer=None):
        super(Accuracy, self).__init__()
        self.writer = writer

    def update(self, labels, predictions, phase, epoch):
        with torch.no_grad():
            predictions = predictions.type_as(labels)
            is_correct = torch.eq(labels, predictions).float()
            self._value = (is_correct.sum() / is_correct.numel()).item()
        if self.writer is not None:
            self.writer.add_scalar(f'Accuracy/{phase}', self._value, epoch)


class Loss(MetricBase):
    _value = None

    def __init__(self, writer=None):
        super(Loss, self).__init__()
        self.writer = writer

    def update(self, loss, phase, epoch):
        self._value = loss
        if self.writer is not None:
            self.writer.add_scalar(f'Accuracy/{phase}', self._value, epoch)