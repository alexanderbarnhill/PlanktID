from torch.utils.tensorboard import SummaryWriter
import os
from metrics import Accuracy, Loss
from options import Options
from trainer import Trainer
from utilities.system import get_training_directory

if __name__ == '__main__':
    options_file = os.path.join(os.getcwd(), "options.json")
    options = Options(options_file)
    training_dir = get_training_directory(base_directory=options.training_opts()["training_directory"])
    writer = SummaryWriter(log_dir=training_dir, flush_secs=10)

    metrics = {
        'loss': Loss(writer=writer),
        'accuracy': Accuracy(writer=writer)
    }

    trainer = Trainer(
        metrics=metrics

    )