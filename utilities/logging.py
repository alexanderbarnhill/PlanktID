import logging
import logging.handlers
import os
import queue

from torch.utils.tensorboard import SummaryWriter


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    """
    Logger in order to track training, validation, and testing of the network
    """
    def __init__(self, name, debug=True, log_dir=None, do_log_name=False):
        level = logging.DEBUG if debug else logging.INFO
        fmt = "%(asctime)s"
        if do_log_name:
            fmt += "|%(name)s"
        fmt += "|%(levelname).1s|%(message)s"
        formatter = logging.Formatter(fmt=fmt, datefmt="%H:%M:%S")

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        handlers = [sh]

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            fs = logging.FileHandler(os.path.join(log_dir, name + ".log"), mode='w')
            fs.setFormatter(formatter)
            handlers.append(fs)

        self._queue = queue.Queue(1000)
        self._handler = logging.handlers.QueueHandler(self._queue)
        self._listener = logging.handlers.QueueListener(self._queue, *handlers)

        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.addHandler(self._handler)

        self._logger.propagate = False

        self._listener.start()
        self.tb = SummaryWriter(log_dir=log_dir, flush_secs=10)

    def close(self):
        self._listener.stop()
        self._handler.close()

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def add_scalar(self, identifier, value, epoch):
        self.tb.add_scalar(identifier, value, epoch)
        self._logger.info(f"Epoch {epoch} -- {identifier}: {value}")