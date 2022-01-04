import json
from typing import Dict

from utilities.logging import Logger


class Options:
    def __init__(self, options_path):
        with open(options_path, 'r') as f:
            self.options = json.load(f)

    def get(self, key):
        return self.options[key]

    def optimizer_opts(self) -> Dict:
        return self.get("optimizer")

    def model_opts(self) -> Dict:
        return self.get("model")

    def data_opts(self) -> Dict:
        return self.get("data")

    def scheduler_opts(self) -> Dict:
        return self.get("scheduler")

    def training_opts(self) -> Dict:
        return self.get("training")

    def transform_opts(self) -> Dict:
        return self.get("augmentations")

    def rec_log(self, key, logger: Logger):
        logger.info(f"{key.capitalize()} Options")
        logger.info(json.dumps(self.options[key], sort_keys=True, indent=4))

    def log(self, logger: Logger):
        for k in self.options.keys():
            self.rec_log(k, logger)
