import json

class Options:
    def __init__(self, options_path):
        with open(options_path, 'r') as f:
            self.options = json.load(f)

    def get(self, key):
        return self.options[key]

    def optimizer_opts(self):
        return self.get("optimizer")

    def model_opts(self):
        return self.get("model")

    def data_opts(self):
        return self.get("data")

    def scheduler_opts(self):
        return self.get("scheduler")

    def training_opts(self):
        return self.get("training")
    