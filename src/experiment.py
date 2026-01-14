from omegaconf import DictConfig

class Experiment:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.trainer = None
        self.validator = None
        self.evaluator = None

    def run(self):
        # This method will orchestrate the experiment
        pass
