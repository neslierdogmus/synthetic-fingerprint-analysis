class Metric:
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        raise NotImplementedError

class Accuracy(Metric):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        return (y_pred.argmax(dim=1) == y_true).float().mean()
