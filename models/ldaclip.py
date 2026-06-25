from models.clean_baseline import LDAClassifierLearner


class Learner(LDAClassifierLearner):
    def __init__(self, args):
        super().__init__(args, network_type="clip")
