from models.clean_baseline import GDAClassifierLearner


class Learner(GDAClassifierLearner):
    def __init__(self, args):
        super().__init__(args, network_type="clip")
