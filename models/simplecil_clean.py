from models.clean_baseline import PrototypeClassifierLearner


class Learner(PrototypeClassifierLearner):
    def __init__(self, args):
        super().__init__(args, network_type="clip")
