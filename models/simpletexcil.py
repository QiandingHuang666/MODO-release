from models.clean_baseline import TextPrototypeClassifierLearner


class Learner(TextPrototypeClassifierLearner):
    def __init__(self, args):
        super().__init__(args, network_type="clip")
