from models.clean_baseline import ParametricClassifierLearner


class Learner(ParametricClassifierLearner):
    def __init__(self, args):
        super().__init__(args, head_type="fc")
