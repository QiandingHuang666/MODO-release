from models.klda_learner import KLDALearner

class Learner(KLDALearner):
    def __init__(self, args):
        super().__init__(args, network_type="clip")
