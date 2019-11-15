from pomdpy.pomdp import HistoricalData

class Belief():
    def __init__(self, p_green=float(1/3), p_yellow=float(1/3), p_red=float(1/3), belief_d=None, confidence_d = None):
        self.green = p_green
        self.yellow = p_yellow
        self.red = p_red
        self.dist = belief_d
        self.dist_confidence = confidence_d

    def __eq__(self, other):
        if self.green == other.green and self.yellow == other.yellow and self.red == other.red and self.dist == other.dist and self.dist_confidence = other.dist_confidence:
            return True
        return False

    def update(self, other):
        self.green = other.green
        self.yellow = other.yellow
        self.red = other.red
        self.dist = other.dist
        self.dist_confidence = other.dist_confidence

    def normalize(self):
        total = self.green + self.yellow + self.red
        self.green /= total
        self.yellow /= total
        self.red /= total

class TrafficLightData(HistoricalData):

    def __init__(self, model, belief=Belief()):
        self.model = model
        self.observations_passed = 0
        self.belief = belief

    def copy(self):
        dat = TrafficLightData(self.model)
        dat.observations_passed = self.observations_passed
        dat.belief = self.belief
        return dat

    def update(self, other_belief):
        self.belief.update(other_belief.belief)

    def create_child(self, action, observation):
        next_data = self.copy()

        self.observations_passed += 1

        ''' ------- Bayes update of belief state -------- '''

        next_data.belief = self.model.belief_update(self.belief, action, observation)
        return next_data

    def generate_legal_actions(self):
        """
        At each non-terminal state, the agent can listen or choose to open the
        door based on the current door probabilities
        """

        return self.model.get_all_actions
