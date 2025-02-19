from pomdpy.pomdp import HistoricalData

from .util import INDEX_TO_ACTION

class Belief():
    def __init__(self, p_green=float(1/3), p_yellow=float(1/3), p_red=float(1/3), belief_d=None, confidence_d = None):
        self.green = p_green
        self.yellow = p_yellow
        self.red = p_red
        self.dist = belief_d
        self.dist_confidence = confidence_d

    def __eq__(self, other):
        if self.green == other.green and self.yellow == other.yellow and self.red == other.red and self.dist == other.dist and self.dist_confidence == other.dist_confidence:
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

    def __init__(self, model, speed, belief=Belief()):
        self.model = model
        self.observations_passed = 0
        self.belief = belief
        self.speed = speed

        self.legal_actions = self.generate_legal_actions

    def copy(self):
        dat = TrafficLightData(self.model, self.speed, self.belief)
        dat.observations_passed = self.observations_passed
        return dat

    def update(self, other_belief):
        self.belief.update(other_belief.belief)

    def create_child(self, action, observation):
        next_data = self.copy()

        self.observations_passed += 1

        ''' ------- Bayes update of belief state -------- '''

        next_data.belief = self.model.belief_update(self.belief, action, observation)
        next_data.speed = observation.speed
        return next_data

    def generate_legal_actions(self):
        """
        At each non-terminal state, the agent can listen or choose to open the
        door based on the current door probabilities
        """

        legal_actions = []
        for index in INDEX_TO_ACTION:
            if self.speed + INDEX_TO_ACTION[index] >= 0 and self.speed + INDEX_TO_ACTION[index] <= self.model.config["max_speed"]:
                legal_actions.append(index)

        return legal_actions
