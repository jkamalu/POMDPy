from pomdpy.discrete_pomdp import DiscreteState

class TrafficLightState(DiscreteState):

    def __init__(self, distance, speed, color):
        self.distance = distance
        self.speed = speed
        self.color = color

    def __eq__(self, other_state):
        equals = self.distance == other_state.distance
        equals = self.speed == other_state.speed and equals
        equals = self.color == other_state.color and equals
        return equals

    def __hash__(self):
        return hash((self.distance, self.speed, self.color))

    def __repr__(self):
        return "[{} light {} units from car going {} units/sec]".format(self.color, self.distance, self.speed)
