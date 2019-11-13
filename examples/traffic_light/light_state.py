from pomdpy.discrete_pomdp import DiscreteState

class TrafficLightState(DiscreteState):

    def __init__(self, position, speed, light):
        self.position = position
        self.speed = speed
        self.light = light

    def __eq__(self, other_state):
        equals = self.position == other_state.position
        equals = self.speed == other_state.speed and equals
        equals = self.light == other_state.light and equals
        return equals

    def __hash__(self):
        return hash((self.position, self.speed, self.light))

    def __repr__(self):
        return "[{} light {} units from car going {} units/sec]".format(self.light, self.position, self.speed)
