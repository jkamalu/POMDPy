from pomdpy.discrete_pomdp import DiscreteState

class TrafficLightState(DiscreteState):

    def __init__(self, position, speed, light):
        self.position = position
        self.speed = speed
        self.light = light

    def copy(self):
        return TrafficLightState(self.position, self.speed, self.light)

    def as_list(self):
        return [self.position, self.speed, self.light]

    def equals(self, other_state):
        equals = self.position == other_state.position
        equals = self.speed == other_state.speed and equals
        equals = self.light == other_state.light and equals
        return equals

    def hash(self):
        return hash((self.position, self.speed, self.light))

    def print_state(self):
        print(self.to_string())

    def to_string(self):
        return "[{}-light, car {} units from start moving at {} units/sec]".format(self.light, self.position, self.speed)
