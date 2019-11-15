from enum import Enum

from util import Acceleration

from pomdpy.discrete_pomdp import DiscreteAction

class TrafficLightAction(DiscreteAction):
    def __init__(self, value):
        self.value = value

    def copy(self):
        return TrafficLightAction(self.value)

    def print_action(self):
        print(self.to_string())

    def to_string(self):
        return "[Acceleration = {}]".format(Acceleration(self.value).name)
