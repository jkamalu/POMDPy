from enum import Enum

from pomdpy.discrete_pomdp import DiscreteAction

class Acceleration(Enum):
    NEGATIVE_LRG = -3
    NEGATIVE_MED = -2
    NEGATIVE_SML = -1
    ZERO = 0
    POSITIVE_LRG = 1
    POSITIVE_MED = 2
    POSITIVE_SML = 3

class TrafficLightAction(DiscreteAction):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "[Acceleration = {}]".format(Acceleration(self.value).name)
