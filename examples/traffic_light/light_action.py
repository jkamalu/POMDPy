from enum import Enum

from util import Acceleration

from pomdpy.discrete_pomdp import DiscreteAction

class TrafficLightAction(DiscreteAction):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "[Acceleration = {}]".format(Acceleration(self.value).name)
