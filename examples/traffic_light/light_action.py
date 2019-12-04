from enum import Enum

from .util import Acceleration
from .util import action_to_index, index_to_action

from pomdpy.discrete_pomdp import DiscreteAction

class TrafficLightAction(DiscreteAction):
    def __init__(self, index):
        super().__init__(index)
        self.index = index

    def distance_to(self, other_action):
        pass

    def copy(self):
        return TrafficLightAction(self.index)

    def print_action(self):
        print(self.to_string())

    def to_string(self):
        return str(self.index)

    def __repr__(self):
        return "{}".format(self.index)
