from __future__ import print_function
from builtins import object
from pomdpy.discrete_pomdp import DiscreteAction


class ActionType(object):
    """
    Lists the possible actions and attributes an integer code to each for the Rock sample problem
    """
    DECELERATE_50 = 0
    DECELERATE_20 = 1
    DECELERATE_5 = 2
    NO_ACCELERATION = 3
    ACCELERATE_5 = 4
    ACCELERATE_20 = 5
    ACCELERATE_50 = 6


class AccelerationAction(DiscreteAction):
    """
    -The Rock sample problem Action class
    -Wrapper for storing the bin number. Also stores the rock number for checking actions
    -Handles pretty printing
    """

    def __init__(self, bin_number):
        super(RockAction, self).__init__(bin_number)
        self.bin_number = action_type

    def copy(self):
        return AccelerationAction(self.bin_number)

    def to_string(self):
        if self.bin_number is ActionType.DECELERATE_50:
            action = "Decelerate by 50"
        elif self.bin_number is ActionType.DECELERATE_20:
            action = "Decelerate by 20"
        elif self.bin_number is ActionType.DECELERATE_5:
            action = "Decelerate by 5"
        elif self.bin_number is ActionType.NO_ACCELERATION:
            action = "No acceleration"
        elif self.bin_number is ActionType.ACCELERATE_5:
            action = "Accelerate by 5"
        elif self.bin_number is ActionType.ACCELERATE_20:
            action = "Accelerate by 20"
        elif self.bin_number is ActionType.ACCELERATE_50:
            action = "Accelerate by 50"
        else:
            action = "Unknown action type"
        return action

    def print_action(self):
        print(self.to_string())

    def distance_to(self, other_point):
        pass
