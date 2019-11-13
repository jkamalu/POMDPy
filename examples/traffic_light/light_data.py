from __future__ import absolute_import
from pomdpy.pomdp import HistoricalData
from .light_action import ActionType
import numpy as np
from enum import Enum
import util

class LightColor(Enum):
    GREEN = 0
    YELLOW = 1
    RED = 2 

class LightData(HistoricalData):
    """
    Used to store the probabilities that the tiger is behind a certain door.
    This is the belief distribution over the set of possible states.
    For a 2-door system, you have
        P( X = 0 ) = p
        P( X = 1 ) = 1 - p
    """
    def __init__(self, model):
        self.model = model
        self.observations_passed = 0
        ''' Initially there is an equal probability of the tiger being in either door'''
        self.color_probabilities = [float(1/3), float(1/3), float(1/3)]
        self.distance_belief = None
        self.distance_confidence = None

    def copy(self):
        dat = LightData(self.model)
        dat.observations_passed = self.observations_passed
        dat.color_probabilities = self.color_probabilities
        dat.distance_belief = self.distance_belief
        dat.distance_confidence = self.distance_confidence
        return dat

    def update(self, other_belief):
        self.color_probabilities = other_belief.data.color_probabilities
        self.distance_belief = other_belief.data.distance_belief
        self.distance_confidence = other_belief.data.distance_belief

    def create_child(self, action, observation):
        next_data = self.copy()

        self.observations_passed += 1
        '''
        '''

        ''' ------- Bayes update of belief state -------- '''
        belief_update(self, old_belief, action, observation)
        (next_data.color_probabilities, next_data.distance_belief, next_data.distance_confidence) = self.model.belief_update((self.color_probabilities, self.distance_belief, self.distance_confidence), action,
                                                                observation)
        
        return next_data

    @staticmethod
    def generate_legal_actions():
        """
        At each non-terminal state, the agent can listen or choose to open the door based on the current door probabilities
        :return:
        """

        return [ActionType.DECELERATE_50, ActionType.DECELERATE_20, ActionType.DECELERATE_5, ActionType.NO_ACCELERATION, ActionType.ACCELERATE_5, ActionType.ACCELERATE_20, ActionType.ACCELERATE_50]

