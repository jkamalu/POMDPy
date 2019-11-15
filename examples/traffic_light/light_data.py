from __future__ import absolute_import
from pomdpy.pomdp import HistoricalData
from .light_action import ActionType
import numpy as np
from enum import Enum

class Belief():
    def __init__(self, p_green=float(1/3), p_yellow=float(1/3), p_red=float(1/3), belief_d=None, confidence_d = None):
        self.green = p_green
        self.yellow = p_yellow
        self.red = p_red
        self.dist = belief_d
        self.dist_confidence = confidence_d

    def __eq__(self, other):
        if self.green == other.green and self.yellow == other.yellow and self.red == other.red and self.dist == other.dist and self.dist_confidence = other.dist_confidence:
            return True
        return False

    def update(self, other):
        self.green = other.green
        self.yellow = other.yellow
        self.red = other.red
        self.dist = other.dist
        self.dist_confidence = other.dist_confidence

class TrafficLightData(HistoricalData):

    def __init__(self, model):
        self.model = model
        self.observations_passed = 0
        self.belief = Belief()
        self.color_probabilities = [float(1/3), float(1/3), float(1/3)]
        self.distance_belief = None
        self.distance_confidence = None

    def copy(self):
        dat = TrafficLightData(self.model)
        dat.observations_passed = self.observations_passed
        dat.belief = self.beloef
        return dat

    def update(self, other_belief):
        self.belief.update(other_belief.belief)

    def create_child(self, action, observation):
        next_data = self.copy()

        self.observations_passed += 1

        ''' ------- Bayes update of belief state -------- '''
        belief_update(self, old_belief, action, observation)

        next_data.color_probabilities = self.model.belief_update((self.color_probabilities, self.distance_belief, self.distance_confidence)
        next_data.distance_belief = action
        next_data.distance_confidence = observation

        return next_data

    @staticmethod
    def generate_legal_actions():
        """
        At each non-terminal state, the agent can listen or choose to open the
        door based on the current door probabilities
        """

        raise NotImplementedError("Actions legality is function of state.")
