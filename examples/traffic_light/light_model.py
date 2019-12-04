import os
import json

import numpy as np
from scipy.stats import truncnorm

from .light_action import TrafficLightAction, Acceleration
from .light_state import TrafficLightState
from .light_observation import TrafficLightObservation
from .light_data import TrafficLightData, Belief
from .util import Acceleration, LightColor
from .util import max_distance, state_to_color_index, calculate_trunc_norm_prob
from .util import MIN_DISTANCE_OBS, MAX_DISTANCE_OBS, MIN_WAVELENGTH_OBS, MAX_WAVELENGTH_OBS, INDEX_TO_ACTION

from pomdpy.pomdp import model
from pomdpy.discrete_pomdp import DiscreteActionPool
from pomdpy.discrete_pomdp import DiscreteObservationPool

class TrafficLightModel(model.Model):

    def __init__(self, problem_name="TrafficLight"):
        super().__init__(problem_name)
        self.num_actions = len(Acceleration)
        path = os.path.join(*__name__.split('.')[:-1], "config.json")
        with open(path, "rt") as fp:
            self.config = json.load(fp)

        self.init_speed = self.config["init_speed"]

    def start_scenario(self):
        position = self.config["init_position"]
        speed = self.config["init_speed"]
        light = self.config["init_light"]
        return TrafficLightState(position, speed, light)

    ''' --------- Abstract Methods --------- '''

    def is_terminal(self, state):
        return state.position >= self.road_length + self.intersection_length

    def sample_an_init_state(self):
        random_position = np.random.randint(self.config["road_length"] // 2)
        speed = self.init_speed
        random_light = np.random.randint(sum(self.config["light_cycle"]))
        return TrafficLightState(random_position, speed, random_light)

    def create_observation_pool(self, solver):
        return DiscreteObservationPool(solver)

    def sample_state_uninformed(self):
        random_position = np.random.randint(self.config["road_length"] // 2)
        random_speed = np.random.randint(self.config["speed_limit"])
        random_light = np.random.randint(sum(self.config["light_cycle"]))
        return TrafficLightState(random_position, random_speed, random_light)

    def sample_state_informed(self, belief):
        return belief.sample_particle()

    def get_all_states(self):
        states = []
        for position in range(len(self.road_length)):
            for speed in range(self.max_speed):
                for light in range(sum(self.light_cycle)):
                    states.append(TrafficLightState(position, speed, light))
        return states

    def get_all_actions(self):
        return [TrafficLightAction(index) for index in INDEX_TO_ACTION]

    def get_all_observations(self):
        observations = []
        for distance_measurement in range(MIN_DISTANCE_OBS, MAX_DISTANCE_OBS + 1):
            for wavelength_measurement in range(MIN_WAVELENGTH_OBS, MAX_WAVELENGTH_OBS + 1):
                for speed in range(self.config["max_speed"] + 1):
                    observations.append(TrafficLightObservation((distance_measurement, wavelength_measurement, speed)))

        return observations

    def get_legal_actions(self, state):
        legal_actions = []
        for index in INDEX_TO_ACTION:
            if state.speed + INDEX_TO_ACTION[index] >= 0 and state.speed + INDEX_TO_ACTION[index] <= self.config["max_speed"]:
                legal_actions.append(TrafficLightAction(index))
        return legal_actions

    def is_valid(self, state):
        return state.position >= 0 and state.speed >= 0

    def reset_for_simulation(self):
        self.start_scenario()

    def reset_for_epoch(self):
        self.start_scenario()

    def update(self, sim_data):
        pass

    def get_max_undiscounted_return(self):
        return 10

    @staticmethod
    def state_transition(state, action):
        speed = state.speed + action
        position = state.position + speed
        light = (state.light) + 1 % sum(self.config["light_cycle"])
        new_state = TrafficLightState(position, speed, light)

    @staticmethod
    def get_transition_matrix():
        """
        |A| x |S| x |S'| matrix, for tiger problem this is 3 x 2 x 2
        :return:
        """
        action_state_state_combos = []
        for action in self.get_all_actions():
            state_state_combos = []
            for state in self.get_all_states():
                transition_state = state_transition(state, action)
                state_combos = []
                for state in self.get_all_states():
                    value = 1 if state == transition_state else 0
                    state_combos.append(value)
                state_state_combos.append(np.array(state_combos))
            action_state_combos.append(np.array(state_state_combos))
        return np.array(action_state_combos)


    @staticmethod
    def get_observation_matrix():
        """
        |A| x |S| x |O| matrix
        :return:
        """
        observations = []
        for action in self.get_all_actions():
            for state in self.get_all_states():
                state_obs_probs = []
                color = state_to_color_index(state)
                observation_probs = []
                for observation in self.get_all_observations():
                    if state.speed + INDEX_TO_ACTION(action.index) != observation.speed:
                        observation_probs.append(0)
                        continue
                    color_mean = self.config["color_means"][color]
                    color_std = self.config["color_stdev"]
                    color_probab = calculate_trunc_norm_prob(observation.wavelength_observed, color_mean, color_std, MIN_WAVELENGTH_OBS, MAX_WAVELENGTH_OBS)

                    dist_mean = state.position
                    dist_std = self.config["distance_stdev"]
                    distance_probab = calculate_trunc_norm_prob(observation.distance_observed, dist_mean, dist_std, MIN_DISTANCE_OBS, MAX_DISTANCE_OBS)
                    observation_probs.append(color_probab * distance_probab)

                state_obs_probs.append(np.array(observation_probs))
            observations.append(np.array(state_obs_probs))
        return np.array(observations)

    def get_reward_matrix(self):
        """
        |A| x |S| matrix
        :return:
        """
        reward_matrix = []
        for action in self.get_all_actions():
            state_rewards = []
            for state in self.get_all_states():
                terminal = state.position >= self.config["road_length"] + self.config["intersection_length"]
                state_rewards.append(self.make_reward(action, state, terminal))
            reward_matrix.append(np.array(state_rewards))
        return np.array(reward_matrix)

    @staticmethod
    def get_initial_belief_state():
        return Belief()

    ''' Factory methods '''

    def create_action_pool(self):
        return DiscreteActionPool(self)

    def create_root_historical_data(self, agent):
        return TrafficLightData(self, self.init_speed)

    ''' --------- BLACK BOX GENERATION --------- '''

    def generate_step(self, state, action):
        if action is None:
            print("ERROR: Tried to generate a step with a null action")
            return None
        elif not isinstance(action, TrafficLightAction):
            action = TrafficLightAction(action)

        result = model.StepResult()
        result.next_state, result.is_terminal = self.make_next_state(state, action)
        result.action = action.copy()
        result.observation = self.make_observation(action, result.next_state)
        result.reward = self.make_reward(action, result.next_state, result.is_terminal)

        return result, self.is_valid(state)

    def make_next_state(self, state, action):
        max_position = self.config["road_length"] + self.config["intersection_length"]
        terminal = state.position >= max_position

        new_speed = state.speed + INDEX_TO_ACTION[action.index]
        new_position = state.position + new_speed
        new_light = (state.light + 1) % sum(self.config["light_cycle"])

        new_state = TrafficLightState(new_position, new_speed, new_light)

        return new_state, terminal

    def make_reward(self, action, state, terminal):
        """
        :param action:
        :param is_terminal:
        :return: reward
        """
        if terminal:
            return 10
        ## Penalize for every timestep not at the goal state.
        rewards = -1
        ## Penalize if the car stops outside the buffer.
        if state.speed == 0 and (state.position > self.config["road_length"] or state.position < self.config["road_length"] - self.config["buffer_length"]):
            rewards -= 5
        ## Penalize if we're in the intersection on a red light.
        if state_to_color_index(state) == 2 and (state.position > self.config["road_length"] and state.position <= self.config["road_length"] + self.config["intersection_length"]):
            rewards -= 100
        ## Penalize for going over the speed limit.
        if state.speed > self.config["speed_limit"]:
            rewards -= (state.speed - self.config["speed_limit"])
        return rewards

    def make_observation(self, action, next_state):
        """
        :param action:
        :return:
        """
        color_index = state_to_color_index(next_state)
        color_mean = self.config["color_means"][color_index]
        color_stdev = self.config["color_stdev"]
        sampled_wavelength = truncnorm.rvs((MIN_WAVELENGTH_OBS - color_mean) / color_stdev, (MAX_WAVELENGTH_OBS - color_mean) / color_stdev, loc=color_mean, scale=color_stdev, size=1)
        wl = int(sampled_wavelength + 0.5)
        dist_mean = self.config["road_length"] - next_state.position
        dist_stdev = self.config["distance_stdev"]
        sampled_distance = truncnorm.rvs((MIN_DISTANCE_OBS - dist_mean) / dist_stdev, (MAX_DISTANCE_OBS - dist_mean) / dist_stdev, loc=dist_mean, scale=dist_stdev, size=1)
        try:
            dist = int(sampled_distance + 0.5)
        except:
            print("sampled_distance = -inf")
            dist = dist_mean
        return TrafficLightObservation((wl, dist, next_state.speed))

    def belief_update(self, old_belief, action, observation):
        if old_belief.dist is not None:
            b_dist = (old_belief.dist * old_belief.dist_confidence + observation.distance_observed * self.config["distance_stdev"]) / (old_belief.dist_confidence + self.config["distance_stdev"])
            b_dist_stdev = (old_belief.dist_confidence * self.config["distance_stdev"]) / (old_belief.dist_confidence + self.config["distance_stdev"])
        else:
            b_dist = (observation.distance_observed * self.config["distance_stdev"]) / self.config["distance_stdev"]
            b_dist_stdev = self.config["distance_stdev"]
        b_dist += observation.speed
        b_colors = [old_belief.green, old_belief.yellow, old_belief.red]
        for color in LightColor:
            color_mean = self.config["color_means"][color.value]
            color_stdev = self.config["color_stdev"]
            color_probab = calculate_trunc_norm_prob(observation.wavelength_observed, color_mean, color_stdev, MIN_WAVELENGTH_OBS, MAX_WAVELENGTH_OBS)
            b_colors[color.value] *= color_probab
        new_belief = Belief(p_green=b_colors[0], p_yellow=b_colors[1], p_red=b_colors[2], belief_d=b_dist, confidence_d=b_dist_stdev)
        new_belief.normalize()
        return new_belief
