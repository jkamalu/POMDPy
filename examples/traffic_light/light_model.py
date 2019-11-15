import json
import numpy as np
from pomdpy.pomdp import model
from scipy.stats import truncnorm
from .light_action import TrafficLightAction, Acceleration
from .light_state import TrafficLightState
from .light_observation import TrafficLightObservation
from .light_data import TrafficLightData
from util import *
from pomdpy.discrete_pomdp import DiscreteActionPool
from pomdpy.discrete_pomdp import DiscreteObservationPool

class TrafficLightModel(model.Model):

    def __init__(self, problem_name="TrafficLight"):
        super().__init__(problem_name)
        self.num_actions = len(Acceleration)
        self.config = json.load("config.json")

    def start_scenario(self):
        position = self.config["init_position"]
        velocity = self.config["init_velocity"]
        light = self.config["init_light"]
        return TrafficLightState(position, velocity, light)

    ''' --------- Abstract Methods --------- '''

    def is_terminal(self, state):
        return state.position >= self.road_length + self.intersection_length

    def sample_an_init_state(self):
        return self.sample_state_uninformed()

    def create_observation_pool(self, solver):
        return DiscreteObservationPool(solver)

    def sample_state_uninformed(self):
        random_position = np.random.randint(self.config["road_length"] // 2)
        random_speed = np.random.randint(self.config["speed_upper_bound"])
        random_light = np.random.randint(sum(self.config["light_cycle"]))
        return TrafficLightState(random_position, random_speed, random_light)

    def get_all_states(self):
        states =
        for position in range(len(self.road_length)):
            for speed in range(self.max_speed):
                for light in range(sum(self.light_cycle)):
                    states.append(TrafficLightState(position, speed, light))
        return states

    def get_all_actions(self):
        return [TrafficLightAction(accel.value) for accel in Acceleration]

    def get_all_observations(self):
        wavelength_min = min(self.color_means) - 2 * self.color_stdev
        wavelength_max = max(self.color_means) + 2 * self.color_stdev

        distance_min = -(self.intersection_length + 2 * self.distance_stdev)
        distance_max = self.road_length + 2 * self.distance_stdev

        observations = []
        for distance_measurements in range(distance_min, distance_max + 1):
            for wavelength_measurements in range(wavelength_min, wavelength_max + 1):
                observations.append([distance_measurement, wavelength_measurements])

        return observations

    def get_legal_actions(self, state):
        legal_actions = []
        for a in Acceleration:
            if state.speed + a.value >= 0 and state.speed + a.value <= self.max_speed:
                legal_actions.append(a.value)
        return legal_actions

    def is_valid(self, state):
        return state.position >= 0

    def reset_for_simulation(self):
        self.start_scenario()

    def reset_for_epoch(self):
        self.start_scenario()

    def update(self, sim_data):
        pass

    def get_max_undiscounted_return(self):
        raise NotImplementedError("Max undiscounted reward needs to be defined.")

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
        for action in self.get_all_actions:
            state_state_combos = []
            for state in self.get_all_states:
                transition_state = state_transition(state, action)
                state_combos = []
                for state' in self.get_all_states:
                    value = 1 if state' == transition_state else 0
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
        for action in self.get_all_actions:
            for state in self.get_all_states:
                for observation in self.get_all_observations:
                color = -1
                light_range = 0
                while(state.light > light_range):
                    color += 1
                    light_range += self.config["light_cycle"][color]
                for observation in self.get_all_observations:


                TrafficLightState(position, speed, light)


    @staticmethod
    def get_reward_matrix():
        """
        |A| x |S| matrix
        :return:
        """
        return np.array([
            [-1., -1.],
            [-20.0, 10.0],
            [10.0, -20.0]
        ])

    @staticmethod
    def get_initial_belief_state():
        return np.array([0.5, 0.5])

    ''' Factory methods '''

    def create_action_pool(self):
        return DiscreteActionPool(self)

    def create_root_historical_data(self, agent):
        return TigerData(self)

    ''' --------- BLACK BOX GENERATION --------- '''

    def generate_step(self, action, state=None):
        if action is None:
            print("ERROR: Tried to generate a step with a null action")
            return None
        elif not isinstance(action, TigerAction):
            action = TigerAction(action)

        result = model.StepResult()
        result.is_terminal = self.make_next_state(action)
        result.action = action.copy()
        result.observation = self.make_observation(action)
        result.reward = self.make_reward(action, result.is_terminal)

        return result

    @staticmethod
    def make_next_state(action):
        if action.bin_number == ActionType.LISTEN:
            return False
        else:
            return True

    def make_reward(self, action, is_terminal):
        """
        :param action:
        :param is_terminal:
        :return: reward
        """

        if action.bin_number == ActionType.LISTEN:
            return -1.0

        if is_terminal:
            assert action.bin_number > 0
            if action.bin_number == self.tiger_door:
                ''' You chose the door with the tiger '''
                # return -20
                return -20.
            else:
                ''' You chose the door with the prize! '''
                return 10.0
        else:
            print("make_reward - Illegal action was used")
            return 0.0

    def make_observation(self, action):
        """
        :param action:
        :return:
        """
        if action.bin_number > 0:
            '''
            No new information is gained by opening a door
            Since this action leads to a terminal state, we don't care
            about the observation
            '''
            return TigerObservation(None)
        else:
            obs = ([0, 1], [1, 0])[self.tiger_door == 1]
            probability_correct = np.random.uniform(0, 1)
            if probability_correct <= 0.85:
                return TigerObservation(obs)
            else:
                obs.reverse()
                return TigerObservation(obs)

    def belief_update(self, old_belief, action, observation):
        """
        Belief is a 2-element array, with element in pos 0 signifying probability that the tiger is behind door 1

        :param old_belief:
        :param action:
        :param observation:
        :return:
        """
        if action > 1:
            return old_belief

        probability_correct = 0.85
        probability_incorrect = 1.0 - probability_correct
        p1_prior = old_belief[0]
        p2_prior = old_belief[1]

        # Observation 1 - the roar came from door 0
        if observation.source_of_roar[0]:
            observation_probability = (probability_correct * p1_prior) + (probability_incorrect * p2_prior)
            p1_posterior = old_div((probability_correct * p1_prior),observation_probability)
            p2_posterior = old_div((probability_incorrect * p2_prior),observation_probability)
        # Observation 2 - the roar came from door 1
        else:
            observation_probability = (probability_incorrect * p1_prior) + (probability_correct * p2_prior)
            p1_posterior = probability_incorrect * p1_prior / observation_probability
            p2_posterior = probability_correct * p2_prior / observation_probability
        return np.array([p1_posterior, p2_posterior])
