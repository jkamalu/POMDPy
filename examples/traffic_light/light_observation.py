from pomdpy.discrete_pomdp import DiscreteObservation

from .util import observation_to_index, index_to_observation

class TrafficLightObservation(DiscreteObservation):

    def __init__(self, measurements):
        super().__init__(observation_to_index(measurements))
        self.wavelength_observed = measurements[0]
        self.distance_observed = measurements[1]
        self.speed = measurements[2]

    def copy(self):
        return TrafficLightObservation((self.wavelength_observed, self.distance_observed, self.speed))

    def equals(self, other_observation):
        return self.wavelength_observed == other_observation.wavelength_observed and self.distance_observed == other_observation.distance_observed and self.speed == other_observation.speed

    def distance_to(self, other_observation):
        return 0 if self.equals(other_observation) else 1

    def __hash__(self):
        return int(self.bin_number)

    def print_observation(self):
        print(self.to_string())

    def to_string(self):
        return "[{}-wavelength, {} units from car moving at {} units/sec]".format(self.wavelength_observed, self.distance_observed, self.speed)
