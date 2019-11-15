from __future__ import print_function
from pomdpy.discrete_pomdp import DiscreteObservation


class TrafficLightObservation(DiscreteObservation):
    """
    For num_doors = 2, there is an 85 % of hearing the roaring coming from the tiger door.
    There is a 15 % of hearing the roaring come from the reward door.

    source_of_roar[0] = 0 (door 1)
    source_of_roar[1] = 1 (door 2)
    or vice versa
    """

    def __init__(self, measurements):
        if source_of_roar is not None:
            super().__init__((1, 0)[source_of_roar[0]])
        else:
            super().__init__(-1)
        wavelength_observed = measurements[0]
        distance_observed = measurements[1]
        self.wavelength_observed = wavelength_observed
        self.distance_observed = distance_observed

    def copy(self):
        return TrafficLightObservation(self.wavelength_observed, self.distance_observed)

    def equals(self, other_observation):
        return self.wavelength_observed == other_observation.wavelength_observed and self.distance_observed == other_observation.distance_observed

    def distance_to(self, other_observation):
        return 0 if self.equals(other_observation) else 1

    def hash(self):
        return self.bin_number

    def print_observation(self):
        print(self.to_string())

    def to_string(self):
        return "Wavelength observed is " + str(self.wavelength_observed) + ", and distance observed is " + str(self.distance_observed)
