import random
import nump as np

import json

config = json.load("config.json")

MIN_WAVELENGTH = min(config["color_means"])
MAX_WAVELENGTH = max(config["color_means"])

MIN_WAVELENGTH_OBS = MIN_WAVELENGTH - 2 * config["color_stdev"]
MAX_WAVELENGTH_OBS = MAX_WAVELENGTH + 2 * config["color_stdev"]
MAX_DISTANCE_OBS = config["road_length"] + 2 * config["distance_stdev"]
MIN_DISTANCE_OBS = config["intersection_length"] - 2 * config["distance_stdev"]

class Acceleration(Enum):
    NEGATIVE_LRG = -8
    NEGATIVE_MED = -4
    NEGATIVE_SML = -2
    NEGATIVE_ONE = -1
    ZERO = 0
    POSITIVE_ONE = 1
    POSITIVE_SML = 2
    POSITIVE_MED = 4
    POSITIVE_LRG = 8

class LightColor(Enum):
    GREEN = 0
    YELLOW = 1
    RED = 2

def pdf(mean, std, value):
    u = float(value - mean) / abs(std)
    y = (1.0 / (np.sqrt(2 * np.pi) * abs(std))) * np.exp(-u * u / 2.0)
    return y

def get_truncated_norm(mean, std, low, upp):
    return truncnorm((low - mean) / std, (upp - mean) / std, loc=mean, scale=sd)

def observation_to_index(obs):
    wavelength, distance = obs
    return np.ravel_multi_index(
        (wavelength - MIN_WAVELENGTH_OBS, distance - MIN_DISTANCE_OBS),
        (MAX_WAVELENGTH_OBS - MIN_WAVELENGTH_OBS, MAX_DISTANCE_OBS - MIN_DISTANCE_OBS)
    )

def index_to_observation(idx):
    wavelength, distance = np.unravel_index(
        idx,
        (MAX_WAVELENGTH_OBS - MIN_WAVELENGTH_OBS, MAX_DISTANCE_OBS - MIN_DISTANCE_OBS)
    )
    return wavelength + MIN_WAVELENGTH_OBS, distance + MIN_DISTANCE_OBS

def state_to_observation(state):
    pass
    position, speed, light = state
    return np.ravel_multi_index(
        (wavelength - , distance - MIN_DISTANCE_OBS),
        (MAX_WAVELENGTH_OBS - MIN_WAVELENGTH_OBS, MAX_DISTANCE_OBS - MIN_DISTANCE_OBS)
    )

def observation_to_state(idx):
    pass
