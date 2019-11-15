import random
import nump as np

import json

config = json.load("config.json")

MIN_WAVELENGTH = min(config["color_means"] - 2 * config["color_stdev"])
MAX_WAVELENGTH = max(config["color_means"] + 2 * config["color_stdev"])

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

def observation_to_index(obs):
    assert len(obs) == 2
    wavelength, distance = obs
    return np.ravel_multi_index(
        (wavelength - MIN_WAVELENGTH, distance),
        (MAX_WAVELENGTH - MIN_WAVELENGTH, config["road_length"])
    )

def index_to_observation(idx):
    wavelength, distance = np.unravel_index(
        idx,
        (MAX_WAVELENGTH - MIN_WAVELENGTH, config["road_length"])
    )
    return wavelength + MIN_WAVELENGTH, distance
