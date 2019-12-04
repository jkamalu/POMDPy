import os
import random
import json

from enum import Enum

import numpy as np
from scipy.stats import truncnorm

path = os.path.join(*__name__.split('.')[:-1], "config.json")
with open(path, "rt") as fp:
    config = json.load(fp)

def min_wavelength():
    return min(config["color_means"])

def max_wavelength():
    return max(config["color_means"])

def max_distance():
    return config["road_length"]

def min_distance():
    return -config["intersection_length"]

MIN_WAVELENGTH = min_wavelength()
MAX_WAVELENGTH = max_wavelength()
MAX_DISTANCE = max_distance()
MIN_DISTANCE = min_distance()

MIN_WAVELENGTH_OBS = MIN_WAVELENGTH - 2 * config["color_stdev"]
MAX_WAVELENGTH_OBS = MAX_WAVELENGTH + 2 * config["color_stdev"]
MAX_DISTANCE_OBS = config["road_length"] + 2 * config["distance_stdev"]
MIN_DISTANCE_OBS = -config["intersection_length"] - 2 * config["distance_stdev"]

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

INDEX_TO_ACTION = {i:a.value for i, a in enumerate(Acceleration)}

class LightColor(Enum):
    GREEN = 0
    YELLOW = 1
    RED = 2

def pdf(mean, std, value):
    u = float(value - mean) / abs(std)
    y = (1.0 / (np.sqrt(2 * np.pi) * abs(std))) * np.exp(-u * u / 2.0)
    return y

def get_truncated_norm(mean, std, low, upp):
    return truncnorm((low - mean) / std, (upp - mean) / std, loc=mean, scale=std)

def calculate_trunc_norm_prob(value, mean, std, low, upp):
    upper = truncnorm.cdf(value + 0.5, (low - mean) / std, (upp - mean) / std, loc=mean, scale=std)
    lower = truncnorm.cdf(value - 0.5, (low - mean) / std, (upp - mean) / std, loc=mean, scale=std)
    return upper - lower

def state_to_color_index(state):
    light = 0
    for i, color in enumerate(config["light_cycle"]):
        light += config["light_cycle"][i]
        if state.light <= light:
            return i

def observation_to_index(obs):
    wavelength, distance, speed = obs
    return np.ravel_multi_index(
        (wavelength - MIN_WAVELENGTH_OBS, distance - MIN_DISTANCE_OBS, speed),
        (MAX_WAVELENGTH_OBS - MIN_WAVELENGTH_OBS + 1, MAX_DISTANCE_OBS - MIN_DISTANCE_OBS + 1, config["max_speed"] + 1)
    )

def index_to_observation(idx):
    wavelength, distance, speed = np.unravel_index(
        idx,
        (MAX_WAVELENGTH_OBS - MIN_WAVELENGTH_OBS + 1, MAX_DISTANCE_OBS - MIN_DISTANCE_OBS + 1, config["max_speed"] + 1)
    )
    return wavelength + MIN_WAVELENGTH_OBS, distance + MIN_DISTANCE_OBS, speed

# def state_to_index(state):
#     position, speed, light = state
#     return np.ravel_multi_index(
#         (position - MIN_DISTANCE, speed, light),
#         (MAX_DISTANCE - MIN_DISTANCE, config["max_speed"], sum(config["light_cycle"]))
#     )

# def index_to_state(idx):
#     position, speed, light = np.unravel_index(
#         idx,
#         (MAX_DISTANCE - MIN_DISTANCE, config["max_speed"], sum(config["light_cycle"]))
#     )
#     return position + MIN_DISTANCE, speed, light

def action_to_index(action):
    return ACTION_TO_INDEX[action]

def index_to_action(idx):
    return INDEX_TO_ACTION[idx]
