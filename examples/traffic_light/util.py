import math
import os.path
import random

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
    y = (1.0 / (math.sqrt(2 * math.pi) * abs(std))) * math.exp(-u * u / 2.0)
    return y

def get_truncated_norm(mean, std, low, upp):
    return truncnorm((low - mean) / std, (upp - mean) / std, loc=mean, scale=sd)

def calculate_trunc_norm_prob(value, mean, std, low, upp):
    upper = truncnorm.cdf(value + 0.5, (low - mean) / std, (upp - mean) / std, loc=mean, scale=sd)
    lower = truncnorm.cdf(value - 0.5, (low - mean) / std, (upp - mean) / std, loc=mean, scale=sd)
    return upper - lower

def state_to_color_index(state):
    color = -1
    light_range = 0
    while(state.light > light_range):
        color += 1
        light_range += self.config["light_cycle"][color]
    return color
