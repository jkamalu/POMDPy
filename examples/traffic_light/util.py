import math
import os.path
import random

class Acceleration(Enum):
    NEGATIVE_LRG = -8
    NEGATIVE_MED = -4
    NEGATIVE_SML = -2
    ZERO = 0
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
