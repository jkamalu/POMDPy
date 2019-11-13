import math
import os.path
import random

def pdf(mean, std, value):
    u = float(value - mean) / abs(std)
    y = (1.0 / (math.sqrt(2 * math.pi) * abs(std))) * math.exp(-u * u / 2.0)
    return y