import numpy as np
from matplotlib import pyplot as plt
import math


def f(x):
    return 8*x + 8*math.sin(x)

def deriv_f(x):
    return 8 + 8*math.cos(x)

def gradient_descent(start, gradient, step_length, limit):
    x = start
    points = [x]
    steps = 0
    while steps < limit:
        d = gradient(x)
        x = x - step_length * d
        points.append(x)
        steps += 1
    return points


