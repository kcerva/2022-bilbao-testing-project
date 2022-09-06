import numpy as np

def get_logistic_map_step(x, r=1):
    y = r*x*(1-x)
    return y

def iterate_f(it, x, r):
    map = []
    for step in range(it):
        x = get_logistic_map_step(x, r)
        map.append(x)
    return np.array(map, dtype=np.float64)
