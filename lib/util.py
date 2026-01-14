import numpy as np
import librosa

def amp2db(S):
    return librosa.amplitude_to_db(np.abs(S), amin=1e-15, top_db=500)

def pow2db(S):
    return librosa.power_to_db(np.abs(S), amin=1e-30, top_db=500)

def db2amp(d):
    return 10**(np.array(d)/20)

def binary_search(f, x_lim, y):
    """
    Finds the greatest number x for which f(x) <= y (f monotonic),
    bounded by x_lim = (x_lo, x_hi).

    Not the most robust implementation, especially if x_lim is
    something other than (1, 2).
    """
    delta = (x_lim[1] - x_lim[0]) / 2
    x = x_lim[0]
    while delta > 1e-20:
        if f(x + delta) <= y: x += delta
        delta /= 2
    return x
