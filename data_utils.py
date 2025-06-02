
import numpy as np

def build_sine_wave(X, frequency, amplitude):
    """Builds the sine wave given a time step,
    a frequency value and an amplitude value""" 
    return amplitude * np.sin(X*frequency)


def build_modified_sine_wave(X, frequency, amplitude, loc, length):
    return modify_sine(build_sine_wave(X,frequency, amplitude), loc, length)


def modify_sine(sine_wave, loc, length):
    """Modified the sine_wave from loc to 
    loc+length""" 
    new_sine_wave = sine_wave.copy()
    x_value = sine_wave[loc]
    new_sine_wave[loc:loc+length] = x_value
    return new_sine_wave