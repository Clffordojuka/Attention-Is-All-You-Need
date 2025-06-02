import json
import numpy as np

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def build_sine_wave(X: np.ndarray, frequency: float, amplitude: float) -> np.ndarray:
    """
    Builds a sine wave from a time array, frequency, and amplitude.
    """
    return amplitude * np.sin(X * frequency)


def modify_sine(sine_wave: np.ndarray, loc: int, length: int) -> np.ndarray:
    """
    Modifies a section of the sine wave by flattening it 
    between indices [loc, loc + length).
    """
    new_sine_wave = sine_wave.copy()
    end = min(loc + length, len(sine_wave))  # Prevent index out-of-bounds
    flat_value = sine_wave[loc]
    new_sine_wave[loc:end] = flat_value
    return new_sine_wave


def build_modified_sine_wave(X: np.ndarray, frequency: float, amplitude: float, loc: int, length: int) -> np.ndarray:
    """
    Generates a sine wave and modifies it in a localized region.
    Useful for simulating anomalies.
    """
    sine_wave = build_sine_wave(X, frequency, amplitude)
    return modify_sine(sine_wave, loc, length)
