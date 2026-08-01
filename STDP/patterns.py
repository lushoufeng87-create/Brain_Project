
import numpy as np

"""
Sensory patterns for Brain Project.
Each function returns a list of neuron indices to stimulate.
"""

def block_pattern(start, width):
    return np.arange(start, start+width)

def make_patterns(N, input_fraction=0.1):
    width=max(2,int(N*input_fraction))
    return {
        "A": block_pattern(0,width),
        "B": block_pattern(N//4,width),
        "C": block_pattern(N//2,width),
        "D": block_pattern(3*N//4-width,width),
    }

def apply_pattern(I, pattern, amplitude=9.0, noise_std=1.0):
    I[pattern]+=amplitude+np.random.normal(0,noise_std,len(pattern))
    return I

def random_training_schedule(patterns, repeats=100):
    names=list(patterns.keys())
    schedule=[]
    for _ in range(repeats):
        np.random.shuffle(names)
        schedule.extend(names.copy())
    return schedule
