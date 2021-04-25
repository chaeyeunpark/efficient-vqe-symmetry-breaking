import json
import sys
import os
from json import encoder
from pathlib import Path
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

N = 14
depth = 8
sigma = 0.001
learning_rate = float(sys.argv[1])
alpha = float(sys.argv[2])

param = {
    "N": N,
    "depth": depth,
    "sigma": sigma,
    "centering": true,
    "learning_rate": learning_rate,
    "grad_clip": 1e+8,
    "alpha": alpha,
}

print(json.dumps(param, indent=4))
