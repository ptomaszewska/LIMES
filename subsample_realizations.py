import sys
import numpy as np
import glob

"""
Script to create 10 realizations of dataset.
It takes as a commandline argument, directory to files generated using embedding.py script.
"""
filenames = glob.glob(sys.argv[1] +"*")
for filename in filenames:
    X = np.load(filename)
    for i in range(10):
        np.save(filename.replace('.npy','')+'-sub{}.npy'.format(i),X[i::10])