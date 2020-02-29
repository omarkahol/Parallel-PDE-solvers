import numpy as np
from math import *
import os

commandStr = 'g++ omp2DPDE.cpp -fopenmp -O3 -o out'
os.system(commandStr)

for nThreads in range(1,5):
  os.system('./out {0} 0'.format(nThreads))