import numpy as np
from math import *
import os

commandStr = 'g++ omp1DPDE.cpp -fopenmp -march=native -O3 -o out'
os.system(commandStr)

for nThreads in range(1,17):
  os.system('./out {0} 0'.format(nThreads))