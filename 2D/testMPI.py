import numpy as np
from math import *
import os

commandStr = 'mpic++ mpi2DPDE.cpp -O3 -o out'
os.system(commandStr)

for nThreads in range(1,11):
  os.system('mpirun -np {0} --oversubscribe ./out 0'.format(nThreads))