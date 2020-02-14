import numpy as np
from math import *
from scipy.integrate import simps
import matplotlib.pyplot as plt
from numpy import genfromtxt
from matplotlib.animation import FuncAnimation
import os

def readResults():
  returnArray = []
  with open("resultsFile.csv","r") as f:
    for line in f:
      returnArray.append(float(line.split(',')[1]))
  return returnArray

def readArray():
  return genfromtxt('dataFile.csv',delimiter=',')

def getTSarray(DX,DT, NX, NT):
  spaceArray=[]
  timeArray = []
  for i in range(NX):
    spaceArray.append(i*DX)
  for t in range(NT):
    timeArray.append(t*DT)
  return (spaceArray, timeArray)

def energy(NT, dataMatrix, xArray):
  energyArray = []
  for t in range(NT):
    energyArray.append(simps(dataMatrix[t,:]**2,xArray))
  return energyArray

#OPENMP APPROACH
commandStr = 'g++ openmpDiffusionConvection.cpp functions.cpp -fopenmp -o out'
os.system(commandStr)
nThreads = range(2,11)
npointsbase = 1000
elapsedTimeArrOMP = []
for n in nThreads:
  npointsthread = int(npointsbase/n)
  commandStr='./out {0} {1} 0.001 0'.format(n,npointsthread)
  os.system(commandStr)
  nx, ntt, dx, dt, t = readResults()
  elapsedTimeArrOMP.append(t)

#MPI APPROACH
commandStr = 'mpic++ mpiDiffusionConvection2.cpp functions.cpp functionsMPI.cpp -o out'
os.system(commandStr)
nThreads = range(2,11)
npointsbase = 1000
elapsedTimeArrMPI = []
for n in nThreads:
  npointsthread = int(npointsbase/n)
  commandStr='mpirun -np {0} --oversubscribe ./out {1} 0.001 0'.format(n, npointsthread);
  os.system(commandStr)
  nx, ntt, dx, dt, t = readResults()
  elapsedTimeArrMPI.append(t)
plt.plot(nThreads,elapsedTimeArrOMP,'k-')
plt.plot(nThreads,elapsedTimeArrMPI,'r-')
plt.show()







