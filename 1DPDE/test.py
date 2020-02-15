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
nThreadsOMP = [1,2,4,8]
npointsbase = 100000
elapsedTimeArrOMP = []
for n in nThreadsOMP:
  npointsthread = int(npointsbase/n)
  commandStr='./out {0} {1} 0.0001 0'.format(n,npointsthread)
  os.system(commandStr)
  nx, ntt, dx, dt, t = readResults()
  elapsedTimeArrOMP.append(t)


#MPI APPROACH
commandStr = 'mpic++ mpiDiffusionConvection2.cpp functions.cpp functionsMPI.cpp -O2 -o out'
os.system(commandStr)
nThreadsMPI = [2,4,8]
npointsbase = 100000
elapsedTimeArrMPI = []
for n in nThreadsMPI:
  npointsthread = int(npointsbase/n)
  commandStr='mpirun -np {0} --oversubscribe ./out {1} 0.0001 0'.format(n, npointsthread);
  os.system(commandStr)
  nx, ntt, dx, dt, t = readResults()
  elapsedTimeArrMPI.append(t)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(nThreadsMPI,elapsedTimeArrMPI,'ro-', label='MPI')
ax.plot(nThreadsOMP,elapsedTimeArrOMP,'bo-', label='OPENMP')
ax.legend()
ax.set_xlabel('Numero di processori')
ax.set_ylabel('Tempo impegato [miscrosecondi]')
ax.set_title('Tempo impiegato in funzione del numero di processori')
plt.show()







