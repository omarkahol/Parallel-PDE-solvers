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

# #OPENMP
fig = plt.figure()
ax = fig.add_subplot(111)
commandStr = 'g++ openmpDiffusionConvectionAuto.cpp functions.cpp -fopenmp -o out'
os.system(commandStr)
nThreadsOMP = [1,2,4,8]
npointsbase = 100
nTime = 5000
elapsedTimeArrOMPauto = []

for n, drawstr in zip(nThreadsOMP,['k-','r-','b-','g-']):
  commandStr='./out {0} {1} 0.01 0 {2}'.format(n,npointsbase,nTime)
  os.system(commandStr)
  nx, ntt, dx, dt, t = readResults()
  dataMatrix = readArray()
  sArr, tArr = getTSarray(dx, dt, npointsbase, nTime)
  enArr = energy(nTime,dataMatrix, sArr)
  ax.plot(tArr, enArr, drawstr, lw = 2, label='processori: {0}'.format(n))

ax.legend()
ax.set_xlabel('Tempo [s]')
ax.set_ylabel('integrale(||u||^2,0,1)')
ax.set_title('Energia nel tempo, openMP')
plt.show()

#MPI
fig = plt.figure()
ax = fig.add_subplot(111)
commandStr = 'mpic++ mpiDiffusionConvection.cpp functions.cpp functionsMPI.cpp -o out'
os.system(commandStr)
nThreadsMPI = [2,4,8]
npointsbase = 100
nTime = 5000
elapsedTimeArrOMPauto = []

for n, drawstr in zip(nThreadsMPI,['k-','r-','b-']):
  npointsthread = int(npointsbase/n)
  commandStr='mpirun -np {0} --oversubscribe ./out {1} 0.01 0 {2}'.format(n, npointsthread,nTime);
  os.system(commandStr)
  nx, ntt, dx, dt, t = readResults()
  print(nx, ntt, dx, dt)
  dataMatrix = readArray()
  sArr, tArr = getTSarray(dx, dt, len(dataMatrix[0,:]), nTime)
  enArr = energy(nTime,dataMatrix, sArr)
  ax.plot(tArr, enArr, drawstr, lw = 2, label='processori: {0}'.format(n))

ax.legend()
ax.set_xlabel('Tempo [s]')
ax.set_ylabel('integrale(||u||^2,0,1)')
ax.set_title('Energia nel tempo, MPI')
plt.show()