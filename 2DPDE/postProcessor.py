import numpy as np
from math import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

fileName = "resultsFile.csv"

array3D = []
with open(fileName,'r') as f:
  for line in f:
    array2D = []
    for xLine in line.split(';'):
      array1D=[]
      for value in xLine.split(','):
        array1D.append(float(value))
      array2D.append(array1D)
    array3D.append(array2D)

dataMatrix = np.array(array3D)
nT, nY, nX = dataMatrix.shape

xAxis = np.linspace(0,1,nX)
yAxis = np.linspace(0,1,nY)
X, Y = np.meshgrid(xAxis, yAxis)

fig = plt.figure()
ax = fig.add_subplot(111,xlim=(0,1),ylim=(0,1), aspect='equal')
ax.set_title('Ut(x,y,t) +C*grad(U)(x,y,t) = k*lap(U)(x,y,t)')
ax.set_xlabel('x');
ax.set_ylabel('y');

def animate(i):
  ax.clear()
  cc = ax.contourf(X,Y,dataMatrix[i,:,:],alpha=0.5, cmap=cm.viridis)
  return cc

ani = FuncAnimation(fig,animate, frames=nT, interval=1, blit=False)
plt.show(ani)

