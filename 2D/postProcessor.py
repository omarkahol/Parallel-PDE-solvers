import numpy as np
from math import *
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

fileName = "iterations.csv"

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
ax = fig.add_subplot(111,xlim=(0,1),ylim=(0,1), projection='3d')


def animate(i):
  ax.clear()
  cc = ax.plot_surface(X,Y,dataMatrix[i,:,:],alpha=1, cmap=cm.viridis)
  ax.set_title('Ut + C1*Ux + C2*Uy = K*(Uxx + Uyy)')
  ax.set_xlabel('x');
  ax.set_ylabel('y');
  ax.set_zlabel('U(x,y)')
  ax.set_zlim((-1,1))
  return cc

ani = FuncAnimation(fig,animate, frames=nT, interval=1, blit=False)
plt.show(ani)