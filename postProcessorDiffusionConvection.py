import numpy as np
from math import *
import matplotlib.pyplot as plt
from numpy import genfromtxt
from matplotlib.animation import FuncAnimation

data = genfromtxt('dataFile.csv',delimiter=',')

fig = plt.figure()
ax = fig.add_subplot(111,xlim=(0,1),ylim=(-1,1))
ax.set_title('Ut (x,t) +C*Ux (x,t) = k*Uxx (x,t)')
ax.set_xlabel('x');
ax.set_ylabel('u(x,t)');
line, = ax.plot([],[],'k-',lw=2)
nPoints = len(data[0,:])
xAxis = np.linspace(0,1,nPoints);

def animate(i):
    line.set_data(xAxis,data[i,:])

ani = FuncAnimation(fig,animate, frames=len(data[:,0]),interval=1);
plt.show(ani)