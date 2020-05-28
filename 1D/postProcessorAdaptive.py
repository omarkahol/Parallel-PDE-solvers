import numpy as np
from math import *
import matplotlib.pyplot as plt
from numpy import genfromtxt
from matplotlib.animation import FuncAnimation

xData, yData = [], []

with open("output.csv", "r") as f:
    for line in f:
        lsx, lsy = [], []
        for xy in line.split(";"):
            if len(xy.split(','))==2:
                x, y = xy.split(",")
                lsx.append(float(x))
                lsy.append(float(y))
        combined = list(zip(lsx, lsy))
        combined.sort(key=lambda x: x[0])
        xData.append([el[0] for el in combined])
        yData.append([el[1] for el in combined])

fig = plt.figure()
ax = fig.add_subplot(111,xlim=(0,1),ylim=(-1,1))
ax.set_title('Ut (x,t) +C*Ux (x,t) = k*Uxx (x,t)')
ax.set_xlabel('x');
ax.set_ylabel('u(x,t)');
line, = ax.plot([],[],'ko-',lw=2)

def animate(i):
    line.set_data(xData[i],yData[i])
    return line,

ani = FuncAnimation(fig, animate, frames=len(xData),interval=1, blit=True);
plt.show(ani)