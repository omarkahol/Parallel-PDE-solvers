import numpy as np 
import matplotlib.pyplot as plt 
from math import *

data = np.genfromtxt('error.csv')

coeff = np.polyfit(np.arange(data.size),data, 1)
fitted = np.polyval(coeff, np.arange(len(data)))

plt.plot(data, 'k-', lw=2, label = 'errore')
plt.plot(fitted, 'r--', lw=2, label='m = {:0.3e}'.format(coeff[1]))
plt.title('errore medio ad ogni iterazione')
plt.xlabel('iterazione')
plt.ylabel('errore')
plt.legend()
plt.show()


data = np.genfromtxt('mesh.csv')
plt.plot(data, 'k-', lw=2, label='mesh size')
plt.plot([0, len(data)], [25, 25], 'b--', lw=1, label='size iniziale')
plt.title('numero di punti della mesh')
plt.legend()
plt.xlabel('iterazione')
plt.ylabel('N')
plt.show()
