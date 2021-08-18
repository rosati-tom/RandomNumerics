# We plot the eigenvalues of random matrices

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from IPython import embed
from os import path
import sys


class random_matrix:

	def __init__(self, size):

		self.size = size

		self.matrix = np.zeros(shape =(self.size,self.size), dtype = complex)
		self.noise  = np.zeros(shape =(2,self.size, self.size), dtype = float)\

		self.eigenvalues = np.zeros(shape = self.size, dtype = complex)

	def sample(self):
		
		self.noise = np.random.normal(size= (2,self.size, self.size))

		self.matrix = (self.noise[0,:,:]+1.0j*self.noise[1,:,:])/np.sqrt(2.0)

	def eigen(self):

		# Gets the eigenvalues
		self.eigenvalues = np.linalg.eig(self.matrix)[0]

		return self.eigenvalues

NN = 30
my_mat = random_matrix(NN**2)

my_mat.sample()
ev = my_mat.eigen()

circle_1 = plt.Circle((0,0), NN, color = 'lightcoral', alpha = 0.5)
#circle_2 = plt.Circle((0,0), 2*NN+1*NN**(-1.0/3.0), color = 'lightcoral', alpha = 0.4)

fig, ax = plt.subplots()
ax.scatter(ev.real, ev.imag)
ax.set_ylim(-NN-3*NN**(-1.0/3.0), NN+3*NN**(-1.0/3.0))
ax.set_xlim(-NN-3*NN**(-1.0/3.0), NN+3*NN**(-1.0/3.0))
ax.add_patch(circle_1)
#ax.add_patch(circle_2)

plt.show()




