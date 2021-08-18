# We simulate the spatial Lambda-Fleming-Viot model

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from IPython import embed
from os import path
import sys

# For the implicit solver we use Newton-Krylov
from scipy import optimize as opt

# GOY with explicit euler-maruyama
# The explicit solver is not very stable (with N=4 one immediately sees that orbits don't close,
# while with the implicit solver they approximately do)

# Better keep the explicit one here though, in case it's faster

class goy_explicit:

	"""
	We simulate the GOY model
	"""
	def __init__(self, initial_state, system_size, dt):

		# Initial values
		self.state  = initial_state
		self.size  = system_size
		self.dt    = dt

		self.noise = np.zeros( shape = self.size, dtype = complex )

		# Needed to define the drift
		self.coefficients = np.zeros( shape = self.size )
		for i in range(self.size):
			self.coefficients [i] = 2**(float(i))

		# And a shifted version of the same coefficients
		self.coefficients_up = np.zeros( shape = self.size )
		self.coefficients_up[:-1] = self.coefficients[1:]
		self.coefficients_up[-1]  = 1.0

		# Left shift (think of a sequence (a_0, a_1, a_1, .... a_N))
		# sent to (a_1, a_2, .... a_N, 0)
		self.shift_up = np.zeros( shape = self.size, dtype = complex)
		self.shift_up[:-1] =self.state[1:]
		self.shift_up[-1] = 0.0

		# and to (a_2, a_3, .... a_N, 0, 0)
		self.shift_dup = np.zeros( shape = self.size, dtype = complex)
		self.shift_dup[:-2] =self.state[2:]
		self.shift_dup[-1] = 0.0
		self.shift_dup[-2] = 0.0

		# Right shift
		# sent to (0, a_0, a_1, a_1, .... a_N-1)
		self.shift_dn = np.zeros( shape = self.size, dtype = complex)
		self.shift_dn[1:] = self.state[:-1]
		self.shift_dn[0] = 0.0

		# Other support variables
		self.drift_1 = np.zeros( shape = self.size, dtype = complex )
		self.drift_2 = np.zeros( shape = self.size, dtype = complex )


	def forward(self):

		# Forward scheme with (explicit) Euler-Maruyama
		self.state = self.state + self.dt*self.drift() 
		self.state[0] = 0.0

		self.reset_shift()


	def drift(self):

		# Defines the drift (interaction) in the system

		self.drift_1 = np.multiply(self.coefficients_up, np.multiply(self.shift_dup, self.shift_up))
		self.drift_1 = np.conj(self.drift_1)

		self.drift_2 = np.multiply(self.coefficients, np.multiply(self.shift_up, self.shift_dn))
		self.drift_2 = np.conj(self.drift_2)

		return 1.0j*(self.drift_1 - self.drift_2)

	def sample_noise(self):

		self.noise = self.noise

	def reset_shift(self):

		# We adjourn the shifted vectors accordin to the new state

		self.shift_up[:-1] =self.state[1:]
		self.shift_up[-1] = 0

		self.shift_dup[:-2] =self.state[2:]
		self.shift_dup[-1] = 0
		self.shift_dup[-2] = 0

		self.shift_dn[1:] = self.state[:-1]
		self.shift_dn[0] = 0



# GOY with implicit euler-maruyama

class goy:
	"""
	We simulate the GOY model
	"""
	def __init__(self, initial_state, system_size, dt):

		# Initial values
		self.state = initial_state
		self.size  = system_size
		self.dt    = dt

		self.noise = np.zeros( shape = self.size, dtype = complex )

		# Needed to define the drift
		self.coefficients = np.zeros( shape = self.size )
		for i in range(self.size):
			self.coefficients [i] = float(i)**2

		# And a shifted version of the same coefficients
		self.coefficients_up = np.zeros( shape = self.size )
		self.coefficients_up[:-1] = self.coefficients[1:]
		self.coefficients_up[-1]  = 1.0

		# Other support variables
		self.drift_1 = np.zeros( shape = self.size, dtype = complex )
		self.drift_2 = np.zeros( shape = self.size, dtype = complex )
		self.drift_tot   = np.zeros( shape = self.size, dtype = complex )

		# For the implicit Euler solver
		self.implicit = np.zeros( shape = self.size, dtype = complex )
		self.shifted_0  = np.zeros( shape = self.size, dtype = complex )
		self.shifted_1  = np.zeros( shape = self.size, dtype = complex )
		self.shifted_2  = np.zeros( shape = self.size, dtype = complex )

	def implicit_forward(self):

		self.state= opt.newton_krylov(self.implicit_f, self.state, rdiff=0.001)

	def implicit_f(self, input_v):

		# This is the function we feed into the implicit solver
		self.implicit = input_v - self.dt*self.drift(input_v) - self.state

		return self.implicit

	def drift(self, input_v):

		# This function computes the drift

		self.drift_1  = np.multiply(self.coefficients_up, np.multiply(self.shift(1, input_v), self.shift(0, input_v)))
		self.drift_1  = np.conj(self.drift_1)

		self.drift_2  = np.multiply(self.coefficients, np.multiply(self.shift(0, input_v), self.shift(2,input_v)))
		self.drift_2  = np.conj(self.drift_2)

		self.drift_tot    = 1.0j*(self.drift_1 - self.drift_2)

		return self.drift_tot


	def shift(self, param, input_v):

		# Shift 1 left
		if param == 0:

			self.shifted_0[:-1] = input_v[1:]
			self.shifted_0[-1]  = 0.0

			return self.shifted_0
			
		# Shift 2 left
		if param == 1:

			self.shifted_1[:-2] = input_v[2:]
			self.shifted_1[-1]  = 0.0
			self.shifted_1[-2]  = 0.0

			return self.shifted_1

		# Shift 1 right
		if param == 2:

			self.shifted_2[1:] = input_v[:-1]
			self.shifted_2[0]  = 0.0

			return self.shifted_2

class energy(goy):

	""" We enhance the GOY class, by computing all sorts of energies """

	def __init__(self,  initial_state, system_size, dt):
		super(energy, self).__init__(initial_state, system_size, dt)

		# Total L^2 energy
		self.absolute = np.absolute(self.state)
		self.total_energy = np.linalg.norm(self.absolute)

		# For the partial energies
		self.partial_energies = np.zeros(shape = self.size)

	# Adjourns the total L2 energy
	def compute(self):

		self.absolute = np.absolute(self.state)
		self.total_energy = np.linalg.norm(self.absolute)

		return self.total_energy

	# Outputs a vector. At entry i we find the L2 energy up to level i.
	def partial(self):

		for i in range(self.size):

			self.partial_energies[i] = np.linalg.norm(np.absolute(self.state[:i]))

		return self.partial_energies


# Parameters of the model
# Dimension of the system
NN = 10
# Initial state of the system
initial_state = np.zeros(shape = NN, dtype=complex)
for i in range(NN):
	initial_state[i] = 5.0/(5.0+0.1*float(i)**2)
# Time increment
dt = 0.001
# Time horizon
time_horizon = 10000

# We define the model with these parameters
my_goy  = energy(initial_state, NN, dt)

eng     = np.zeros( shape = time_horizon )
time    = np.arange(time_horizon) 

# We define the plot
fig, (ax1, ax2) = plt.subplots(2)
count = 0

# We let the system run
for i in range(time_horizon):

	my_goy.implicit_forward()
	eng[i]     = my_goy.compute()

	# Every now and then we plot the energy profile
	if (dt*i>0.05*count):
		ax2.plot(np.arange(NN), my_goy.partial())
		count+=1

ax1.plot(time, eng)
plt.show()


# embed()

# fig,ax = plt.subplots()

# for i in range(NN):

# 	ax.scatter(trajectories[:,i].real, trajectories[:,i].imag)
# 	ax.scatter(trajectories_explicit[:,i].real, trajectories_explicit[:,i].imag)

# plt.show()


# trajectories = np.zeros(shape = (time_horizon, NN), dtype = complex)

# plt.clf()

# plt.plot(time, eng)
# plt.show()







