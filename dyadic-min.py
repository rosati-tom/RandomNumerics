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

# GOY model with constrained optimization, hoping this forces the dynamics to preserve the energy.
# The implicit solver looses energy,
# while the explicit one increases energy.

# We need to use scipy minimization with either COBYLA or SLSQP

class goy:
	"""
	We simulate the GOY model
	"""
	def __init__(self, initial_state, system_size, dt, sigma):

		# Initial values
		self.state = initial_state
		self.size  = system_size
		self.dt    = dt

		# Needed to define the drift
		self.coefficients = np.zeros( shape = self.size, dtype = complex )
		for i in range(self.size):
			self.coefficients [i] = float(i)**2

		# And a shifted version of the same coefficients
		self.coefficients_up = np.zeros( shape = self.size, dtype = complex)
		self.coefficients_up[:-1] = self.coefficients[1:]
		self.coefficients_up[-1]  = 1.0

		# This adds the noise
		self.sigma      = sigma
		self.noise      = np.zeros( shape = self.size, dtype = complex )
		self.noise_real = np.zeros( shape = (2,self.size), dtype = float)
		self.sample_noise()


		# Other support variables
		self.drift_1    = np.zeros( shape = self.size, dtype = complex )
		self.drift_2    = np.zeros( shape = self.size, dtype = complex )
		self.drift_tot  = np.zeros( shape = self.size, dtype = complex )
		self.cur_drift  = np.zeros( shape = self.size, dtype = complex )

		# For the minimization solver
		self.implicit   = np.zeros( shape = self.size, dtype = complex )
		self.shifted_0  = np.zeros( shape = self.size, dtype = complex )
		self.shifted_1  = np.zeros( shape = self.size, dtype = complex )
		self.shifted_2  = np.zeros( shape = self.size, dtype = complex )

		# For the energy (which we use as a constraint)
		self.total_energy_initial = np.zeros(shape = 1, dtype = complex)
		self.total_energy_initial = np.linalg.norm(np.abs(self.state))
		self.total_energy_min     = np.zeros(shape = 1, dtype = complex)

		# The constraint
		self.epsilon      = 0.0001
		self.constraint   = opt.NonlinearConstraint(self.energy_diff, -self.epsilon, self.epsilon)

		# To pass between reals and complex numbers
		self.state_wrapped      = np.zeros(shape = (2*self.size), dtype = float)
		self.state_wrapped_supp = np.zeros(shape = (2, self.size), dtype = float)
	
	def implicit_forward(self):

		self.complexify()

		# In the case with noise we have to adjourn the constraints,
		# so that the noise can change the L2 norm.
		self.total_energy_initial = np.sqrt(np.dot(self.state, np.conj(self.state)).real+2*np.dot(self.state,np.conj(self.noise)).real)

		# We also adjourn the drift:
		self.cur_drift = self.drift(self.state)

		# This gives back complex numbers as 2-dim real vectors (for this reason we wrap it in the complexification)
		self.state_wrapped = opt.minimize(self.implicit_f, self.state_wrapped, method='SLSQP', constraints = self.constraint).x

		self.decomplexify()

		# We sample the new noise
		self.sample_noise()
		
	def implicit_f(self, input_v):	

		self.implicit_complex_v = input_v[:self.size]+1.0j*input_v[self.size:]

		# This is the function we feed into the implicit solver
		# What follows is half implicit half explicit
		self.implicit = self.implicit_complex_v - 0.997*self.dt*self.drift(self.implicit_complex_v) -0.003*self.dt*self.cur_drift - self.noise - self.state 

		# This is the fully implicit minimizer
		# self.implicit = self.implicit_complex_v - self.dt*self.drift(self.implicit_complex_v) - self.noise - self.state 

		return np.linalg.norm(np.abs(self.implicit))

	def drift(self, input_v):

		# This function computes the drift

		self.drift_1  = np.multiply(self.coefficients_up, np.multiply(self.shift(1, input_v), self.shift(0, input_v)))
		self.drift_1  = np.conj(self.drift_1)

		self.drift_2  = np.multiply(self.coefficients, np.multiply(self.shift(0, input_v), self.shift(2,input_v)))
		self.drift_2  = np.conj(self.drift_2)

		self.drift_tot= 1.0j*(self.drift_1 - self.drift_2)

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

	def energy_diff(self, input_v):

		self.implicit_complex_v = input_v[:self.size]+1.0j*input_v[self.size:]

		# Adjourns the total energy
		self.total_energy_min = np.linalg.norm(np.absolute(self.implicit_complex_v))

		return np.abs(self.total_energy_min - self.total_energy_initial)

	def sample_noise(self):

		self.noise_real = np.random.normal( loc = 0.0, scale = np.sqrt(self.dt*0.5*self.sigma), size =(2, self.size))
		self.noise = self.noise_real[0,:] + 1.0j*self.noise_real[1,:] 

	def complexify(self):

		self.state_wrapped_supp  = np.array([self.state.real, self.state.imag])
		self.state_wrapped[:self.size] = self.state_wrapped_supp[0,:]
		self.state_wrapped[self.size:] = self.state_wrapped_supp[1,:]

	def decomplexify(self):

		self.state = self.state_wrapped[:self.size] + 1.0j*self.state_wrapped[self.size:]
		

class energy(goy):

	""" We enhance the GOY class, by computing energy levels """

	def __init__(self,  initial_state, system_size, dt, sigma):
		super(energy, self).__init__(initial_state, system_size, dt, sigma)

		# For the partial energies
		self.total_energy     = np.linalg.norm(np.absolute(self.state))
		self.partial_energies = np.zeros(shape = self.size)

	def compute(self):

		# Adjourns the total energy
		self.total_energy = np.linalg.norm(np.absolute(self.state))

		return self.total_energy

	# Outputs a vector. At entry i we find the L2 energy up to level i.
	def partial(self):

		for i in range(self.size):

			self.partial_energies[i] = np.linalg.norm(np.absolute(self.state[:i]))

		return self.partial_energies


# Parameters of the model
# Dimension of the system
NN = 100

# Initial state of the system
initial_state = np.ones(shape = NN, dtype=complex)

# In this case chosen at random
initial_noise = np.random.normal(loc = 0.0, scale = 1.0, size =(2,NN))
initial_state = initial_noise[0,:] + 1.0j*initial_noise[1,:]
for i in range(NN):
	initial_state[i] = 5.0*np.sin(float(i))*initial_state[i]/(5.0+0.005*np.sqrt(float(i)))

# Below is a deterministic initial condition
#for i in range(NN):
#	initial_state[i] = 5.0/(5.0+0.005*np.sqrt(float(i))**2)


# Time increment
dt = 0.005
# Time horizon
time_horizon = 500

# We define the model with these parameters
my_goy  = energy(initial_state, NN, dt, sigma = 5.0)
goy_det = energy(initial_state, NN, dt, sigma = 0.0)
# goy_imp = goy_impl(initial_state, NN, dt)

eng     = np.zeros( shape = time_horizon ) + my_goy.compute()
eng_det = np.zeros( shape = time_horizon ) + goy_det.compute()
time    = np.arange(time_horizon) 
space   = np.arange(NN)

# trajectories = np.zeros(shape = (time_horizon, NN), dtype = complex)
# trajectories_implicit = np.zeros(shape = (time_horizon, NN), dtype = complex)

# We set up the picture
fig, (ax1, ax2) = plt.subplots(2)
fig.tight_layout()
time_text = ax1.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax1.transAxes)
plt.title("The energy profile of the GOY model")
ax1.set_ylim([my_goy.compute() - 3.0, my_goy.compute() + 3.0])

# Initialization
lines_1, = ax1.plot(time, eng)
lines_3, = ax1.plot(time, eng_det)
lines_2, = ax2.plot(np.arange(NN), my_goy.partial())
lines_4, = ax2.plot(np.arange(NN), goy_det.partial())

def init():

    return [lines_1,] + [lines_3,] + [lines_2,] + [lines_4,]

# To build the animation
def animate(i):
	# Real time is:

	ani_time   = dt*i
	# Redefine the plot
	eng[i]     = my_goy.compute()
	eng_det[i] = goy_det.compute()

	lines_1.set_data(time, eng)
	lines_3.set_data(time, eng_det)
	lines_2.set_data(space, my_goy.partial())
	lines_4.set_data(space, goy_det.partial())

	# Set the new time
	time_text.set_text("Time = {:2.3f}".format(ani_time) )
	
	# We print the step we are in:
	sys.stdout.flush()
	#sys.stdout.write("\r Step = {}, Value = {}, Derivative = {}".format(i, fkpp_sample.state_a[middle], (delta_x**(-2))*(fkpp_sample.state_a[middle-1]+fkpp_sample.state_a[middle+1]-2*fkpp_sample.state_a[middle])))
	sys.stdout.write("\r Step = {}".format(i))
	
	# And we do the next step:
	my_goy.implicit_forward()
	goy_det.implicit_forward()

	return [lines_1,] + [lines_3,] + [lines_2,] + [lines_4,] + [time_text,]

# To run and save the animation
ani = FuncAnimation(fig, animate, init_func=init, frames=time_horizon-10, interval = 10, blit = True)
ani.save('goy_noisy.gif', writer = 'imagemagick')


# We let the animation go.
# ani = FuncAnimation(fig, animate, init_func = init, frames=100)
# ani = FuncAnimation(fig, animate, frames=100, interval=10, blit=True)
# ani.save('goy.gif', writer='imagemagick')


# # Parameters for the plot
# count = 0

# # We let the system run
# for i in range(time_horizon):

# 	my_goy.implicit_forward()
# 	# goy_imp.implicit_forward()
# 	eng[i]  = my_goy.compute()

# 	# Every now and then we plot the energy profile
# 	if (dt*i>0.01*count):
# 		ax2.plot(np.arange(NN), my_goy.partial())
# 		count+=1

# 	trajectories[i,:] = my_goy.state
# 	# trajectories_implicit[i,:] = goy_imp.state

# 	# print("\n\n")
# 	# print(trajectories[i,:])
# 	# print(trajectories_implicit[i,:])
# 	# print("\n\n")

# ax1.plot(time, eng)
# plt.show()




# fig,ax = plt.subplots()

# for i in range(NN):

# 	ax.scatter(trajectories[:,i].real, trajectories[:,i].imag, marker = 'o', lw =0.001)
# 	ax.scatter(trajectories_implicit[:,i].real, trajectories_implicit[:,i].imag, marker ='v', lw = 0.001)

# plt.show()




# plt.clf()

# plt.plot(time, eng)
# plt.show()







