

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from IPython import embed
from os import path
import sys

class Tal_Model:
	"""
	We sample N random variables in a box
	and we let the process evolve.
	"""
	def __init__(self, N, box_size):
		# Initial values
		self.size     = N 
		self.box_size = box_size
		self.state    = 0.5*np.random.uniform(low = 0.0, high = self.box_size, size=self.size)

		self.next = 0

		# To compute the longtime average
		self.incr = np.zeros(shape = self.size)
		self.sorted = np.zeros(shape = self.size)
		self.average_incr = np.zeros(shape = self.size)

		# Keep track of time
		self.time = 1


	def do_step(self):

		self.next = np.random.uniform(low = 0.0, high = self.box_size)

		self.to_jump = np.argmin(np.abs(self.state-self.next))

		self.state[self.to_jump] = self.next

		# We compute the lontime quantities
		self.longtime()

		# We adjourn the time
		self.time +=1

	def compute(self):

		self.sorted = np.sort(self.state)
		self.incr[0] = self.sorted[0]
		for i in np.arange(1,self.size):
			self.incr[i] = self.sorted[i] - self.sorted[i-1]

	def longtime(self):

		self.compute()

		self.average_incr = self.average_incr*self.time + self.incr
		self.average_incr = self.average_incr/(self.time+1.0)

# Parameters of the domain
NN = 10
tryals = 600000
box_size = 1.0
model = Tal_Model(NN, box_size)

# Uniformly distributed random variables
my_uni = np.random.uniform(low = 0.0, high = box_size , size=(NN,tryals))
my_uni_tamp = np.sort(my_uni, axis = 0)
my_uni_incr = np.zeros(shape = (NN, tryals))
my_uni_incr[0,:] = my_uni_tamp[0,:]

for i in np.arange(1,NN):
	my_uni_incr[i,:] = my_uni_tamp[i,:] - my_uni_tamp[i-1,:]

my_avrg = np.average(my_uni_incr, axis = 1)

# We compute the density of the INCRNUM increment.


MAXNUM = 800
samples_model = np.zeros(shape = MAXNUM)
samples_unifo = np.zeros(shape = MAXNUM)

fig, (ax1, ax2) = plt.subplots(2)

for INCRNUM in range(0,NN-1):

	for x in range(0,MAXNUM):
		
		model.state = np.random.uniform(low = 0.0, high = box_size , size=NN)
		for y in range(0,5000):
			model.do_step()

		samples_model[x] = model.incr[INCRNUM]

		my_uni = np.random.uniform(low = 0.0, high = box_size , size=(NN))
		my_uni_tamp = np.sort(my_uni)
		my_uni_incr = np.zeros(shape = (NN))
		my_uni_incr[0] = my_uni_tamp[0]

		for i in np.arange(1,NN):
			my_uni_incr[i] = my_uni_tamp[i] - my_uni_tamp[i-1]

		samples_unifo[x] = my_uni_incr[INCRNUM]

		sys.stdout.flush()
		sys.stdout.write("\r Step = {}, {}".format(INCRNUM, x))



	ax1.hist(samples_model, bins = 60)
	ax2.hist(samples_unifo, bins = 60)	
	fig.savefig('10-particle-{}.png'.format(INCRNUM))	

	ax1.cla()
	ax2.cla()	
# for x in range(1,50000):
	
# 	model.do_step()
	
# 	# print(my_avrg-model.average_incr)

# 	sys.stdout.flush()
# 	sys.stdout.write("\r Step = {},, Diff = {}".format(x, np.linalg.norm(my_avrg-model.average_incr)))

# 	sys.stdout.write("\r Step = {}".format(i))

# #We set up the picture
# fig       = plt.figure()
# ax        = plt.axes(xlim=(0, space_horizon-5.5), ylim = (-0.02, 1.02))
# time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
# my_lines,  = ax.plot([],[], lw = 2)
# plt.title("The neutral SLFV process")

# def init():
#     my_lines.set_data([], [])
#     return my_lines,

# # We let the animation go.
# delta_t=0.01
# ani = FuncAnimation(fig, animate, init_func=init, frames=2000, interval=10, blit=True)
# ani.save('slfv.gif', writer='imagemagick')

# This does the animation

# def animate(i):
# 	# Real time is:

# 	ani_time = delta_t*i

# 	# Redefine the plot
# 	my_lines.set_data(space_points, my_slfv.state)

# 	# Set the new time
# 	time_text.set_text("Time = {:2.3f}".format(ani_time) )
# 	# We print the step we are in:
# 	sys.stdout.flush()
# 	#sys.stdout.write("\r Step = {}, Value = {}, Derivative = {}".format(i, fkpp_sample.state_a[middle], (delta_x**(-2))*(fkpp_sample.state_a[middle-1]+fkpp_sample.state_a[middle+1]-2*fkpp_sample.state_a[middle])))
# 	sys.stdout.write("\r Step = {}".format(i))
# 	# And we do the next step:
# 	my_slfv.go_to_time(ani_time)
# 	return [my_lines,] + [time_text,]

# We define the parameters of the process