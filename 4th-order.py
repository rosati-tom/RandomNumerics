


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from IPython import embed
from os import path
import sys

class eqn:
	"""
	We simulate d_t u - D^2 u = 0.
	"""
	def __init__(self, x_0, x_1):
		# Initial state of the first system.
		self.state_a = x_0

		# Initial state of the second system.
		self.state_b = x_1

		# We initialize the spatial noise variable.
		self.noise =  0*np.random.normal(size = (space_pts) , scale = np.sqrt(1/delta_x)) + 1
		for i in range(0,int(space_pts/4)):
			self.noise[i]=0.5*(delta_x*i - delta_x*space_pts/4)
		for i in range(int(space_pts/4), space_pts):
			self.noise[i]=(delta_x*i - delta_x*space_pts/4)

	def do_step(self):

		# We do one more step in the implicit Euler approximations
		self.state_a = np.dot(resolvent, self.state_a - (np.multiply(np.multiply( \
			self.state_a, self.state_a-1), self.noise))*delta_t )

		self.state_b = np.dot(resolvent, self.state_b - (np.multiply(np.multiply( \
			self.state_b, self.state_b-1), self.noise))*delta_t )
		

def animate(i):
	# Real time is:
	ani_time = i*delta_t

	# Redefine the plot
	lines_a.set_data(space, sample.state_a)
	lines_b.set_data(space, sample.state_b)

	# Set the new time
	time_text.set_text("Time = {:2.3f}".format(ani_time) )
	# We print the step we are in:
	sys.stdout.flush()
	sys.stdout.write("\r Step = {}, Value = {}, Derivative = {}".format(i, sample.state_a[middle], (delta_x**(-2))*(sample.state_a[middle-1]+sample.state_a[middle+1]-2*sample.state_a[middle])))
	#sys.stdout.write("\r Step = {}, Value = {}, Derivative = {}".format(i, 1,2))
	# And we do the next step:
	sample.do_step()
	return [lines_a,] + [lines_b,] + [time_text,]

# Space-Time discretisation
delta_t = 1/20*1.0
delta_x = 1/50*1.0
delta_tx = (delta_t)/(delta_x**4)

# Box size:
L = 10

# Space discretisation
space = np.arange(0.0, 2*np.pi*L + 0.001, delta_x)
space_pts = len(space)
middle = int(space_pts/4)-1

# We create a sample path
# with initial condition x_0, x_1:
x_0 = 0.5+0.0*np.abs(0.5*np.sin(space))
x_1 = np.abs(0.5*np.cos(space))

sample = eqn(x_0, x_1)



# This is the resolvent of the squqre Laplacian matrix
# Each row of the square laplacian looks like [1/4, -1, 3/2, -1, 1/4]
# It is the laplacian with Dirichlet boundary, and we normalize
# the matrix (1+Delta^2) to have 1 on the diagonal.

#off_value_1 = (2.0/3.0)*(1/(1+(3.0/2.0)*(delta_tx)) -1)
#off_value_2 = (-1.0/4.0)*(2.0/3.0)*(1/(1+(3.0/2.0)*(delta_tx)) -1)

off_value_1 = -1.0*delta_tx/(1+(3.0/2.0)*(delta_tx))
off_value_2 = (1.0/4.0)*delta_tx/(1+(3.0/2.0)*(delta_tx))
main_diag   = np.ones(shape = (space_pts))
offu_diag_1 = off_value_1*np.ones(shape = (space_pts-1))
offu_diag_2 = off_value_2*np.ones(shape = (space_pts-2))
to_invert   = scipy.sparse.diags([offu_diag_2, offu_diag_1, main_diag, offu_diag_1, offu_diag_2], [-2, -1, 0, 1, 2]).toarray()

#We then invert the matrix to get the resolvent.
resolvent   = scipy.linalg.inv(to_invert)/(1+(6.0/4.0)*(delta_tx))


#We set up the picture
fig       = plt.figure()
ax        = plt.axes(xlim=(0, 2*np.pi*L), ylim = (-0.2, 1.2))
time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
lines_a,  = ax.plot([],[], lw = 2)
lines_b,  = ax.plot([],[], lw = 2)
plt.title("4th Order Equation")

embed()

# We let the animation go.
ani = FuncAnimation(fig, animate, frames= 100, repeat=False)
mywriter = animation.FFMpegWriter(fps=10, codec="libx264", bitrate=60000, extra_args=['-pix_fmt', 'yuv420p'])
ani.save('4th-order.mp4',writer=mywriter)



