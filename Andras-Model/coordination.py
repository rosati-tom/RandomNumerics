# We simulate the FKPP equation with coordination

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from IPython import embed
from os import path
import scipy.sparse
import scipy.sparse.linalg
import sys

class poissonpp:
	"""
	We sample a Poisson point process
	For the moment only in time (we take fixed jump sizes)
	"""
	# def __init__(self, lam, space_horizon, time_horizon):
	def __init__(self, time_horizon):
		# Initial values
		# self.lam = lam
		# self.space_horizon = space_horizon
		self.time_horizon = time_horizon

	def sample(self):

		# Creates a point_num x 2 array, with first column ordered times and second column associated space position

		# self.point_num = np.random.poisson(self.lam*self.space_horizon*self.time_horizon,size = None)
		self.point_num = np.random.poisson(self.time_horizon, size = None)

		self.points = np.random.rand(self.point_num)
		self.points = self.points*self.time_horizon
		# self.points = self.points*[self.time_horizon, self.space_horizon]
		# self.points = my_sort(self.points)

		self.points.sort()
		

# def my_sort(my_matrix):

# 	# Automatically sorts along the first column (in our case according to times)
# 	indexes = np.argsort(my_matrix, axis = 0)[:,0]

# 	return my_matrix[indexes, :]

class neumann:
	"""
	We create a class to solve 1D PDEs with Neumann boundary conditions.
	This is not so trivial. We use the symmetrization of the matrix presented
	in the notes attached.
	"""

	def __init__(self, d_t, d_x, mesh_size):


		# Key parameter for the implicit scheme
		self.alpha = d_t/d_x**2

		self.mesh_size = mesh_size

		# Normalization parameter for the resolvent 
		# ---> Id - d_t/d_x**2 \Delta 
		self.off_value = 0.5*(1/(1+2*self.alpha) -1)

		# The main diagonal
		self.main_diag = np.ones(shape = (mesh_size))

		# Off the main diagonal
		self.offu_diag =self.off_value*np.ones(shape = (mesh_size-1))

		# Correction for the boundary terms
		# (note that this is not enough and we have to introduce a correction
		# also in the forcing)
		self.main_diag[0] = 0.5
		self.main_diag[mesh_size-1] =0.5
		
		self.to_invert = scipy.sparse.diags([self.offu_diag, self.main_diag, self.offu_diag], [-1, 0, 1]).toarray()
		#We then invert the matrix to get the resolvent.
		self.resolvent = scipy.linalg.inv(self.to_invert)/(1+2*self.alpha)


	def apply_rslv(self, force):

		# We rescale appropriately the forcing

		self.force_scaled = force
		self.force_scaled[0] = force[0]*0.5
		self.force_scaled[self.mesh_size-1] = force[self.mesh_size-1]*0.5

		return np.dot(self.resolvent, self.force_scaled)
		

class fkpp:
	"""
	We model the fkpp equation (1D) with coortination.
	"""
	def __init__(self, x_0, x_1, delta_t, delta_x, space_pts):

		# Initial state of the first system.
		self.state_a = x_0

		# Initial state of the second system.
		self.state_b = x_1

		# We count the time to the next switch
		self.count =0

		#for i in range(0,int(space_pts/4)):
		#	self.noise[i]=0.5*(delta_x*i - delta_x*space_pts/4)
		#for i in range(int(space_pts/4), space_pts):
		#	self.noise[i]=(delta_x*i - delta_x*space_pts/4)

		# We start the solver
		self.slv = neumann(delta_t, delta_x, space_pts)

		# We introduce the jump times for the FKPP equation
		self.jump_times = poissonpp( time_horizon = 100.0)
		self.jump_times.sample()
		self.jump_index = 0

	def do_step(self):

		# We do one more step in the implicit Euler approximations
		self.state_a = self.slv.apply_rslv( self.state_a - (np.multiply( self.state_a, self.state_a - 1))*delta_t )

		self.state_b = self.slv.apply_rslv(self.state_b - (np.multiply( self.state_b, self.state_b - 1))*delta_t )

		self.count += delta_t

		if self.count > self.jump_times.points[self.jump_index]:

			# At jump times we make a jump
			# Here the jump size parameter is set by hand.

			print('Jumped!')

			self.state_a = self.state_a + 1.0*np.multiply(self.state_a, self.state_a-1)

			self.state_b = self.state_b + 1.0*np.multiply(self.state_b, self.state_b-1)

			# I the we finished the jumps we adjourn the counters and resample the noise

			self.jump_index +=1

			if self.jump_index == self.jump_times.point_num-5:

				self.count = 0
				self.jump_index = 0

				self.jump_times.resample

# This does the animation
def animate(i):
	# Real time is:

	ani_time = delta_t*i

	# Redefine the plot
	my_lines.set_data(space_points, my_fkpp.state_a)

	# Set the new time
	time_text.set_text("Time = {:2.3f}".format(ani_time) )
	# We print the step we are in:
	sys.stdout.flush()
	#sys.stdout.write("\r Step = {}, Value = {}, Derivative = {}".format(i, fkpp_sample.state_a[middle], (delta_x**(-2))*(fkpp_sample.state_a[middle-1]+fkpp_sample.state_a[middle+1]-2*fkpp_sample.state_a[middle])))
	sys.stdout.write("\r Step = {}".format(i))
	# And we do the next step:
	my_fkpp.do_step()
	return [my_lines,] + [time_text,]

# We define the parameters of the SLFV

# Parameters of the domain
space_horizon = 150.0
dx = 0.05
space_points = np.arange(0,space_horizon,dx)
space_len = len(space_points)

delta_t = 0.04

# Initial condition
my_init = np.zeros(shape = space_len)
for x in range(space_len):
	if x > 0.5*space_len:
		my_init[x] = 1.0

# We initialize the SLFV process
my_fkpp = fkpp(my_init, my_init, delta_t, dx, space_len)

#We set up the picture
fig       = plt.figure()
ax        = plt.axes(xlim=(0, space_horizon-5.5), ylim = (-0.02, 1.02))
time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
my_lines,  = ax.plot([],[], lw = 2)
plt.title("Coordinated FKPP equation")

def init():
    my_lines.set_data([], [])
    return my_lines,

# We let the animation go.
ani = FuncAnimation(fig, animate, init_func=init, frames=5000, interval=10, blit=True)
ani.save('cfkpp.gif', writer='imagemagick')
# ani.save(filename = 'neutral_slfv.html')

# ani       = animation.FuncAnimation(fig, animate, frames=400, interval = 70, blit = True)

# ani.save(filename = 'neutral_slfv.html', extra_args=['-vcodec', 'libx264'], bitrate = 20000)


# INSTRUCTION FOR PUTTING VIDEO IN PRESENTATION.

# 1) RUN: ffmpeg -i <input> -vf scale="trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -profile:v high -pix_fmt yuv420p -g 25 -r 25 output.mp4
#	 on powershell. The result (output.mp4) is the video you will use.
# 2)  IN Latex, with package movie9 write:
#   \includemedia[
#  width=0.7\linewidth,
#  totalheight=0.7\linewidth,
#  activate=onclick,
#  %passcontext,  %show VPlayer's right-click menu
#  addresource=ballistic_out.mp4,
#  flashvars={
#    %important: same path as in `addresource'
#    source=ballistic_out.mp4
#  }
#]{\fbox{Click!}}{VPlayer.swf}


# # Definition of the Poisson PP
# ppp = poissonpp(lam, space_horizon,time_horizon)

# # To check that the Poisson point process works
# points = ppp.sample()
# plt.scatter(points[:,0], points[:,1])
# plt.show()








