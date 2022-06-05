

# Solve the fkpp equation for large times w/ slow noise
# We use Space-Time discretesation (1D Finite elements)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
from matplotlib import colors
from IPython import embed
from os import path
import sys

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
		self.offu_diag =self.off_value*np.ones(shape = (space_pts-1))

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
	We model the fkpp equation (1D).
	"""
	def __init__(self, x_0, x_1, tau, sigma, delta_t, delta_x, space_pts):

		# Initial state of the first system.
		self.state = x_0

		# We initialize the spatial noise variable.
		# This parameter sets the number of random variables we will use.
		self.noise_points = 4.0

		self.alpha = self.noise_points/space_pts

		self.noise_rv =  np.random.normal(size = int(self.noise_points), scale = sigma)

		self.noise = np.zeros(shape = space_pts)
		for i in range(0, space_pts):
			self.noise[i] = self.noise_rv[int(i*self.alpha)]



		# We count the time to the next switch
		self.count=0

		# We start the solver
		self.slv=neumann(delta_t, delta_x, space_pts)

	def do_step(self):

		# We do one more step in the implicit Euler approximations
		self.state_a = self.slv.apply_rslv( self.state - (np.multiply(np.multiply( self.state, self.state - 1), self.noise))*delta_t )

		self.count += delta_t

		if self.count > tau:
			self.noise_rv =  np.random.normal(size = int(self.noise_points), scale = sigma)

			self.noise = np.zeros(shape = space_pts)
			for i in range(0, space_pts):
				self.noise[i] = self.noise_rv[int(i*self.alpha)]

			self.count = 0

# This creates the picture

def animate(i):
	# Real time is:
	ani_time = i*delta_t

	# Redefine the plot
	lines.set_data(space, fkpp_sample.state)

	# Set the new time
	time_text.set_text("Time = {:2.3f}".format(ani_time) )
	# We print the step we are in:
	sys.stdout.flush()
	sys.stdout.write("\r Step = {}, Value = {}, Derivative = {}".format(i, fkpp_sample.state[middle], (delta_x**(-2))*(fkpp_sample.state[middle-1]+fkpp_sample.state[middle+1]-2*fkpp_sample.state[middle])))
	
	# And we do the next step:
	fkpp_sample.do_step()
	return [lines,] + [time_text,]


# Space-Time discretisation
delta_t = 1/1000
delta_x = 1/150

# Box size:
L = 1

# Space discretisation
space = np.arange(0.0, 2*np.pi*L + 0.001, delta_x)
space_pts = len(space)
middle = int(space_pts/4)-1

# We create a sample path
# with initial condition x_0:
x_0 = np.abs(0.5*np.sin(space))

# Tipical time
tau = 0.001

# Strength of the noise
sigma = 15

fkpp_sample = fkpp(x_0, tau, sigma, delta_t, delta_x, space_pts)

# We define the functional regarding which we want to consider the longtime
# averages.

def functional(state):

	return np.amax(np.multiply( state, state - 1))

#We set up the picture
fig       = plt.figure()
ax        = plt.axes(xlim=(0, 2*np.pi*L), ylim = (-0.2, 1.2))
time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
lines_a,  = ax.plot([],[], lw = 2)
lines_b,  = ax.plot([],[], lw = 2)
plt.title("FKPP Equation")

# We let the animation go.
ani       = animation.FuncAnimation(fig, animate, frames=50000, interval = 50, blit = True)

ani.save('fkpp.mp4', bitrate = 60000)

#ani.save(filename = 'fkpp_slow_noise.html', extra_args=['-vcodec', 'libx264'], bitrate = 20000)


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
