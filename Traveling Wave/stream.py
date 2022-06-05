
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from IPython import embed
from os import path
import sys


# Velocity
vel = 2.0
# Density of the plot
dens = 3.5

# We set up the plot

# Create the axes 
x = np.arange(-1, 2, 0.05) 
y = np.arange(-2, 2, 0.05) 
  
# Create the meshgrid 
X, Y = np.meshgrid(x, y) 

# Initiation of the plot
# hight component  
u = Y  
# velocity component  
v = vel*Y - X*(1-X)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

# Plotting stream plot 
ax.streamplot(X, Y, u, v, density = dens) 

axcolor = 'lightgoldenrodyellow'
axvel = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

svel = Slider(axvel, 'Velocity', -1.0, 5.0, valinit=vel, valstep=0.02)

def update(val):
    vel = svel.val
    ax.cla()
    ax.streamplot(X, Y, Y,vel*Y - X*(1-X), density = dens)

svel.on_changed(update)

#resetax = plt.axes([0.8, 0.025, 0.1, 0.04])


def reset(event):
    svel.reset()

plt.show()