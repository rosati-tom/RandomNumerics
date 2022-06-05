
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from IPython import embed
from os import path
import sys


# Density of the plot
dens = 2.5

# We set up the plot

# Create the axes 
x = np.arange(0, 5, 0.05) 
y = np.arange(0, 5, 0.05) 
  
# Create the meshgrid 
X, Y = np.meshgrid(x, y) 

# Initiation of the plot
a_1 = 1
a_2 = 1
b_1 = 1
b_2 = 1
# Prey  
u = a_1*X - a_1*Y*X 
# Predator  
v = -b_1*Y + b_2*X*Y

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

# Plotting stream plot 
ax.streamplot(X, Y, u, v, density = dens) 

axcolor = 'lightgoldenrodyellow'
axdens = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

sdens = Slider(axdens, 'Density', 0.0, 5.0, valinit=dens, valstep=0.02)

def update(val):
    dens = sdens.val
    ax.cla()
    ax.streamplot(X, Y, a_1*X - a_1*Y*X,-b_1*Y + b_2*X*Y, density = dens)

sdens.on_changed(update)

def reset(event):
    sdens.reset()

plt.show()