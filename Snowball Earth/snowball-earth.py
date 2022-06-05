
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from IPython import embed
from os import path
import sys

import scipy.integrate as integrate



# Solar constant
s_0 = 1.5

# Greenhouse effect constant
g_0 = 0.2

# We set up the plot

# Create the axes 
T     = np.arange(200, 310, 0.05) 
T_pot = np.arange(150, 350, 0.05) 
  
def coalbedo(T):
	return 15+ 5*np.tanh((T-265)/5)

def potential(T, s_0, g_0):
	return -integrate.quad(lambda x: s_0*coalbedo(x)-(10**(-8))*(1-g_0)*(x**4), 190, T)[0]

v_potential = np.vectorize(potential)


# Initiation of the plot
fig, (ax_1, ax_2) = plt.subplots(1,2)
plt.subplots_adjust(left=0.25, bottom=0.25)

# Plotting stream plot 
ax_1.plot(T, s_0*coalbedo(T), T, (10**(-8))*(1-g_0)*(T**4))
ax_2.plot(T_pot, v_potential(T_pot, s_0, g_0))

axcolor = 'lightgoldenrodyellow'
ax_solar = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_green = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)


slider_solar = Slider(ax_solar, 'Solar constant', 0.0, 4.3, valinit=s_0, valstep=0.003)
slider_green = Slider(ax_green, 'Greenhouse effect', 0.0, 1.0, valinit=g_0, valstep=0.003)


def update(val):
    s_0 = slider_solar.val
    g_0 = slider_green.val
    ax_1.cla()
    ax_2.cla()
    ax_1.plot(T, s_0*coalbedo(T), T, (10**(-8))*(1-g_0)*(T**4))
    ax_2.plot(T_pot, v_potential(T_pot, s_0, g_0))

slider_solar.on_changed(update)
slider_green.on_changed(update)

#resetax = plt.axes([0.8, 0.025, 0.1, 0.04])

def reset(event):
    slider_solar.reset()
    slider_green.reset()


plt.show()