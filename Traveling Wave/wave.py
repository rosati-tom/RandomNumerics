
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


# Velocity
v_0 = 1.5
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
v = -v_0*Y - X*(1-X)

# Set up data
N = 200
x = np.linspace(0, 4*np.pi, N)
y = np.sin(x)
source = ColumnDataSource(data=dict(x=x, y=y))

# Bokeh plot 
plot = figure(plot_height=400, plot_width=400, title="Stream plot",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0, 4*np.pi], y_range=[-2.5, 2.5])

plot.streamplot('Hight', 'Velocity', source=source, line_width=3, line_alpha=0.6)
  

  
fig = plt.figure(figsize = (12, 7)) 
  
# Plotting stream plot 
plt.streamplot(X, Y, u, v, density = dens) 
  
# show plot 
plt.show() 

# Set up widgets
velocity = Slider(title="Velocity", value=2.5, start=0.0, end=5.0, step=0.03)
density = Slider(title="Plot density", value=3.5, start=0.0, end=5.0, step=0.03)

# Set up callbacks
def update_title(attrname, old, new):
    plot.title.text = text.value

text.on_change('value', update_title)

def update_data(attrname, old, new):

    # Get the current slider values
    vel = velocity.value
    d = density.value

    # Generate the new curve
    u = Y
  
	# y-component zero 
	v = -vel*Y - X*(1-X)
    source.data = dict(x=x, y=y)

for w in [offset, amplitude, phase, freq]:
    w.on_change('value', update_data)


# Set up layouts and add to document
inputs = column(text, offset, amplitude, phase, freq)

curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "Streamplot"