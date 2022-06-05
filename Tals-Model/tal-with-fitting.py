

import numpy as np
import scipy
import scipy.stats
import pandas as pd
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
NN = 5
tryals = 50000
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


MAXNUM = 30000
samples_model = np.zeros(shape = MAXNUM)
samples_unifo = np.zeros(shape = MAXNUM)

fig, (ax1, ax2) = plt.subplots(2)

INCRNUM = 0

for x in range(0,MAXNUM):
	
	model.state = np.random.uniform(low = 0.0, high = box_size , size=NN)
	for y in range(0,5000):
		model.do_step()

	samples_model[x] = model.incr[INCRNUM]

	sys.stdout.flush()
	sys.stdout.write("\r Step = {}, {}".format(INCRNUM, x))


# Code from: https://pythonhealthcare.org/2018/05/03/81-distribution-fitting-to-data/

y_std = samples_model
# y_std = my_uni_incr[1,:]
size  = len(y_std)

# Set list of distributions to test
# See https://docs.scipy.org/doc/scipy/reference/stats.html for more

# Turn off code warnings (this is not recommended for routine use)
# import warnings
# warnings.filterwarnings("ignore")

# Set up list of candidate distributions to use
# See https://docs.scipy.org/doc/scipy/reference/stats.html for more

dist_names = ['beta',
              'expon',
              'gamma',
              'lognorm',
              'norm',
              'pearson3',
              'triang',
              'uniform',
              'weibull_min', 
              'weibull_max']

# Set up empty lists to stroe results
chi_square = []
p_values = []

# Set up 50 bins for chi-square test
# Observed data will be approximately evenly distrubuted aross all bins
percentile_bins = np.linspace(0,100,51)  
percentile_cutoffs = np.percentile(y_std, percentile_bins)
observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
cum_observed_frequency = np.cumsum(observed_frequency)

# Loop through candidate distributions

for distribution in dist_names:
    # Set up distribution and get fitted distribution parameters
    dist = getattr(scipy.stats, distribution)
    param = dist.fit(y_std)
    
    # Obtain the KS test P statistic, round it to 5 decimal places
    p = scipy.stats.kstest(y_std, distribution, args=param)[1]
    p = np.around(p, 5)
    p_values.append(p)    
    
    # Get expected counts in percentile bins
    # This is based on a 'cumulative distrubution function' (cdf)
    cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2], 
                          scale=param[-1])
    expected_frequency = []
    for bin in range(len(percentile_bins)-1):
        expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
        expected_frequency.append(expected_cdf_area)
    
    # calculate chi-squared
    expected_frequency = np.array(expected_frequency) * size
    cum_expected_frequency = np.cumsum(expected_frequency)
    ss = sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
    chi_square.append(ss)
        
# Collate results and sort by goodness of fit (best at top)

results = pd.DataFrame()
results['Distribution'] = dist_names
results['chi_square'] = chi_square
results['p_value'] = p_values
results.sort_values(['chi_square'], inplace=True)
    
# Report results

print ('\nDistributions sorted by goodness of fit:')
print ('----------------------------------------')
print (results)

embed()
