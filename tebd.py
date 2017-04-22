import numpy as np
import math
import helpers_new2
import datetime
from matplotlib import pyplot as plt

# Simulation and model parameters
model = {'J': 1.0, 'U': 0.0}
# Note: set 'it' == True to find ground state, False to calculate real time evolution
# If doing imaginary time evolution, we need to add in a chemical potential term to conserve number
# mu is in units of U
sim = {'d': 6, 'chi': 20, 'L': 3, 'delta': 0.1, 'N': 500, 'it': False, 'mu': 0.5}

# Choose which expectation values to log:
# Skip: how many iterations to skip between logging expectation values
# Must have skip = 'N' - 1 to log 'aa'
logs = {'rho': False, 'a': True, 'n': True, 'aa': True, 'skip': 0}

# Choose your initial state:
# state_flag = 0 for Fock states, = 1 for coherent states
# flag = 2 to initialize on site k
init = {'nbar': 1, 'flag': 2, 'site': 1};

# Which parameter(s) to sweep?
sweep_par = ['U', 'mu']
# What range to sweep over?
# Script will iterate through this array
par1_range = [model['U']] # np.arange(2, 2.2, 0.2)
par2_range = [sim['mu']] # np.arange(1, 0.2, 0.1)
sweep_range = [(x,y) for x in par1_range for y in par2_range]

for i in range(0, len(sweep_range)):
	for j in range(0, len(sweep_par)):
		if (sweep_par[j] in model):
			model[sweep_par[j]] = sweep_range[i][j]
		elif (sweep_par[j] in sim):
			sim[sweep_par[j]] = sweep_range[i][j]
		elif (sweep_par[j] in init):
			init[sweep_par[j]] = sweep_range[i][j]

	# Open a file for saving results
	filename = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
	f = open(filename + ".txt", 'w')
	# Log parameters to file
	f.write("d = {0}, chi = {1}, L = {2}\n".format(sim['d'], sim['chi'], sim['L']))
	f.write("delta = {0}, N = {1}\n".format(sim['delta'], sim['N']))
	f.write("J = {0}, U = {1}\n".format(model['J'], model['U']))
	f.write("nbar = {0}, flag = {1}\n".format(init['nbar'], init['flag']))
	f.write("rho = {0}, a = {1}, n = {2}, skip = {3}\n".format(logs['rho'], logs['a'], logs['n'], logs['skip']))

	# Run the simulation
	simulation = helpers_new2.TEBD(model, sim, init, logs)
	simulation.Run_Simulation()

	spdm = simulation.aa
	f.write("{0}\n".format(np.array2string(spdm)))

	# # Get the data
	# a_avg = simulation.a_avg
	# # Average to find average superfluid order parameter
	# a_gs = np.mean(np.absolute(a_avg[:,-1]))
	# print a_gs
	# f.write("order par = {0}\n".format(a_gs))

	f.write("error = {0}".format(simulation.tau))
	# Close file
	f.close()

	print sweep_range[i]

# print spdm
# print simulation.a_avg[:,-1]

a_avg = simulation.a_avg
# Plot stuff
L = sim['L']; chi = sim['chi']; d = sim['d']; delta = sim['delta']; N = sim['N']
f, ax = plt.subplots(L, sharex=True, sharey=True)
ts = np.linspace(0, (N+1)*delta, num=1+N/(logs['skip'] + 1))
indmax = int(1 + N/(logs['skip'] + 1))
for i in range(0, L):
	ax[i].plot(ts, np.absolute(a_avg[i,0:indmax]), 'bo', label="TEBD site {0}".format(i))
	ax[i].plot(ts, np.absolute(np.sqrt(init['nbar']) * np.exp(init['nbar'] * np.expm1(-1j * ts * model['U']))), 'r-', linewidth=2)
	ax[i].legend()
	ax[i].set_yticks(np.linspace(0,1.5,num=4))
	ax[i].set_yticklabels(np.linspace(0,1.5,num=4))
	ax[i].set_ylim([-0.4, 1.4])
ax[0].set_title(r"TEBD simulation: $L = {0}$, $d = {1}$, $\chi = {2}$".format(L,d,chi), fontsize=18)
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

f.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

plt.xlabel(r"t ($\hbar/U$)", fontsize=16)
plt.ylabel(r"$|\langle a(t) \rangle|$", fontsize=16)
plt.savefig(filename + "_a.pdf", format="pdf")

n_avg = simulation.n_avg
print n_avg[:,0]
print n_avg[:,-2]
print n_avg[:,-1]
# Plot stuff
L = sim['L']; chi = sim['chi']; d = sim['d']; delta = sim['delta']; N = sim['N']
f, ax = plt.subplots(L, sharex=True, sharey=True)
for i in range(0, L):
	ax[i].plot(ts, np.absolute(n_avg[i,0:indmax]), 'bo', label="TEBD site {0}".format(i))
	ax[i].legend()
	ax[i].set_yticks(np.linspace(0,3,num=4))
	ax[i].set_yticklabels(np.linspace(0,3,num=4))
	ax[i].set_ylim([-0.4, 3.4])
ax[0].set_title(r"TEBD simulation: $L = {0}$, $d = {1}$, $\chi = {2}$".format(L,d,chi), fontsize=18)
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

f.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

plt.xlabel(r"t ($\hbar/U$)", fontsize=16)
plt.ylabel(r"$\langle n(t) \rangle$", fontsize=16)
plt.savefig(filename + "_n.pdf", format="pdf")
plt.show()