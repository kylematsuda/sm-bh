import numpy as np
import math
import helpers
import datetime
from matplotlib import pyplot as plt

# Open a file for saving results
filename = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
f = open(filename + ".txt", 'w')

# Simulation and model parameters
model = {'J': 0.0, 'U': 20.0}
sim = {'d': 8, 'chi': 50, 'L': 10, 'delta': 0.01, 'N': 100}

# Choose which expectation values to log:
# Skip: how many iterations to skip between logging expectation values
logs = {'rho': False, 'a': True, 'n': False, 'skip': 0}

# Choose your initial state:
# state_flag = 0 for Fock states, = 1 for coherent states
init = {'nbar': 0.7, 'flag': 1};

# Log parameters to file
f.write("d = {0}, chi = {1}, L = {2}\n".format(sim['d'], sim['chi'], sim['L']))
f.write("delta = {0}, N = {1}\n".format(sim['delta'], sim['N']))
f.write("J = {0}, U = {1}\n".format(model['J'], model['U']))
f.write("nbar = {0}, flag = {1}\n".format(init['nbar'], init['flag']))
f.write("rho = {0}, a = {1}, n = {2}, skip = {3}".format(logs['rho'], logs['a'], logs['n'], logs['skip']))

# Run the simulation
simulation = helpers.TEBD(model, sim, init, logs)
simulation.Run_Simulation()

# Get the data
a_avg = simulation.a_avg

# Close file
f.close()

# Plot stuff
L = sim['L']; chi = sim['chi']; d = sim['d']; delta = sim['delta']; N = sim['N']
for i in range(0, L):
	plt.figure()
	plt.plot(np.arange(0, (N+1)*delta, delta), np.absolute(a_avg[i,:]), label="site {0}".format(i))
	plt.legend()
	plt.title(r"TEBD simulation: $L = {0}$, $d = {1}$, $\chi = {2}$".format(L,d,chi))
	plt.xlabel(r"t ($\hbar/J$)")
	plt.ylabel(r"$\langle a(t) \rangle$")
# plt.savefig(filename + "_a.pdf", format="pdf")
plt.show()


### OLD VERSION BELOW ###

# # Open a file for saving results
# filename = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
# f = open(filename + ".txt", 'w')

# # Get simulation and model parameters
# d = helpers.d; chi = helpers.chi; L = helpers.L; delta = helpers.delta; N = helpers.N
# J = helpers.J; U = helpers.U;
# f.write("d = {0}, chi = {1}, L = {2}, delta = {3}, N = {4}\n".format(d, chi, L, delta, N))
# f.write("J = {0}, U = {1}\n".format(J, U))

# # Choose your initial state:
# # state_flag = 0 for Fock states, = 1 for coherent states
# occupation = 0.7; state_flag = 1;
# f.write("occupation = {0}, state_flag = {1}\n".format(occupation, state_flag))

# # Initialize the state, and the various tensors we need:
# init = helpers.Initialize_States(L, d, occupation, state_flag)
# sim = helpers.TensorGroup(init)

# # Unitary two-site operators
# V_odd = helpers.V_odd
# V_even = helpers.V_even

# # Array to hold expectation values
# a_avg = np.zeros((L, N+1), dtype=np.complex64)

# # Calculate initial expectation values of |a|
# for r in range(0, L):
# 	a_avg[r,0] = np.trace(np.dot(sim.Single_Site_Rho(r), helpers.a))

# # Loop: do all the odds, then evens, then odds
# for i in range(1, N+1):
# 	# Evolve odd links delta * t / 2
# 	for h in range(0, L-1):
# 		if h % 2 == 1:
# 			sim.Update(V_odd, h)
# 	if ((L-1) % 2 == 1):
# 		sim.Update(helpers.V_odd_last, L-2)	
# 	# Evolve even links delta * t
# 	for j in range(0, L-1):
# 		if j % 2 == 0:
# 			sim.Update(V_even, j)
# 	if ((L-1) % 2 == 0):
# 		sim.Update(helpers.V_even_last, L-2)	
# 	# Evolve odd links delta * t / 2
# 	for k in range(0, L-1):
# 		if k % 2 == 1:
# 			sim.Update(V_odd, k)
# 	if ((L-1) % 2 == 1):
# 		sim.Update(helpers.V_odd_last, L-2)	
	
# 	# Calculate expectation values of |a|
# 	for r in range(0, L):
# 		a_avg[r,i] = np.trace(np.dot(sim.Single_Site_Rho(r), helpers.a))

# 	if (i % 50 == 0):
# 		print "step {0} done".format(i)


# # Print error accumulated
# f.write("error = {0}".format(sim.tau))
# # Close file
# f.close()


# for i in range(0, L):
# 	plt.figure()
# 	plt.plot(np.arange(0, (N+1)*delta, delta), np.absolute(a_avg[i,:]), label="site {0}".format(i))
# 	plt.legend()
# 	plt.title(r"TEBD simulation: $L = {0}$, $d = {1}$, $\chi = {2}$".format(L,d,chi))
# 	plt.xlabel(r"t ($\hbar/J$)")
# 	plt.ylabel(r"$\langle a(t) \rangle$")
# # plt.savefig(filename + "_a.pdf", format="pdf")
# plt.show()
