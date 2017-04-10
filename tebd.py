import numpy as np
import math
import helpers

# Get simulation and model parameters
d = helpers.d; chi = helpers.chi; L = helpers.L; delta = helpers.delta; N = helpers.N
J = helpers.J; U = helpers.U;

# Choose your initial state:
occupation = 1.2; state_flag = 0;

# Initialize the state, and the various tensors we need:
init = helpers.Initialize_States(L, d, occupation, state_flag)
sim = helpers.TensorGroup(init)

# Unitary two-site operators
V_odd = helpers.V_odd
V_even = helpers.V_even
# Note that the last link has a slightly different form,
# therefore need separate unitaries
V_odd_last = helpers.V_odd_last
V_even_last = helpers.V_even_last

# Loop: do all the odds, then evens, then odds
for i in range(0, N):
	for h in range(0, L-1):
		if h % 2 == 1:
			if (h == L - 2):
				sim.Update(V_odd_last, h)
			else:
				sim.Update(V_odd, h)
	for j in range(0, L-1):
		if j % 2 == 0:
			if (j == L - 2):
				sim.Update(V_even_last, j)
			else:
				sim.Update(V_even, j)
	for k in range(0, L-1):
		if k % 2 == 1:
			if (k == L - 2):
				sim.Update(V_odd_last, k)
			else:
				sim.Update(V_odd, k)

# Print error accumulated
print sim.tau

# Calculate expectation values of a
a_avg = np.zeros(L, dtype=np.complex64)
for r in range(0, L):
	a_avg[r] = np.trace(np.dot(sim.Single_Site_Rho(r), helpers.a))

# Calculate expectation values of n
n_avg = np.zeros(L)
for r in range(0, L):
	n_avg[r] = np.trace(np.dot(sim.Single_Site_Rho(r), helpers.n_op))

print a_avg
print n_avg
# Print mean value of |<a>| = |Psi|
print (np.linalg.norm(np.absolute(a_avg))**2)/L
