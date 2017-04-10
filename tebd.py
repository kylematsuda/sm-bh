import numpy as np
import math
import helpers

# Init simulation and model parameters
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

# Loop: do all the odds, then evens, then odds
for i in range(0, N):
	for h in range(0, L-1):
		if h % 2 == 1:
			sim.Update(V_odd, h)
	for j in range(0, L-1):
		if j % 2 == 0:
			sim.Update(V_even, j)
	for k in range(0, L-1):
		if k % 2 == 1:
			sim.Update(V_odd, k)

print sim.tau