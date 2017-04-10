import numpy as np
import math
import helpers

# Init simulation and model parameters
d = helpers.d; chi = helpers.chi; L = helpers.L; delta = helpers.delta; N = helpers.N
J = helpers.J; U = helpers.U;
# Choose your initial state:
occupation = 1.3; state_flag = 0;

# Initialize the state, and the various tensors we need:
init = helpers.Initialize_States(L, d, occupation, state_flag)
sim = helpers.TensorGroup(init)

# Unitary two-site operators
V_odd = helpers.V_odd
V_odd_sq = helpers.V_odd_sq
V_even = helpers.V_even

# Initial step: do all the odd guys
for i in range(0, L-2):
	if i % 2 == 1:
		sim.Update(V_odd, i)

# Loop: do all the evens, then all the odds (twice)
for i in range(0, N):
	for i in range(0, L-2):
		if i % 2 == 0:
			sim.Update(V_even, i)
	for i in range(0, L-2):
		if i % 2 == 1:
			sim.Update(V_odd_sq, i)

# Final step: do all the odd guys
for i in range(0, L-2):
	if i % 2 == 1:
		sim.Update(V_odd, i)

print sim.Gamma[0]