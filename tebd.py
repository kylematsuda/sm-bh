import numpy as np
import math
import helpers

# Init simulation and model parameters
d = helpers.d; chi = helpers.chi; L = helpers.L; delta = helpers.delta
J = helpers.J; U = helpers.U;
# Choose your initial state:
occupation = 1.3; state_flag = 0;

# Initialize the state, and the various tensors we need:
init = helpers.Initialize_States(L, d, occupation, state_flag)
sim = helpers.TensorGroup(init)

# Unitary two-site operator
V = helpers.V_2site


sim.Update(V, 0)
print sim.Gamma