import numpy as np
import math
import helpers

# number cutoff, entanglement cutoff
d = helpers.d; chi = helpers.chi; L = helpers.L;
# parameters
J = 1; U = 1; delta = 0.01;

init = helpers.Initialize_States(L, d, 1.2, flag=1)
thing = helpers.TensorGroup(init)

# build a, a_dag, n, n_2site
a = np.zeros((d,d))
for i in range(0,d-1):
	a[i, i+1] = math.sqrt(i+1)
a_dag = np.transpose(a)
n = np.dot(a_dag, a)
n_2site = np.kron(n, np.identity(d))
# build two site Hamiltonian:
H_2site = -J * (np.kron(a_dag, a) + np.kron(a, a_dag)) + (U / 2) * np.dot(n_2site, n_2site - np.identity(d**2))
# Diagonalize 
w,v = np.linalg.eig(H_2site)
U = np.reshape(np.dot(np.dot(v,np.diag(np.exp(-delta*(w)))), np.transpose(v)), (d,d,d,d))

thing.update(U, 1)


