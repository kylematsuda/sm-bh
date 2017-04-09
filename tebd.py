import numpy as np
import math

# dimension cutoff
d = 3
# parameters
J = 1; U = 1;

# build a, a_dag
a = np.zeros((d,d))
for i in range(0,d-1):
	a[i, i+1] = math.sqrt(i+1)
a_dag = np.transpose(a)
n = np.dot(a_dag, a)
n_2site = np.kron(n, np.identity(d))
# build two site Hamiltonian:
H_2site = -J * (np.kron(a_dag, a) + np.kron(a, a_dag)) + (U / 2) * np.dot(n_2site, n_2site - np.identity(d**2))  


print a
print a_dag
print n
print H_2site