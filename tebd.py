import numpy as np
import math

# number cutoff
d = 7
# parameters
J = 1; U = 1; delta = 0.01;

# Initialze state vectors
# L sites, number cutoff nmax
# number_onsite is the number on each site
# if number_onsite is an integer, initialize n = number_onsite Mott insulator
# if not, initialize w/ probability (number_onsite - floor(number_onsite))
# flag: 0 is Fock state, 1 is (truncated) coherent state with mean occupation = number_onsite
def initialize_states(L, n_max, n_onsite, flag=0):
	mat = np.zeros((L, n_max))

	if (flag == 0):
		if (math.floor(n_onsite) >= 0):
			# Check that you're under the number cutoff
			if (math.floor(n_onsite) < n_max):
				mat[:,math.floor(n_onsite)] = 1
			# Initialize to vacuum -- you done fucked up
			else:
				return mat

			# Check for fractional occupation
			if (n_onsite != math.floor(n_onsite)):
				filling = n_onsite - math.floor(n_onsite)
				rands = np.random.rand(L)
				for i in range(0, L):
					if rands[i] < filling:
						mat[i,math.floor(n_onsite)+1] = 1
						mat[i,math.floor(n_onsite)] = 0
	elif (flag == 1):
		norm = 0
		for i in range(0, n_max):
			norm += math.pow(n_onsite, i) / math.factorial(i)
		norm = math.sqrt(norm)
		for i in range(0, n_max):
			mat[:,i] = math.pow(n_onsite, float(i) / 2) / (math.sqrt(math.factorial(i)) * norm)
	return mat




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


