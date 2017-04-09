import numpy as np
import math

# number cutoff, entanglement cutoff
d = 7; chi = 10; L = 10;
# parameters
J = 1; U = 1; delta = 0.01;

# Initialze state vectors (product state)
# L sites, number cutoff nmax
# n_onsite is the number on each site
# flag: 0 is Fock state, 1 is (truncated) coherent state with mean occupation = n_onsite
# flag = 0: if number_onsite is an integer, initialize n = n_onsite Mott insulator
# 		if not, initialize w/ probability (n_onsite - floor(n_onsite))
# flag = 1: approximate coherent state with alpha = sqrt(n_onsite)
def initialize_states(L, n_max, n_onsite, flag=0):
	# Initialize a matrix of zeros
	mat = np.zeros((L, n_max))

	# If not coherent states
	if (flag == 0):
		if (math.floor(n_onsite) >= 0):
			# Check that you're under the number cutoff
			if (math.floor(n_onsite) < n_max):
				mat[:,math.floor(n_onsite)] = 1
			# If not, initialize to vacuum
			else:
				return mat

			# Check for fractional occupation
			if (n_onsite != math.floor(n_onsite)):
				filling = n_onsite - math.floor(n_onsite)
				rands = np.random.rand(L)
				# Populate next highest state with probability = filling
				for i in range(0, L):
					if rands[i] < filling:
						mat[i,math.floor(n_onsite)+1] = 1
						mat[i,math.floor(n_onsite)] = 0
	# If coherent state
	elif (flag == 1):
		# Calculate normalization factor
		norm = 0
		for i in range(0, n_max):
			norm += math.pow(n_onsite, i) / math.factorial(i)
		norm = math.sqrt(norm)
		# Build coherent states
		for i in range(0, n_max):
			mat[:,i] = math.pow(n_onsite, float(i) / 2) / (math.sqrt(math.factorial(i)) * norm)
	return mat

# Initialize the coefficient tensors
# See http://inside.mines.edu/~lcarr/theses/mishmash_thesis_2008.pdf,
# Appendix A, assuming product wavefunctions
def lambda0():
	mat = np.zeros((L, chi))
	mat[:,0] = 1
	return mat

def Gamma0(coeffs):
	mat = np.zeros((L, d, chi, chi))
	for i in range(0, L):
		mat[i, :, 0, 0] = coeffs[i,:]
	return mat

init = initialize_states(L, d, 1.2, flag=1)
print init
print lambda0()
print Gamma0(init)


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






