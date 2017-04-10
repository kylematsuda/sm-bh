import numpy as np
import math

# Simulation parameters:
# d = Number cutoff, chi = Entanglement cutoff,
# L = Number of sites
d = 7; chi = 10; L = 10; delta = 0.01

# Model parameters:
J = 1; U = 1;

# Class for handling the Lambda, Gamma, and Theta tensors
class TensorGroup(object):

	def __init__(self, coeffs):
		self.Lambda = lambda0()
		self.Gamma = Gamma0(coeffs)

	def Build_Theta(self, l):
		Lambda = self.Lambda
		Gamma = self.Gamma
		theta = np.tensordot(np.diag(Lambda[l,:]), Gamma[l+1,:,:,:], axes=(1,1))
		theta = np.tensordot(Gamma[l,:,:,:], theta, axes=(2,0))
		theta = np.tensordot(np.diag(Lambda[l-1,:]), theta, axes=(1,1))
		theta = np.tensordot(theta, np.diag(Lambda[l+1,:]), axes=(-1, 0))

		# Now indices are [a_(l-1), i_l, i_(l+1), a_(l+1)]
		# Swap them to [i_l, i_(l+1), a_(l-1), a_(l+1)] to avoid confusion
		theta = np.transpose(theta,(1,2,0,3))
		return theta

	def Update(self, V, l):
		# Build the appropriate Theta tensor
		Theta = self.Build_Theta(l)
		# Apply the unitary matrix V
		Theta = np.tensordot(V, Theta, axes=([2,3], [0,1]))
		
		# Reshape to a square matrix and do singular value decomposition:
		Theta = np.reshape(np.transpose(Theta, (0,2,1,3)), (d*chi, d*chi))
		# A and transpose.C contain the new Gamma[l] and Gamma[l+1]
		# B contains new Lambda[l]
		A, B, C = np.linalg.svd(Theta); C = C.T

		# Truncate at chi eigenvalues and enforce normalization
		self.Lambda[l, :] = B[0:chi] / np.linalg.norm(B[0:chi])

		# Find the new Gammas

		# Gamma_l:
		# First, reshape A
		A = np.reshape(A[:,0:chi], (d, chi, chi))
		# Next, "divide off" Lambda[l-1]
		# (see http://inside.mines.edu/~lcarr/theses/mishmash_thesis_2008.pdf,
		#	equation 3.40)
		Gamma_l_new = np.tensordot(np.diag(OneOver(self.Lambda[l-1,:])), A, axes=(1,1))
		Gamma_l_new = np.transpose(Gamma_l_new, (1,0,2))
		self.Gamma[l,:,:,:] = Gamma_l_new

		# Gamma_(l+1):
		# Do the same thing (note that C has already been transposed)
		C = np.reshape(C[:,0:chi], (d, chi, chi))
		Gamma_lp1_new = np.tensordot(np.diag(OneOver(self.Lambda[l+1,:])), C, axes=(1,1))
		Gamma_lp1_new = np.transpose(Gamma_lp1_new, (1,0,2))
		self.Gamma[l+1,:,:,:] = Gamma_lp1_new

# Helper functions for initialization:

# Initialize state vectors (product state)
# L sites, number cutoff nmax
# n_onsite is the number on each site
# flag: 0 is Fock state, 1 is (truncated) coherent state with mean occupation = n_onsite
# flag = 0: if number_onsite is an integer, initialize n = n_onsite Mott insulator
# 		if not, initialize w/ probability (n_onsite - floor(n_onsite))
# flag = 1: approximate coherent state with alpha = sqrt(n_onsite)
def Initialize_States(L, n_max, n_onsite, flag):
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
		norm = np.sqrt(norm)
		# Build coherent states
		for i in range(0, n_max):
			mat[:,i] = math.pow(n_onsite, float(i) / 2) / (np.sqrt(math.factorial(i)) * norm)
	return mat

# Initialize the coefficient tensors
# See http://inside.mines.edu/~lcarr/theses/mishmash_thesis_2008.pdf,
# Appendix A, assuming product wavefunctions
# Important not to inadverdently cast to real numbers instead of complex!
def lambda0():
	mat = np.zeros((L, chi), dtype=np.complex64)
	mat[:,0] = 1
	return mat

def Gamma0(coeffs):
	mat = np.zeros((L, d, chi, chi), dtype=np.complex64)
	for i in range(0, L):
		mat[i, :, 0, 0] = coeffs[i,:]
	return mat

# Divide a 1d array
def OneOver(array):
	for i in range(0, len(array)):
		if array[i] != 0:
			array[i] = 1/complex(array[i])
	return array


# Build a, a_dag, n, n_2site
a = np.zeros((d,d))
for i in range(0,d-1):
	a[i, i+1] = math.sqrt(i+1)
a_dag = np.transpose(a)
n_op = np.dot(a_dag, a)
n_2site = np.kron(n_op, np.identity(d))
# build two site Hamiltonian:
H_2site = -J * (np.kron(a_dag, a) + np.kron(a, a_dag)) + (U / 2) * np.dot(n_2site, n_2site - np.identity(d**2))
# Diagonalize 
w,v = np.linalg.eig(H_2site)
V_2site = np.reshape(np.dot(np.dot(v,np.diag(np.exp(-delta*(w)))), np.transpose(v)), (d,d,d,d))

