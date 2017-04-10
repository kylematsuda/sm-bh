import numpy as np
import math

# Model parameters:
J = 1; U = 1.5 * J;

# # Cutoff for small numbers
# cutoff = 1E-10

# Simulation parameters:
# d = Number cutoff, chi = Entanglement cutoff,
# L = Number of sites
# Note: require chi > d!!!
d = 5; chi = 50; L = 10; delta = 0.01 / J; N = 100;

# Class for handling the Lambda, Gamma, and Theta tensors
class TensorGroup(object):

	def __init__(self, coeffs):
		self.Lambda = lambda0()
		self.Gamma = Gamma0(coeffs)
		self.tau = 0

	def Build_Theta(self, l):
		Lambda = self.Lambda
		Gamma_lp1 = self.Gamma[l+1]
		Gamma_l = self.Gamma[l]

		# if (l != 0 and l != L-2):
		# 	theta = np.tensordot(np.diag(Lambda[l,:]), Gamma_lp1[:,:,:], axes=(1,1))
		# 	theta = np.tensordot(Gamma_l[:,:,:], theta, axes=(2,0))
		# 	theta = np.tensordot(np.diag(Lambda[l-1,:]), theta, axes=(1,1))
		# 	theta = np.tensordot(theta, np.diag(Lambda[l+1,:]), axes=(-1, 0))

		# 	# Now indices are [a_(l-1), i_l, i_(l+1), a_(l+1)]
		# 	# Swap them to [i_l, i_(l+1), a_(l-1), a_(l+1)] to avoid confusion
		# 	theta = np.transpose(theta,(1,2,0,3))
	
		# # Structure of Gamma^(0) and Gamma^(L-1) are different...
		# elif (l == 0):
		# 	theta = np.tensordot(np.diag(Lambda[l,:]), Gamma_lp1[:,:,:], axes=(1,1))
		# 	theta = np.tensordot(Gamma_l[:,:], theta, axes=(1,0))
		# 	theta = np.tensordot(theta, np.diag(Lambda[l+1,:]), axes=(-1, 0))
		# elif (l == L-2):
		# 	theta = np.tensordot(np.diag(Lambda[l,:]), Gamma_lp1[:,:], axes=(1,1))
		# 	theta = np.tensordot(Gamma_l[:,:,:], theta, axes=(-1,0))
		# 	theta = np.tensordot(np.diag(Lambda[l-1,:]), theta, axes=(1,1))
		# 	# Swap indices to be (i_l, i_l + 1, a_(l-1))
		# 	theta = np.transpose(theta, (1,2,0))
		# return theta


		# Trying alternate def of Theta to reduce numerical errors
		if (l != 0 and l != L-2):
			theta = np.tensordot(np.diag(Lambda[l,:]), Gamma_lp1[:,:,:], axes=(1,1))
			theta = np.tensordot(Gamma_l[:,:,:], theta, axes=(2,0))

			# # Now indices are [i_l, a_(l-1), i_(l+1), a_(l+1)]
			# # Swap them to [i_l, i_(l+1), a_(l-1), a_(l+1)] to avoid confusion
			theta = np.transpose(theta, (0,2,1,3))

		# Structure of Gamma^(0) and Gamma^(L-1) are different...
		elif (l == 0):
			theta = np.tensordot(np.diag(Lambda[l,:]), Gamma_lp1[:,:,:], axes=(1,1))
			theta = np.tensordot(Gamma_l[:,:], theta, axes=(1,0))
			# Indices already [i_l, i_(l + 1), a_(l + 1)]
		elif (l == L-2):
			theta = np.tensordot(np.diag(Lambda[l,:]), Gamma_lp1[:,:], axes=(1,1))
			theta = np.tensordot(Gamma_l[:,:,:], theta, axes=(-1,0))
			# Swap indices to be [i_l, i_(l + 1), a_(l - 1)]
			theta = np.transpose(theta, (0,2,1))
		return theta

	# This follows the discussion in
	# http://inside.mines.edu/~lcarr/theses/mishmash_thesis_2008.pdf,
	# section 3.2.3.
	# Note that while the linked thesis indexes from 1,
	# we index from zero here to avoid confusion in the code.
	# Therefore, l runs from [0, ..., L - 1] in our case,
	# instead of [1, ..., L] as shown in the thesis.
	def Update(self, V, l):
		# Build the appropriate Theta tensor
		Theta = self.Build_Theta(l)
		# Apply the unitary matrix V
		Theta = np.tensordot(V, Theta, axes=([-2,-1], [0,1]))
		
		# Need to treat boundary subsystems differently...
		if (l != 0 and l != L - 2):
			# Reshape to a square matrix and do singular value decomposition:
			Theta = np.reshape(np.transpose(Theta, (0,2,1,3)), (d*chi, d*chi))
			# A and transpose.C contain the new Gamma[l] and Gamma[l+1]
			# B contains new Lambda[l]
			A, B, C = np.linalg.svd(Theta)
			C = np.transpose(C)

			# Truncate at chi eigenvalues and enforce normalization
			self.Lambda[l, :] = B[0:chi] / np.linalg.norm(B[0:chi])

			# Keep track of the truncation error accumulated on this step
			self.tau += delta * (1 - np.linalg.norm(B[0:chi])**2)

			# Find the new Gammas

			# Gamma_l:
			# First, reshape A
			A = np.reshape(A[:, 0:chi], (d, chi, chi))
			# Next, "divide off" Lambda[l-1]
			# (see http://inside.mines.edu/~lcarr/theses/mishmash_thesis_2008.pdf,
			#	equation 3.40)
			
			# Trying alternative def of Theta:
			# Gamma_l_new = np.tensordot(np.diag(OneOver(self.Lambda[l-1,:])), A, axes=(1,1))
			# Gamma_l_new = np.transpose(Gamma_l_new, (1,0,2))
			Gamma_l_new = A
			self.Gamma[l] = Gamma_l_new

			# Gamma_(l+1):
			# Do the same thing (note that C has already been transposed)
			C = np.reshape(C[:, 0:chi], (d, chi, chi))
			
			# Trying alternative def of Theta
			# Gamma_lp1_new = np.tensordot(np.diag(OneOver(self.Lambda[l+1,:])), C, axes=(1,1))
			# Gamma_lp1_new = np.transpose(Gamma_lp1_new, (1,0,2))
			Gamma_lp1_new = C
			self.Gamma[l+1] = Gamma_lp1_new

		# The Gamma_0 tensor has one less index... need to treat slightly differently
		elif (l == 0):
			# Reshape to a square matrix and do singular value decomposition:
			Theta = np.reshape(Theta, (d, d*chi))
			# A and transpose.C contain the new Gamma[l] and Gamma[l+1]
			# B contains new Lambda[l]
			A, B, C = np.linalg.svd(Theta)
			C = np.transpose(C)

			# Enforce normalization
			# Don't need to truncate here because chi is bounded by
			# the dimension of the smaller subsystem, here equals d < chi
			self.Lambda[l,0:d] = B / np.linalg.norm(B)

			# Keep track of the truncation error accumulated on this step
			self.tau += delta * (1 - np.linalg.norm(B)**2)

			# Find the new Gammas
			# Gamma_l:
			self.Gamma[l][:,0:d] = A

			# Gamma_(l+1):
			# Treat the l = 1 case normally...
			C = np.reshape(C[:,0:chi], (d, chi, chi))
			# Gamma_lp1_new = np.tensordot(np.diag(OneOver(self.Lambda[l+1,:])), C, axes=(1,1))
			# Gamma_lp1_new = np.transpose(Gamma_lp1_new, (1,0,2))
			Gamma_lp1_new = C
			self.Gamma[l+1] = Gamma_lp1_new

		elif (l == L - 2):
			# Reshape to a square matrix and do singular value decomposition:
			Theta = np.reshape(np.transpose(Theta, (0,2,1)), (d*chi, d))
			# A and transpose.C contain the new Gamma[l] and Gamma[l+1]
			# B contains new Lambda[l]
			A, B, C = np.linalg.svd(Theta)
			C = np.transpose(C)

			# Enforce normalization
			# Don't need to truncate here because chi is bounded by
			# the dimension of the smaller subsystem, here equals d < chi
			self.Lambda[l,0:d] = B / np.linalg.norm(B)

			# Keep track of the truncation error accumulated on this step
			self.tau += delta * (1 - np.linalg.norm(B)**2)

			# Find the new Gammas

			# Treat the L-2 case normally:
			A = np.reshape(A[:,0:chi], (d, chi, chi))
			# Gamma_l_new = np.tensordot(np.diag(OneOver(self.Lambda[l-1,:])), A, axes=(1,1))
			# Gamma_l_new = np.transpose(Gamma_l_new, (1,0,2))
			Gamma_l_new = A
			self.Gamma[l] = Gamma_l_new

			# Gamma_(L-1):
			self.Gamma[l+1][:,0:d] = C
		# print (l, self.tau)
		# print B
		# print (l, np.shape(A), np.shape(C))


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
	mat = np.zeros((L, n_max), dtype=np.complex64)

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
	mat = np.zeros((L-1, chi), dtype=np.complex64)
	mat[:,0] = 1
	return mat

def Gamma0(coeffs):
	Gamma = []

	mat = np.zeros((d, chi), dtype=np.complex64)
	mat[:,0] = coeffs[0,:]
	Gamma.append(mat)

	for i in range(1, L-1):
		mat = np.zeros((d, chi, chi), dtype=np.complex64)
		mat[:, 0, 0] = coeffs[i,:]
		Gamma.append(mat)

	mat = np.zeros((d, chi), dtype=np.complex64)
	mat[:,0] = coeffs[L-1,:]
	Gamma.append(mat)
	return Gamma

# # Divide a 1d array
# def OneOver(array):
# 	for i in range(0, len(array)):
# 		if np.absolute(array[i]) >= cutoff:
# 			array[i] = 1/complex(array[i])
# 		else:
# 			array[i] = 0
# 	return array


# Operator definitions:

# Build a, a_dag, n, n_2site
a = np.zeros((d,d))
for i in range(0,d-1):
	a[i, i+1] = math.sqrt(i+1)
a_dag = np.transpose(a)
n_op = np.dot(a_dag, a)
# Build two site Hamiltonian:
H_2site = -J * (np.kron(a_dag, a) + np.kron(a, a_dag)) + (U / 2) * np.kron(np.dot(n_op, n_op - np.identity(d)), np.identity(d))
# Need to treat the last link differently...
# Need to include onsite term for the last site also
H_2site_last = -J * (np.kron(a_dag, a) + np.kron(a, a_dag)) + (U / 2) * np.kron(np.dot(n_op, n_op - np.identity(d)), np.identity(d)) + (U/2) * np.kron(np.identity(d), np.dot(n_op, n_op - np.identity(d)))
# Diagonalize 
w,v = np.linalg.eig(H_2site)
V_odd = np.reshape(np.dot(np.dot(v,np.diag(np.exp(-1j*delta*(w) / 2))), np.transpose(v)), (d,d,d,d))
V_even = np.reshape(np.dot(np.dot(v,np.diag(np.exp(-1j*delta*(w)))), np.transpose(v)), (d,d,d,d))
# Same thing for last site
wp, vp = np.linalg.eig(H_2site_last)
V_odd_last = np.reshape(np.dot(np.dot(vp,np.diag(np.exp(-1j*delta*(wp) / 2))), np.transpose(vp)), (d,d,d,d))
V_even_last = np.reshape(np.dot(np.dot(vp,np.diag(np.exp(-1j*delta*(wp)))), np.transpose(vp)), (d,d,d,d))


