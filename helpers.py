# Kyle Matsuda, Tanya Roussy, and Will Tobias
#
# Helper and initialization functions for our
# implementation of the TEBD algorithm for the 
# final project of Physics 7230.
#
# Sources:
# Ryan Mishmash Master's thesis: http://inside.mines.edu/~lcarr/theses/mishmash_thesis_2008.pdf
# Urbanek and Soldan, Computer Physics Communications 199, 170-177 (2016).
#
# The script tebd.py controls the parameters of the simulation,
# runs the simulation, and plots data.
#
# Note: this is the cleaned-up version of the code;
# partial implementations of two-site reduced density matrices,
# imaginary time evolution, two-species evolution can be accessed
# in older versions of the code on github.

import numpy as np
import math

# Class for handling the Lambda, Gamma, and Theta tensors
class TEBD(object):
	# Initialize
	def __init__(self, model, sim, init, logs):
		self.Lambda = lambda0(sim)
		coeffs = Initialize_States(sim, init)
		self.tensorA = Gamma0(coeffs, sim)
		L = sim['L']
		# Build the A_l tensors from the Gamma_l's
		for i in range(1, L):
			self.tensorA[i] = np.transpose(np.tensordot(np.diag(self.Lambda[i-1]), self.tensorA[i], axes=(0,1)), (1,0,2))
		self.tau = 0
		self.model = model
		self.sim = sim
		self.init = init
		self.logs = logs
		self.rhos = np.zeros((sim['L'], sim['N']+1, sim['d'], sim['d']), dtype=np.complex64)
		self.a_avg = np.zeros((sim['L'], sim['N']+1), dtype=np.complex64)
		self.n_avg = np.zeros((sim['L'], sim['N']+1))
		self.aa = np.zeros((sim['L'], sim['L']), dtype=np.complex64)

	# The simulation loop
	def Run_Simulation(self):
		# Define simulation parameters
		N = self.sim['N']
		L = self.sim['L']
		d = self.sim['d']
		logs = self.logs

		# Define operators
		ops = Operators(self.model, self.sim)
		V_odd = ops.V_odd
		V_even = ops.V_even
		V_0 = ops.V_0
		V_Lm2 = ops.V_Lm2
		a_op = ops.a
		a_dag = ops.a_dag
		n_op = ops.n_op

		# Create arrays to hold data
		# and store initial values
		if (logs['rho']):
			for r in range(0, L):
				self.rhos[r,0] = self.Single_Site_Rho(r)
		if (logs['a']):
			for r in range(0, L):
				self.a_avg[r,0] = np.trace(np.dot(self.Single_Site_Rho(r), a_op))
		if (logs['n']):
			for r in range(0, L):
				self.n_avg[r,0] = np.real(np.trace(np.dot(self.Single_Site_Rho(r), n_op)))

		# # Loop: do all the odds, then evens, then odd
		for i in range(1, N+1):
			# Evolve odd links delta * t / 2
			for h in range(0, L-2):
				if h % 2 == 1:
					self.Update(V_odd, h)
			if (L-2) % 2 == 1:
				self.Update(V_Lm2, L-2)		
			
			# Evolve even links delta * t
			self.Update(V_0, 0)
			for j in range(1, L-2):
				if j % 2 == 0:
					self.Update(V_even, j)
			if (L-2) % 2 == 0:
				self.Update(V_Lm2, L-2)	
			
			# Evolve odd links delta * t / 2
			for k in range(0, L-2):
				if k % 2 == 1:
					self.Update(V_odd, k)
			if (L-2) % 2 == 1:
				self.Update(V_Lm2, L-2)
			
			# Log data:
			if (i % (logs['skip'] + 1) == 0):
				ind = int(i / (logs['skip'] + 1))
				if (logs['rho']):
					# Store single particle density matrices
					for r in range(0, L):
						self.rhos[r,ind] = self.Single_Site_Rho(r)
				if (logs['a']):
					# Store expectation values of a
					for r in range(0, L):
						self.a_avg[r,ind] = np.trace(np.dot(self.Single_Site_Rho(r), a_op))
				if (logs['n']):
					# Store expectation values of n
					for r in range(0, L):
						self.n_avg[r,ind] = np.real(np.trace(np.dot(self.Single_Site_Rho(r), n_op)))

			# Can delete this later:
			if (i % 50 == 0):
				print "step {0} done".format(i)
	
	# Build the Theta tensor
	# This is denoted as Psi-bar in
	# Urbanek and Soldan equation 16
	def Build_Theta(self, l):
		L = self.sim['L']
		chi = self.sim['chi']
		Lambda = self.Lambda
		A_l = self.tensorA[l]
		A_lp1 = self.tensorA[l+1]

		if (l != self.sim['L']-2):
			theta = np.tensordot(A_l, A_lp1, axes=(-1,1))
			# theta = np.tensordot(theta, np.diag(Lambda[l+1]), axes=(-1, 0))
			theta = np.transpose(theta, (0,2,1,3))
		elif (l == self.sim['L']-2):
			theta = np.tensordot(A_l, A_lp1, axes=(-1,1))
			theta = np.transpose(theta, (0,2,1,3))
		return theta

	# This follows the discussion in
	# Urbanek and Soldan section 3.1
	# Note that while the paper indexes from 1,
	# we index from zero here to avoid confusion in the code.
	# Therefore, l runs from [0, ..., L - 1] in our case.
	def Update(self, V, l):
		d = self.sim['d']; L = self.sim['L'];
		chi = self.sim['chi']; delta = self.sim['delta'];

		# Build the appropriate Theta tensor
		Theta = self.Build_Theta(l)
		# Apply the unitary matrix V
		Theta = np.tensordot(V, Theta, axes=([2,3], [0,1]))
		
		# Build Phi
		if (l != L - 2):
			Phi = np.tensordot(Theta, np.diag(self.Lambda[l+1]), axes=(-1,0))
		else:
			Phi = Theta

		# Reshape to a square matrix and do singular value decomposition:
		Phi = Phi / np.linalg.norm(np.absolute(Phi))
		Phi = np.reshape(np.transpose(Phi, (0,2,1,3)), (d*chi, d*chi))
		# A and transpose.C contain the new Gamma[l] and Gamma[l+1]
		# B contains new Lambda[l]
		A, B, C = np.linalg.svd(Phi)

		# Truncate at chi eigenvalues and enforce normalization
		norm = np.linalg.norm(B[0:chi])
		self.Lambda[l] = B[0:chi] / norm
		# print (l, self.Lambda[l])

		# Keep track of the truncation error accumulated on this step
		self.tau += delta * (1 - norm**2)

		# Find the new A_l's:
		# A_l:
		A_l = np.reshape(A[:, 0:chi], (d, chi, chi))
		self.tensorA[l] = A_l

		# A_(l+1):
		# See Urbanek and Soldan eq. 19
		A_l_dag = np.transpose(np.conjugate(A_l), (0,2,1))
		A_lp1 = np.transpose(np.tensordot(A_l_dag, Theta, axes=([0,-1],[0,2])), (1,0,2))
		self.tensorA[l+1] = A_lp1

	# Calculate the reduced density matrix,
	# tracing over all sites except site k
	# See Mishmash thesis for derivation
	def Single_Site_Rho(self, k):
		L = self.sim['L']
		Lambda = self.Lambda
		A_k = self.tensorA[k]

		# Need to treat boundaries differently...
		# See Mishmash thesis pg. 73 for formulas
		# Note that this is modified since we're storing
		# A_l tensors, not Gamma_l tensors
		if (k != 0 and k != L - 1):
			Rho_L = np.tensordot(np.conjugate(A_k), np.diag(Lambda[k]), axes=(-1,0))
			Rho_R = np.tensordot(A_k, np.diag(Lambda[k,:]), axes=(-1,0))
			Rho = np.tensordot(Rho_L, Rho_R, axes=([1,-1],[1,-1]))
			Rho = np.transpose(Rho)
		elif (k == 0):
			Rho_L = np.tensordot(np.conjugate(A_k), np.diag(Lambda[k+1,:]), axes=(-1,0))
			Rho_R = np.tensordot(A_k, np.diag(Lambda[k+1,:]), axes=(-1,0))
			Rho = np.tensordot(Rho_L, Rho_R, axes=([1,-1],[1,-1]))
			Rho = np.transpose(Rho)
		elif (k == L - 1):
			Rho = np.tensordot(np.conjugate(A_k), A_k, axes=([1,2], [1,2]))
			Rho = np.transpose(Rho)
		return Rho

# Operator definitions:
class Operators(object):
	def __init__(self, model, sim):
		# Set up ladder operators	
		d = sim['d']
		L = sim['L']
		a = np.zeros((d,d), dtype=np.complex64)
		for i in range(0,d-1):
			a[i, i+1] = math.sqrt(i+1)
		a_dag = np.transpose(a)
		n_op = np.dot(a_dag, a)

		self.a = a
		self.a_dag = a_dag
		self.n_op = n_op
	
		# Set up Hamiltonian
		J = model['J']; U = model['U']; delta = sim['delta']

		# Build two site Hamiltonian:
		hop = -J * (np.kron(a_dag, a) + np.kron(a, a_dag))
		onsite = (U / 4) * np.kron(np.dot(n_op, n_op - np.identity(d)), np.identity(d)) + (U / 4) * np.kron(np.identity(d), np.dot(n_op, n_op - np.identity(d)))
		H_2site = hop + onsite
		H_0 = H_2site + (U / 4) * np.kron(np.dot(n_op, n_op - np.identity(d)), np.identity(d))
		H_Lm2 = H_2site + (U / 4) * np.kron(np.identity(d), np.dot(n_op, n_op - np.identity(d)))
		# Diagonalize 
		w,v = np.linalg.eigh(H_2site)
		# Create the (d x d x d x d) unitary operator
		self.V_odd = np.reshape(np.dot(np.dot(v,np.diag(np.exp(-1j*delta*(w) / 2))), np.transpose(v)), (d,d,d,d))
		self.V_even = np.reshape(np.dot(np.dot(v,np.diag(np.exp(-1j*delta*(w)))), np.transpose(v)), (d,d,d,d))
		# Do the same for V_0, V_{L-2}
		w0, v0 = np.linalg.eigh(H_0)
		self.V_0 = np.reshape(np.dot(np.dot(v0,np.diag(np.exp(-1j*delta*(w0)))), np.transpose(v0)), (d,d,d,d))
		w_Lm2, v_Lm2 = np.linalg.eigh(H_Lm2)
		if (L - 2) % 2 == 0:
			self.V_Lm2 = np.reshape(np.dot(np.dot(v_Lm2,np.diag(np.exp(-1j*delta*(w_Lm2)))), np.transpose(v_Lm2)), (d,d,d,d))
		else:
			self.V_Lm2 = np.reshape(np.dot(np.dot(v_Lm2,np.diag(np.exp(-1j*delta*(w_Lm2) / 2))), np.transpose(v_Lm2)), (d,d,d,d))

# Helper functions for initialization:

# Initialize state vectors (product state)
# L sites, number cutoff nmax
# n_onsite is the number on each site
# flag: 0 is Fock state, 1 is (truncated) coherent state with mean occupation = n_onsite
# flag = 0: if number_onsite is an integer, initialize n = n_onsite Mott insulator
# 		if not, initialize w/ probability (n_onsite - floor(n_onsite))
# flag = 1: approximate coherent state with alpha = sqrt(n_onsite)
# flag = 2: put floor(nbar) particles on 'site'
def Initialize_States(sim, init):
	n_max = sim['d']
	L = sim['L']
	n_onsite = init['nbar']
	flag = init['flag']
	site = init['site']

	# Initialize a matrix of zeros
	mat = np.zeros((L, n_max), dtype=np.complex64)

	# If not coherent states
	if (flag == 0):
		if (math.floor(n_onsite) >= 0):
			# Check that you're under the number cutoff
			if (math.floor(n_onsite) < n_max):
				mat[:,math.floor(n_onsite)] = 1.0
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
						ind = int(math.floor(n_onsite))
						mat[i,ind+1] = 1.0
						mat[i,ind] = 0.0
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

	# If want to initialize n_onsite particles on a given site
	elif (flag == 2):
		# Check that you're under the number cutoff
		if (math.floor(n_onsite) < n_max):
			mat[:, 0] = 1.0
			mat[site, 0] = 0.0
			mat[site,math.floor(n_onsite)] = 1.0
	return mat

# Initialize the coefficient tensors
# See Mishmash Appendix A, assuming product wavefunctions
# Important not to inadverdently cast to real numbers instead of complex!
def lambda0(sim):
	L = sim['L']; chi = sim['chi']
	# From Appendix A, pg. 173
	mat = np.zeros((L-1, chi), dtype=np.complex64)
	mat[:,0] = 1.0
	return mat

# From Appendix A, pg. 173
def Gamma0(coeffs, sim):
	d = sim['d']; L = sim['L']; chi = sim['chi']
	Gamma = []

	for i in range(0, L):
		mat = np.zeros((d, chi, chi), dtype=np.complex64)
		mat[:, 0, 0] = coeffs[i,:]
		Gamma.append(mat)
	return Gamma
