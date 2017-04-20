import numpy as np
import math

# Cutoff for division by small numbers
cutoff = 1E-2

# Class for handling the Lambda, Gamma, and Theta tensors
class TEBD(object):

	# Initialize
	def __init__(self, model, sim, init, logs):
		self.Lambda = lambda0(sim)
		coeffs = np.kron(Initialize_States(sim, init), Initialize_Impurity(sim, init))
		self.Gamma = Gamma0(coeffs, sim)
		self.tau = 0
		self.model = model
		self.sim = sim
		self.init = init
		self.logs = logs
		self.rhos = np.zeros((sim['L'], sim['N']+1, sim['d'], sim['d']), dtype=np.complex64)
		self.a_avg = np.zeros((sim['L'], sim['N']+1), dtype=np.complex64)
		self.b_avg = np.zeros((sim['L'], sim['N']+1), dtype=np.complex64)
		self.n_a_avg = np.zeros((sim['L'], sim['N']+1))
		self.n_b_avg = np.zeros((sim['L'], sim['N']+1))
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
		V_last = ops.V_last
		a_op = ops.a
		a_dag = ops.a_dag
		b_op = ops.b
		b_dag = ops.b_dag
		n_a = ops.n_a
		n_b = ops.n_b

		# Create arrays to hold data
		# and store initial values
		if (logs['rho']):
			for r in range(0, L):
				self.rhos[r,0] = self.Single_Site_Rho(r)
		if (logs['a']):
			for r in range(0, L):
				self.a_avg[r,0] = np.trace(np.dot(self.Single_Site_Rho(r,'a'), a_op))
		if (logs['b']):
			for r in range(0, L):
				self.b_avg[r,0] = np.trace(np.dot(self.Single_Site_Rho(r,'b'), b_op))
		if (logs['na']):
			for r in range(0, L):
				self.n_a_avg[r,0] = np.real(np.trace(np.dot(self.Single_Site_Rho(r,'a'), n_a)))
		if (logs['nb']):
			for r in range(0, L):
				self.n_b_avg[r,0] = np.real(np.trace(np.dot(self.Single_Site_Rho(r,'b'), n_b)))

		# # Loop: do all the odds, then evens, then odds
		for i in range(1, N+1):
			# Evolve odd links delta * t / 2
			for h in range(0, L-1):
				if h % 2 == 1:
					self.Update(V_odd, h)
			if ((L-1) % 2 == 1):
				self.Update(V_last, L-2)	
			
			# Evolve even links delta * t
			for j in range(0, L-1):
				if j % 2 == 0:
					self.Update(V_even, j)
			if ((L-1) % 2 == 0):
				self.Update(V_last, L-2)	
			
			# Evolve odd links delta * t / 2
			for k in range(0, L-1):
				if k % 2 == 1:
					self.Update(V_odd, k)
			if ((L-1) % 2 == 1):
				self.Update(V_last, L-2)	
			
			# Log data:
			if (i % (logs['skip'] + 1) == 0):
				if (logs['rho']):
					# Store single site density matrices
					for r in range(0, L):
						self.rhos[r,i] = self.Single_Site_Rho(r)
				if (logs['a']):
					# Store expectation values of a
					for r in range(0, L):
						self.a_avg[r,i] = np.trace(np.dot(self.Single_Site_Rho(r,'a'), a_op))
				if (logs['b']):
					# Store expectation values of b
					for r in range(0, L):
						self.b_avg[r,i] = np.trace(np.dot(self.Single_Site_Rho(r,'b'), b_op))
				if (logs['na']):
					# Store expectation values of n_a
					for r in range(0, L):
						self.n_a_avg[r,i] = np.real(np.trace(np.dot(self.Single_Site_Rho(r,'a'), n_a)))
				if (logs['nb']):
					# Store expectation values of n
					for r in range(0, L):
						self.n_b_avg[r,i] = np.real(np.trace(np.dot(self.Single_Site_Rho(r,'b'), n_b)))
				
				# Only store SPDM at the very end
				# Probably only want to do this for calculating GS
				if (logs['aa'] and logs['skip'] == N - 1):
					for l in range(0, L):
						for k in range(l+1, L):
							# Calculate <a^dag_k, a_l>
							self.aa[l,k] = np.trace(np.dot(np.reshape(self.Two_Site_Rho(l, k), (d*d, d*d)), np.kron(a_op, a_dag)))
						# Diagonal terms are number expectation values
						self.aa[l,l] =np.trace(np.dot(self.Single_Site_Rho(l,'a'), n_op))

			# Can delete this later:
			if (i % 50 == 0):
				print "step {0} done".format(i)


	# Build the Theta tensor
	def Build_Theta(self, l):
		Lambda = self.Lambda
		Gamma_lp1 = self.Gamma[l+1]
		Gamma_l = self.Gamma[l]

		if (l != 0 and l != self.sim['L']-2):
			theta = np.tensordot(np.diag(Lambda[l,:]), Gamma_lp1, axes=(1,1))
			theta = np.tensordot(theta, np.diag(Lambda[l+1,:]), axes=(-1,-1))
			theta = np.tensordot(Gamma_l, theta, axes=(2,0))
			theta = np.tensordot(np.diag(Lambda[l-1,:]), theta, axes=(0, 1))

			# # Now indices are [a_(l-1), i_l, i_(l+1), a_(l+1)]
			# # Swap them to [i_l, i_(l+1), a_(l-1), a_(l+1)] to avoid confusion
			theta = np.transpose(theta, (1,2,0,3))

		# Structure of Gamma^(0) and Gamma^(L-1) are different...
		elif (l == 0):
			theta = np.tensordot(np.diag(Lambda[l,:]), Gamma_lp1, axes=(1,1))
			theta = np.tensordot(Gamma_lp1, np.diag(Lambda[l+1,:]), axes=(-1,-1))
			theta = np.tensordot(Gamma_l, theta, axes=(-1,-1))

			# Indices already [i_l, i_(l + 1), a_(l + 1)]
		elif (l == self.sim['L']-2):
			theta = np.tensordot(np.diag(Lambda[l,:]), Gamma_lp1, axes=(1,1))
			theta = np.tensordot(Gamma_l, theta, axes=(-1,0))
			theta = np.tensordot(np.diag(Lambda[l-1,:]), theta, axes=(0, 1))
			# Swap indices to be [i_l, i_(l + 1), a_(l - 1)]
			theta = np.transpose(theta, (1,2,0))
		return theta

	# This follows the discussion in
	# http://inside.mines.edu/~lcarr/theses/mishmash_thesis_2008.pdf,
	# section 3.2.3.
	# Note that while the linked thesis indexes from 1,
	# we index from zero here to avoid confusion in the code.
	# Therefore, l runs from [0, ..., L - 1] in our case,
	# instead of [1, ..., L] as shown in the thesis.
	def Update(self, V, l):
		d = self.sim['d']; L = self.sim['L'];
		chi = self.sim['chi']; delta = self.sim['delta'];

		# Build the appropriate Theta tensor
		Theta = self.Build_Theta(l)
		# Apply the unitary matrix V
		Theta = np.tensordot(V, Theta, axes=([2,3], [0,1]))
		
		# Need to treat boundary subsystems differently...
		if (l != 0 and l != L - 2):
			# Reshape to a square matrix and do singular value decomposition:
			Theta = np.reshape(np.transpose(Theta, (0,2,1,3)), (d*chi, d*chi))
			Theta *= 1 / np.linalg.norm(np.absolute(Theta))
			# A and transpose.C contain the new Gamma[l] and Gamma[l+1]
			# B contains new Lambda[l]
			A, B, C = np.linalg.svd(Theta)
			C = np.transpose(C)

			# Truncate at chi eigenvalues and enforce normalization
			norm = np.linalg.norm(B[0:chi])
			B = B / norm
			self.Lambda[l,:] = B[0:chi]

			# Keep track of the truncation error accumulated on this step
			self.tau += delta * (1 - norm**2)

			# Find the new Gammas:
			# Gamma_l:
			A = np.reshape(A[:, 0:chi], (d, chi, chi))
			Gamma_l_new = np.tensordot(A, np.diag(OneOver(self.Lambda[l-1])), axes=(1, 0))
			self.Gamma[l] = np.transpose(Gamma_l_new, (0, 2, 1))

			# Gamma_(l+1):
			C = np.reshape(C[:, 0:chi], (d, chi, chi))
			Gamma_lp1_new = np.tensordot(C, np.diag(OneOver(self.Lambda[l+1])), axes=(-1, 0))
			self.Gamma[l+1] = Gamma_lp1_new

		# The Gamma_0 tensor has one less index... need to treat slightly differently
		elif (l == 0):
			# Reshape to a square matrix and do singular value decomposition:
			Theta = np.reshape(Theta, (d, d*chi))
			Theta *= 1 / np.linalg.norm(np.absolute(Theta))
			# A and transpose.C contain the new Gamma[l] and Gamma[l+1]
			# B contains new Lambda[l]
			A, B, C = np.linalg.svd(Theta)
			C = np.transpose(C)

			# Enforce normalization
			# Don't need to truncate here because chi is bounded by
			# the dimension of the smaller subsystem, here equals d < chi
			norm = np.linalg.norm(B)
			B = B / norm
			self.Lambda[l,0:d] = B

			# Keep track of the truncation error accumulated on this step
			self.tau += delta * (1 - norm**2)

			# Find the new Gammas:
			# Gamma_l:
			# Note: can't truncate because the matrix
			# is smaller than in the non-edge cases
			# Right multiplying by Lambda_l means that we're effectively storing
			# the A tensors instead of the Gamma tensors.
			# See Quantum Gases: Finite Temperature and Non-Equilibrium Dynamics, pg. 341
			self.Gamma[l][:, 0:d] = A
			# Treat the l = 1 case normally...
			C = np.reshape(C[:, 0:chi], (d, chi, chi))
			Gamma_lp1_new = np.tensordot(C, np.diag(OneOver(self.Lambda[l+1,:])), axes=(-1, 0))
			self.Gamma[l+1] = Gamma_lp1_new

		elif (l == L - 2):
			# Reshape to a square matrix and do singular value decomposition:
			Theta = np.reshape(np.transpose(Theta, (0,2,1)), (d*chi, d))
			Theta *= 1 / np.linalg.norm(np.absolute(Theta))
			# A and transpose.C contain the new Gamma[l] and Gamma[l+1]
			# B contains new Lambda[l]
			A, B, C = np.linalg.svd(Theta)
			C = np.transpose(C)

			# Enforce normalization
			# Don't need to truncate here because chi is bounded by
			# the dimension of the smaller subsystem, here equals d < chi
			norm = np.linalg.norm(B)
			B = B / norm
			self.Lambda[l,0:d] = B

			# Keep track of the truncation error accumulated on this step
			self.tau += delta * (1 - norm**2)

			# Find the new Gammas:
			# Treat the L-2 case normally:
			A = np.reshape(A[:,0:chi], (d, chi, chi))
			# Right multiplying by Lambda_l means that we're effectively storing
			# the A tensors instead of the Gamma tensors.
			# See Quantum Gases: Finite Temperature and Non-Equilibrium Dynamics, pg. 341
			Gamma_l_new = np.tensordot(A, np.diag(OneOver(self.Lambda[l-1])), axes=(1, 0))
			self.Gamma[l] = np.transpose(Gamma_l_new, (0,2,1))

			# Gamma_(L-1):
			# Note: can't truncate because the matrix
			# is smaller than in the non-edge cases
			self.Gamma[l+1][:,0:d] = C

	# Calculate the reduced density matrix,
	# tracing over all sites except site k
	# subspace = 'a': majority species
	# subspace = 'b': minority species
	# See Mishmash thesis for derivation
	def Single_Site_Rho(self, k, subspace=0):
		L = self.sim['L']
		da = self.sim['da']
		db = self.sim['db']

		Gamma_k = self.Gamma[k]
		# Need to treat boundaries differently...
		# See Mishmash thesis pg. 73 for formulas
		if (k != 0 and k != L - 1):
			Rho_L = np.tensordot(np.diag(self.Lambda[k-1,:]), np.conjugate(Gamma_k), axes=(1,1))
			Rho_L = np.tensordot(Rho_L, np.diag(self.Lambda[k,:]), axes=(-1,0))
			Rho_R = np.tensordot(np.diag(self.Lambda[k,:]), Gamma_k, axes=(1,1))
			Rho_R = np.tensordot(Rho_R, np.diag(self.Lambda[k,:]), axes=(-1,0))
			Rho = np.tensordot(Rho_L, Rho_R, axes=([0,2],[0,2]))
			Rho = np.transpose(Rho)
		elif (k == 0):
			Rho_L = np.tensordot(np.conjugate(Gamma_k), np.diag(self.Lambda[k,:]), axes=(-1,0))
			Rho_R = np.tensordot(Gamma_k, np.diag(self.Lambda[k,:]), axes=(-1,-1))
			Rho = np.tensordot(Rho_L, Rho_R, axes=(-1,-1))
			Rho = np.transpose(Rho)
		elif (k == L - 1):
			Rho_L = np.tensordot(np.diag(self.Lambda[k-1,:]), np.conjugate(Gamma_k), axes=(-1,-1))
			Rho_R = np.tensordot(np.diag(self.Lambda[k-1,:]), Gamma_k, axes=(-1, -1))
			Rho = np.tensordot(Rho_L, Rho_R, axes=(0,0))
			Rho = np.transpose(Rho)

		if (subspace == 0):
			return Rho
		else:
			Rho = np.reshape(Rho, (da, db, da, db))
			if (subspace == 'a'):
				Rho = np.tensordot(np.identity(db), Rho, axes=([0,1], [1,3]))
			elif (subspace == 'b'):
				Rho = np.tensordot(np.identity(da), Rho, axes=([0,1], [0, 2]))
		return Rho

	# Calculate the reduced density matrix,
	# tracing over all sites except site k and l
	# See http://sourceforge.net/projects/opentebd/
	# documentation
	def Two_Site_Rho(self, l, k):
		L = self.sim['L']; chi = self.sim['chi']; d = self.sim['d']
		Lambda = self.Lambda
		Gamma = self.Gamma
		swap = 0

		# Assume l < k
		if (k < l):
			temp = k
			k = l
			l = temp
			swap = 1

		# Form tensor:	
		if (k < L - 2):
			tensor_l = np.tensordot(np.diag(Lambda[k]), Gamma[k], axes=(0,1))
			tensor_l = np.tensordot(tensor_l, np.diag(Lambda[k+1]), axes=(-1, 0))
			tensor_r = np.tensordot(np.diag(Lambda[k+1]), np.conjugate(Gamma[k]), axes=(0,-1))
			tensor_r = np.tensordot(tensor_r, np.diag(Lambda[k]), axes=(-1,0))
			tensor = np.tensordot(tensor_l, tensor_r, axes=(-1, 0))
			# Arrange indices as (ik, ik', ak-1, ak-1')
			tensor = np.transpose(tensor, (1,2,0,3))
		elif (k == L - 2):
			tensor_l = np.tensordot(np.diag(Lambda[k]), Gamma[k], axes=(0,1))
			tensor_l = np.tensordot(tensor_l, np.identity(chi), axes=(-1, 0))
			tensor_r = np.tensordot(np.identity(chi), np.conjugate(Gamma[k]), axes=(0,-1))
			tensor_r = np.tensordot(tensor_r, np.diag(Lambda[k]), axes=(-1,0))
			tensor = np.tensordot(tensor_l, tensor_r, axes=(-1, 0))
			# Arrange indices as (ik, ik', ak-1, ak-1')
			tensor = np.transpose(tensor, (1,2,0,3))
		elif (k == L - 1):
			tensor = np.outer(Gamma[k], np.conjugate(Gamma[k]))
			tensor = np.reshape(tensor, (d, chi, d, chi))
			tensor = np.transpose(tensor, (0,2,1,3))
		for i in range(0, l-k-1):
			tensor = np.tensordot(Gamma[k-1], tensor, axes=(-1,2))
			tensor = np.tensordot(np.diag(Lambda[k-1]), tensor, axes=(-1, 1))
			tensor = np.tensordot(tensor, np.conjugate(Gamma[k-1]), axes=([1,-1], [0, -1]))
			tensor = np.tensordot(tensor, np.diag(Lambda[k-1]), axes=(-1, 0))
			np.transpose(tensor, (1,2,0,3))

		if (l != 0):
			rho_ts = np.tensordot(tensor, np.conjugate(Gamma[l]), axes=(-1, -1))
			rho_ts = np.tensordot(rho_ts, np.diag(Lambda[l]), axes=(-1, 0))
			rho_ts = np.tensordot(Gamma[l], rho_ts, axes=(-1,2))
			rho_ts = np.tensordot(np.diag(Lambda[l]), rho_ts, axes=([0, 1], [1,-1]))
			# Transpose to (il, il', ik, ik')
			rho_lk = np.transpose(rho_ts, (0,3,1,2))
		else:
			rho_ts = np.tensordot(tensor, np.conjugate(Gamma[l]), axes=(-1, -1))
			rho_ts = np.tensordot(Gamma[l], rho_ts, axes=(-1,2))
			# Transpose to (il, il', ik, ik')
			rho_lk = np.transpose(rho_ts, (0,3,1,2))
		return rho_lk


# Operator definitions:
# Not really important for this to be a class
class Operators(object):
	def __init__(self, model, sim):
		# Set up ladder operators	
		da = sim['da']
		d = sim['d']
		db = sim['db']

		a = np.zeros((da,da), dtype=np.complex64)
		for i in range(0,da-1):
			a[i, i+1] = math.sqrt(i+1)
		a_dag = np.transpose(a)
		n_a = np.dot(a_dag, a)

		# Set up operators for the impurity
		b = np.zeros((db,db), dtype=np.complex64)
		for i in range(0,db-1):
			b[i,i+1] = math.sqrt(i+1)
		b_dag = np.transpose(b)
		n_b = np.dot(b_dag, b)
	
		# Set up Hamiltonian
		Ja = model['Ja']; Ua = model['Ua']; Jb = model['Jb']; Uab = model['Uab']; delta = sim['delta']

		if (sim['it']):
			mu = sim['mu'] * Ua
		else:
			mu = 0

		# Operators saved here are defined in terms of the
		# base subspaces A and B, not the tensor product!!!
		self.a = a
		self.a_dag = a_dag
		self.n_a = n_a
		self.b = b
		self.b_dag = b_dag
		self.n_b = n_b

		a = np.kron(a, np.identity(db)); a_dag = np.kron(a_dag, np.identity(db))
		b = np.kron(np.identity(da), b); b_dag = np.kron(np.identity(da), b_dag)
		n_a = np.kron(n_a, np.identity(db)); n_b = np.kron(np.identity(da), n_b)
		n_ab = np.dot(n_a, n_b)

		# Defined in the tensor product space
		self.n_ab = n_ab

		# Build two site Hamiltonian:
		J_2site = -Ja * (np.kron(a_dag, a) + np.kron(a, a_dag)) - Jb * (np.kron(b_dag, b) + np.kron(b, b_dag))
		U_2site = (Ua / 2) * np.kron(np.dot(n_a, n_a - np.identity(d)), np.identity(d)) + Uab * np.kron(n_ab, np.identity(d)) - mu * np.kron(n_a, np.identity(d))
		H_2site = J_2site + U_2site
		# Need to treat the last link differently...
		# Use as a single-site operator
		H_last = (Ua/2) * np.kron(np.identity(d), np.dot(n_a, n_a - np.identity(d))) + Uab * np.kron(np.identity(d), n_ab) - mu * np.kron(np.identity(d), n_a)
		# Diagonalize 
		w,v = np.linalg.eig(H_2site)
		# Check if we are looking at tau = it evolution to find ground state:
		if (not sim['it']):
			self.V_odd = np.reshape(np.dot(np.dot(v,np.diag(np.exp(-1j*delta*(w) / 2))), np.transpose(v)), (d,d,d,d))
			self.V_even = np.reshape(np.dot(np.dot(v,np.diag(np.exp(-1j*delta*(w)))), np.transpose(v)), (d,d,d,d))
		else:
			self.V_odd = np.reshape(np.dot(np.dot(v,np.diag(np.exp(-delta*(w) / 2))), np.transpose(v)), (d,d,d,d))
			self.V_even = np.reshape(np.dot(np.dot(v,np.diag(np.exp(-delta*(w)))), np.transpose(v)), (d,d,d,d))
		# Same thing for last site
		wp, vp = np.linalg.eig(H_last)
		if ((sim['L'] - 1) % 2 == 0):
			if (not sim['it']):
				self.V_last = np.reshape(np.dot(np.dot(vp,np.diag(np.exp(-1j*delta*(wp)))), np.transpose(vp)), (d,d,d,d))
			else:
				self.V_last = np.reshape(np.dot(np.dot(vp,np.diag(np.exp(-delta*(wp)))), np.transpose(vp)), (d,d,d,d))
		else:
			if (not sim['it']):
				self.V_last = np.reshape(np.dot(np.dot(vp,np.diag(np.exp(-1j*delta*(wp) / 2))), np.transpose(vp)), (d,d,d,d))
			else:
				self.V_last = np.reshape(np.dot(np.dot(vp,np.diag(np.exp(-delta*(wp) / 2))), np.transpose(vp)), (d,d,d,d))


# Helper functions for initialization:

# Initialize state vectors (product state)
# L sites, number cutoff nmax
# n_onsite is the number on each site
# flag: 0 is Fock state, 1 is (truncated) coherent state with mean occupation = n_onsite
# flag = 0: if number_onsite is an integer, initialize n = n_onsite Mott insulator
# 		if not, initialize w/ probability (n_onsite - floor(n_onsite))
# flag = 1: approximate coherent state with alpha = sqrt(n_onsite)
def Initialize_States(sim, init):
	n_max = sim['da']
	L = sim['L']
	n_onsite = init['nbar']
	flag = init['flag']

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

def Initialize_Impurity(sim, init):
	L = sim['L']
	k = init['site']

	# Impurity Hilbert space dimension
	db = sim['db']

	# Initialize a matrix of zeros
	mat = np.zeros((L, db), dtype=np.complex64)
	
	# Initialize impurity with probability 1 on the chosen site
	
	# If Hilbert space dimension = 1, then no impurities
	# If input -1, then no impurities for convenience
	if db == 1 or k == -1:
		mat[:,0] = np.ones(L)
		return mat
	# Otherwise initialize the impurity on the chosen site
	else:
		for i in range(0, L):
			if (i == k):
				mat[i,1] = 1
			else:
				mat[i,0] = 1
		return mat

# Initialize the coefficient tensors
# See http://inside.mines.edu/~lcarr/theses/mishmash_thesis_2008.pdf,
# Appendix A, assuming product wavefunctions
# Important not to inadverdently cast to real numbers instead of complex!
def lambda0(sim):
	L = sim['L']; chi = sim['chi']
	# From Appendix A, pg. 173
	mat = np.zeros((L-1, chi), dtype=np.complex64)
	mat[:,0] = 1
	return mat

# From Appendix A, pg. 173
def Gamma0(coeffs, sim):
	d = sim['d']; L = sim['L']; chi = sim['chi']
	Gamma = []

	# Note that the first and last Gamma tensors
	# have a different form (two indices instead of three),
	# so they need to be initialized separately
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

def OneOver(array):
	for i in range(0, len(array)):
		if np.abs(array[i]) >= cutoff:
			array[i] = 1.0 / array[i]
		else:
			array[i] = 0
	return array
