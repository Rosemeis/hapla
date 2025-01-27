import numpy as np
from math import ceil
from hapla import shared_cy
from hapla import admix_cy

##### hapla - functions #####
### hapla struct
# SVD through eigendecomposition
def eigSVD(H):
	D, V = np.linalg.eigh(np.dot(H.T, H))
	S = np.sqrt(D)
	U = np.dot(H, V*(1.0/S))
	return np.ascontiguousarray(U[:,::-1]), np.ascontiguousarray(S[::-1]), \
		np.ascontiguousarray(V[:,::-1])

# Randomized PCA with dynamic shift
def randomizedSVD(Z_agg, p_vec, a_vec, K, batch, power, rng):
	M, N = Z_agg.shape
	W = ceil(M/batch)
	a = 0.0
	L = K + 10
	H = np.zeros((N, L))
	X = np.zeros((batch, N))
	A = rng.standard_normal(size=(M, L))

	# Prime iteration
	for w in np.arange(W):
		M_w = w*batch
		if w == (W-1): # Last batch
			X = np.zeros((M - M_w, N))
		shared_cy.batchZ(Z_agg, X, p_vec, a_vec, M_w)
		H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
	Q, _, _ = eigSVD(H)
	H.fill(0.0)

	# Power iterations
	for p in np.arange(power):
		print(f"\rPower iteration {p+1}/{power}", end="")
		X = np.zeros((batch, N))
		for w in np.arange(W):
			M_w = w*batch
			if w == (W-1): # Last batch
				X = np.zeros((M - M_w, N))
			shared_cy.batchZ(Z_agg, X, p_vec, a_vec, M_w)
			A[M_w:(M_w + X.shape[0])] = np.dot(X, Q)
			H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
		H -= a*Q
		Q, S, _ = eigSVD(H)
		H.fill(0.0)
		if S[-1] > a:
			a = 0.5*(S[-1] + a)

	# Extract singular vectors
	X = np.zeros((batch, N))
	for w in np.arange(W):
		M_w = w*batch
		if w == (W-1): # Last batch
			X = np.zeros((M - M_w, N))
		shared_cy.batchZ(Z_agg, X, p_vec, a_vec, M_w)
		A[M_w:(M_w + X.shape[0])] = np.dot(X, Q)
	U, S, V = eigSVD(A)
	return U[:,:K], S[:K], np.dot(Q, V)[:,:K]



### hapla admix
# Update for admixture estimation
def steps(Z, P, Q, P_tmp, Q_tmp, k_vec, c_vec, y):
	admix_cy.updateP(Z, P, Q, P_tmp, Q_tmp, k_vec, c_vec)
	admix_cy.updateQ(Q, Q_tmp, Z.shape[0])
	if y is not None:
		admix_cy.superQ(Q, y)

# Accelerated update for admixture estimation
def quasi(Z, P0, Q0, P_tmp, Q_tmp, P1, P2, Q1, Q2, k_vec, c_vec, y):
	# 1st EM step
	admix_cy.accelP(Z, P0, P1, Q0, P_tmp, Q_tmp, k_vec, c_vec)
	admix_cy.accelQ(Q0, Q1, Q_tmp, Z.shape[0])
	if y is not None:
		admix_cy.superQ(Q1, y)

	# 2nd EM step
	admix_cy.accelP(Z, P1, P2, Q1, P_tmp, Q_tmp, k_vec, c_vec)
	admix_cy.accelQ(Q1, Q2, Q_tmp, Z.shape[0])
	if y is not None:
		admix_cy.superQ(Q2, y)

	# Acceleation update
	admix_cy.alphaP(P0, P1, P2, k_vec, c_vec, Q0.shape[1])
	admix_cy.alphaQ(Q0, Q1, Q2)
	if y is not None:
		admix_cy.superQ(Q0, y)
