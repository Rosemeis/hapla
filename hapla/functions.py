import numpy as np
from math import ceil
from hapla import shared_cy
from hapla import admix_cy

##### hapla - functions #####
### hapla struct
# Randomized PCA (PCAone Halko algorithm)
def randomizedSVD(Z_agg, p_vec, a_vec, K, batch, rng):
	M = Z_agg.shape[0]
	N = Z_agg.shape[1]
	L = K + 10
	B = 64
	S = np.arange(M, dtype=np.uint32)
	H = np.zeros((N, L))
	O = rng.standard_normal(size=(N, L))

	# PCAone block power iterations
	for e in np.arange(6):
		print(f"\rEpoch {e+1}/7", end="")
		rng.shuffle(S)
		A = np.zeros((ceil(M/B), L))
		for b in np.arange(B):
			s = S[(b*A.shape[0]):min((b+1)*A.shape[0], M)]
			W = ceil(s.shape[0]/batch)
			X = np.zeros((batch, N))
			if b == (B-1):
				A = np.zeros((s.shape[0], L))
			if ((e == 0) and (b > 0)) or (e > 0):
				O, _ = np.linalg.qr(H, mode="reduced")
				H.fill(0.0)
			for w in np.arange(W):
				M_w = w*batch
				if w == (W-1): # Last batch
					X = np.zeros((s.shape[0] - M_w, N))
				shared_cy.blockZ(Z_agg, X, p_vec, a_vec, s, M_w)
				A[M_w:(M_w + X.shape[0])] = np.dot(X, O)
				H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
		B = B//2
	
	# Standard power iteration
	print("\rEpoch 7/7", end="")
	W = ceil(M/batch)
	A = np.zeros((M, L))
	X = np.zeros((batch, N))
	O, _ = np.linalg.qr(H, mode="reduced")
	H.fill(0.0)
	for w in np.arange(W):
		M_w = w*batch
		if w == (W-1): # Last batch
			X = np.zeros((M - M_w, N))
		shared_cy.batchZ(Z_agg, X, p_vec, a_vec, M_w)
		A[M_w:(M_w + X.shape[0])] = np.dot(X, O)
		H += np.dot(X.T, A[M_w:(M_w + X.shape[0])])
	Q, R1 = np.linalg.qr(A, mode="reduced")
	Q, R2 = np.linalg.qr(Q, mode="reduced")
	R = np.dot(R1, R2)
	C = np.linalg.solve(R.T, H.T)
	U_hat, S, V = np.linalg.svd(C, full_matrices=False)
	U = np.dot(Q, U_hat)
	return np.ascontiguousarray(U[:,:K]), S[:K], np.ascontiguousarray(V[:K,:].T)



### hapla admix
# Update for admixture estimation
def steps(Z, P, Q, Q_tmp, k_vec, c_vec, y, S, C):
	admix_cy.updateP(Z, P, Q, Q_tmp, k_vec, c_vec, C)
	admix_cy.updateQ(Q, Q_tmp, S)
	if y is not None:
		admix_cy.superQ(Q, y)

# Accelerated update for admixture estimation
def quasi(Z, P0, Q0, Q_tmp, P1, P2, Q1, Q2, k_vec, c_vec, y, S, C):
	# 1st EM step
	admix_cy.accelP(Z, P0, P1, Q0, Q_tmp, k_vec, c_vec, C)
	admix_cy.accelQ(Q0, Q1, Q_tmp, S)
	if y is not None:
		admix_cy.superQ(Q1, y)

	# 2nd EM step
	admix_cy.accelP(Z, P1, P2, Q1, Q_tmp, k_vec, c_vec, C)
	admix_cy.accelQ(Q1, Q2, Q_tmp, S)
	if y is not None:
		admix_cy.superQ(Q2, y)

	# Acceleation update
	admix_cy.alphaP(P0, P1, P2, k_vec, c_vec, Q0.shape[1])
	admix_cy.alphaQ(Q0, Q1, Q2)
	if y is not None:
		admix_cy.superQ(Q0, y)
