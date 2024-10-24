import subprocess
import numpy as np
from math import ceil
from hapla import shared_cy
from hapla import admix_cy

##### hapla - functions #####
### hapla struct
# Randomized PCA (PCAone Halko algorithm)
def randomizedSVD(Z, p, a, K, batch, threads):
	m = Z.shape[0]
	n = Z.shape[1]
	B = ceil(m/batch)
	L = K + 20
	O = np.random.standard_normal(size=(n, L)).astype(np.float32)
	A = np.zeros((m, L), dtype=np.float32)
	H = np.zeros((n, L), dtype=np.float32)
	for power in range(12):
		Z_bat = np.zeros((batch, n), dtype=np.float32)
		if power > 0:
			O, _ = np.linalg.qr(H, mode="reduced")
			H.fill(0.0)
		for b in range(B):
			m_b = b*batch
			if b == (B-1): # Last batch
				Z_bat = np.zeros((m - m_b, n), dtype=np.float32)
			shared_cy.batchZ(Z, Z_bat, p, a, m_b, threads)
			A[m_b:(m_b + Z_bat.shape[0])] = np.dot(Z_bat, O)
			H += np.dot(Z_bat.T, A[m_b:(m_b + Z_bat.shape[0])])
	Q, R1 = np.linalg.qr(A, mode="reduced")
	Q, R2 = np.linalg.qr(Q, mode="reduced")
	R = np.dot(R1, R2)
	C = np.linalg.solve(R.T, H.T)
	Uhat, S, V = np.linalg.svd(C, full_matrices=False)
	U = np.dot(Q, Uhat)
	del A, C, H, O, Q, R, R1, R2, Uhat, Z_bat
	U = np.ascontiguousarray(U[:,:K])
	V = np.ascontiguousarray(V[:K,:].T)
	return U, S[:K], V



### hapla admix
# Update for admixture estimation
def step(Z, P, Q, Q_tmp, k_vec, y, S, N, threads):
	admix_cy.updateP(Z, P, Q, Q_tmp, k_vec, N, threads)
	admix_cy.updateQ(Q, Q_tmp, S, threads)
	if y is not None:
		admix_cy.superQ(Q, y, N, threads)

# Accelerated update for admixture estimation
def accel(Z, P0, Q0, Q_tmp, P1, P2, Q1, Q2, k_vec, y, S, N, threads):
	# 1st EM step
	admix_cy.accelP(Z, P0, P1, Q0, Q_tmp, k_vec, N, threads)
	admix_cy.accelQ(Q0, Q1, Q_tmp, S, threads)
	if y is not None:
		admix_cy.superQ(Q1, y, N, threads)

	# 2nd EM step
	admix_cy.accelP(Z, P1, P2, Q1, Q_tmp, k_vec, N, threads)
	admix_cy.accelQ(Q1, Q2, Q_tmp, S, threads)
	if y is not None:
		admix_cy.superQ(Q2, y, N, threads)

	# Acceleation update
	admix_cy.alphaP(P0, P1, P2, k_vec, threads)
	admix_cy.alphaQ(Q0, Q1, Q2, threads)
	if y is not None:
		admix_cy.superQ(Q0, y, N, threads)
