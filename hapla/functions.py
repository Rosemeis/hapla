import subprocess
import numpy as np
from math import ceil
from hapla import shared_cy
from hapla import admix_cy
from hapla import fatash_cy

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
	for power in range(11):
		Z_b = np.zeros((batch, n), dtype=np.float32)
		if power > 0:
			O, _ = np.linalg.qr(H, mode="reduced")
			H.fill(0.0)
		for b in range(B):
			m_b = b*batch
			if b == (B-1): # Last batch
				Z_b = np.zeros((m - m_b, n), dtype=np.float32)
			shared_cy.batchZ(Z, Z_b, p, a, m_b, threads)
			A[m_b:(m_b + Z_b.shape[0])] = np.dot(Z_b, O)
			H += np.dot(Z_b.T, A[m_b:(m_b + Z_b.shape[0])])
	Q, R = np.linalg.qr(A, mode="reduced")
	C = np.linalg.solve(R.T, H.T)
	Uhat, S, V = np.linalg.svd(C, full_matrices=False)
	U = np.dot(Q, Uhat)
	del A, C, H, O, Q, R, Uhat, Z_b
	U = np.ascontiguousarray(U[:,:K])
	V = np.ascontiguousarray(V[:K,:].T)
	return U, S[:K], V



### hapla admix
# SQUAREM update for admixture estimation
def squarem(Z, P, Q, P0, Q0, Q_new, dP1, dP2, dP3, dQ1, dQ2, dQ3, K_vec, threads):
	np.copyto(P0, P, casting="no")
	np.copyto(Q0, Q, casting="no")

	# 1st EM step
	admix_cy.accelP(Z, P, Q, Q_new, dP1, K_vec, threads)
	admix_cy.accelQ(Q, Q_new, dQ1, Z.shape[0])

	# 2nd EM step
	admix_cy.accelP(Z, P, Q, Q_new, dP2, K_vec, threads)
	admix_cy.accelQ(Q, Q_new, dQ2, Z.shape[0])

	# Acceleation update
	admix_cy.alphaP(P, P0, dP1, dP2, dP3, K_vec, threads)
	admix_cy.alphaQ(Q, Q0, dQ1, dQ2, dQ3)



### hapla fatash
# Log-likehood wrapper for SciPy optimization
def loglikeWrapper(param, *args):
	E, Q, T, A, v, i = args
	fatash_cy.calcTransition(T, Q, i//2, param)
	return fatash_cy.loglikeFatash(E, Q, T, A, v, i)



### Phenotype generation
# PLINK help function for PCA and phenotype generation
def extract_length(filename):
	process = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE)
	result, _ = process.communicate()
	return int(result.split()[0])
