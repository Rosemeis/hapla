import subprocess
import numpy as np
from math import ceil
from hapla import shared_cy

##### hapla - functions #####
### hapla struct
# Randomized PCA (PCAone Halko algorithm)
def randomizedSVD(Z, p, a, K, batch, threads):
	m = Z.shape[0]
	n = Z.shape[1]
	B = ceil(m/batch)
	L = K + 16
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
	Q, R1 = np.linalg.qr(A, mode="reduced")
	Q, R2 = np.linalg.qr(Q, mode="reduced")
	R = np.dot(R1, R2)
	C = np.linalg.solve(R.T, H.T)
	Uhat, S, V = np.linalg.svd(C, full_matrices=False)
	U = np.dot(Q, Uhat)
	del A, C, H, O, Q, R, Uhat, Z_b
	U = np.ascontiguousarray(U[:,:K])
	V = np.ascontiguousarray(V[:K,:].T)
	return U, S[:K], V



### Phenotype generation
# PLINK help function for PCA and phenotype generation
def extract_length(filename):
	process = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE)
	result, _ = process.communicate()
	return int(result.split()[0])
