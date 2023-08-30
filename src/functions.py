import subprocess
import numpy as np
from math import ceil
from src import shared_cy

##### hapla - functions #####
### hapla pca
# Randomized PCA (PCAone Halko algorithm)
def randomizedSVD(Z_tilde, p, s, K, batch, threads):
	m = Z_tilde.shape[0]
	n = Z_tilde.shape[1]
	B = ceil(m/batch)
	L = K + 20
	O = np.random.standard_normal(size=(n, L))
	A = np.zeros((m, L), dtype=np.float32)
	H = np.zeros((n, L), dtype=np.float32)
	for p in range(11):
		Z_b = np.zeros((batch, n))
		if p > 0:
			O, _ = np.linalg.qr(H, mode="reduced")
			H.fill(0.0)
		for b in range(B):
			m_b = b*batch
			if (m_b + batch) >= m: # Last batch
				Z_b = np.zeros((m - m_b, n), dtype=np.float32)
			shared_cy.batchZ(Z_tilde, Z_b, p, s, m_b, threads)
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


### hapla regress
# Fast SVD function
def fastSVD(A):
	if A.shape[0] > A.shape[1]:
		X = np.dot(A.T, A)
		trans = True
	else:
		X = np.dot(A, A.T)
		trans = False
	D, V = np.linalg.eigh(X)
	D, V = D[D > 1e-8], V[:, D > 1e-8]
	S = np.sqrt(D)
	if trans:
		U = np.dot(A, V)/S
		return U, S, V
	else:
		U = np.dot(A.T, V)/S
		return V, S, U



### Phenotype generation
def extract_length(filename):
	process = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE)
	result, _ = process.communicate()
	return int(result.split()[0])
