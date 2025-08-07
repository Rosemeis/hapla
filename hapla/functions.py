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
def randomizedSVD(Z_agg, p_vec, a_vec, D, chunk, power, rng):
	M, N = Z_agg.shape
	W = ceil(M/chunk)
	a = 0.0
	L = max(D + 10, 20)
	H = np.zeros((N, L), dtype=np.float32)
	X = np.zeros((chunk, N), dtype=np.float32)
	A = rng.standard_normal(size=(M, L), dtype=np.float32)

	# Prime iteration
	for w in np.arange(W):
		M_w = w*chunk
		M_e = min((w + 1)*chunk, M)
		M_x = M_e - M_w
		shared_cy.chunkZ(Z_agg[M_w:M_e], X[:M_x], p_vec[M_w:M_e], a_vec[M_w:M_e])
		H += np.dot(X[:M_x].T, A[M_w:M_e])
	Q, _, _ = eigSVD(H)
	H.fill(0.0)

	# Power iterations
	for p in np.arange(power):
		print(f"\rPower iteration {p + 1}/{power}", end="")
		for w in np.arange(W):
			M_w = w*chunk
			M_e = min((w + 1)*chunk, M)
			M_x = M_e - M_w
			shared_cy.chunkZ(Z_agg[M_w:M_e], X[:M_x], p_vec[M_w:M_e], a_vec[M_w:M_e])
			A[M_w:M_e] = np.dot(X[:M_x], Q)
			H += np.dot(X[:M_x].T, A[M_w:M_e])
		H -= a*Q
		Q, S, _ = eigSVD(H)
		H.fill(0.0)
		if S[-1] > a:
			a = 0.5*(S[-1] + a)

	# Extract singular vectors
	for w in np.arange(W):
		M_w = w*chunk
		M_e = min((w + 1)*chunk, M)
		M_x = M_e - M_w
		shared_cy.chunkZ(Z_agg[M_w:M_e], X[:M_x], p_vec[M_w:M_e], a_vec[M_w:M_e])
		A[M_w:M_e] = np.dot(X[:M_x], Q)
	U, S, V = eigSVD(A)
	return U[:,:D], S[:D], np.dot(Q, V)[:,:D]

# Memory efficient randomized PCA with dynamic shift
def memorySVD(Z, p_vec, a_vec, k_vec, c_vec, D, chunk, power, rng):
	W = Z.shape[0]
	N = Z.shape[1]//2
	M = c_vec[W]
	B = ceil(chunk/ceil(M/W))
	C = ceil(W/B)
	a = 0.0
	L = max(D + 10, 20)
	H = np.zeros((N, L), dtype=np.float32)
	X = np.zeros((np.max(k_vec[:W])*B, N), dtype=np.float32)
	A = rng.standard_normal(size=(M, L), dtype=np.float32)

	# Prime iteration
	for c in np.arange(C):
		W_b = c*B
		W_e = min((c + 1)*B, W)
		C_b = c_vec[W_b]
		C_e = c_vec[W_e]
		C_x = C_e - C_b
		shared_cy.memoryC(Z[W_b:W_e], X[:C_x], p_vec[C_b:C_e], a_vec[C_b:C_e], k_vec[W_b:W_e], c_vec[W_b:W_e])
		H += np.dot(X[:C_x].T, A[C_b:C_e])
	Q, _, _ = eigSVD(H)
	H.fill(0.0)

	# Power iterations
	for p in np.arange(power):
		print(f"\rPower iteration {p + 1}/{power}", end="")
		for c in np.arange(C):
			W_b = c*B
			W_e = min((c + 1)*B, W)
			C_b = c_vec[W_b]
			C_e = c_vec[W_e]
			C_x = C_e - C_b
			shared_cy.memoryC(Z[W_b:W_e], X[:C_x], p_vec[C_b:C_e], a_vec[C_b:C_e], k_vec[W_b:W_e], c_vec[W_b:W_e])
			A[C_b:C_e] = np.dot(X[:C_x], Q)
			H += np.dot(X[:C_x].T, A[C_b:C_e])
		H -= a*Q
		Q, S, _ = eigSVD(H)
		H.fill(0.0)
		if S[-1] > a:
			a = 0.5*(S[-1] + a)

	# Extract singular vectors
	for c in np.arange(C):
		W_b = c*B
		W_e = min((c + 1)*B, W)
		C_b = c_vec[W_b]
		C_e = c_vec[W_e]
		C_x = C_e - C_b
		shared_cy.memoryC(Z[W_b:W_e], X[:C_x], p_vec[C_b:C_e], a_vec[C_b:C_e], k_vec[W_b:W_e], c_vec[W_b:W_e])
		A[C_b:C_e] = np.dot(X[:C_x], Q)
	U, S, V = eigSVD(A)
	return U[:,:D], S[:D], np.dot(Q, V)[:,:D]



### hapla admix
# Update for ancestry estimation
def steps(Z, P, Q, Q_tmp, k_vec, c_vec, y, L):
	admix_cy.updateP(Z, P, Q, Q_tmp, k_vec, c_vec, L)
	admix_cy.updateQ(Q, Q_tmp, Z.shape[0])
	if y is not None:
		admix_cy.superQ(Q, y)

# Accelerated update for ancestry estimation
def quasi(Z, P0, Q0, Q_tmp, P1, P2, Q1, Q2, k_vec, c_vec, y, L):
	# 1st EM step
	admix_cy.accelP(Z, P0, P1, Q0, Q_tmp, k_vec, c_vec, L)
	admix_cy.accelQ(Q0, Q1, Q_tmp, Z.shape[0])
	if y is not None:
		admix_cy.superQ(Q1, y)

	# 2nd EM step
	admix_cy.accelP(Z, P1, P2, Q1, Q_tmp, k_vec, c_vec, L)
	admix_cy.accelQ(Q1, Q2, Q_tmp, Z.shape[0])
	if y is not None:
		admix_cy.superQ(Q2, y)

	# Acceleation update
	admix_cy.jumpP(P0, P1, P2, k_vec, c_vec, Q0.shape[1])
	admix_cy.jumpQ(Q0, Q1, Q2)
	if y is not None:
		admix_cy.superQ(Q0, y)

# Batch accelerated update for ancestry estimation
def batQuasi(Z, P0, Q0, Q_tmp, P1, P2, Q1, Q2, k_vec, c_vec, s_bat, y, L):
	# 1st EM step
	admix_cy.accelBatchP(Z, P0, P1, Q0, Q_tmp, k_vec, c_vec, s_bat, L)
	admix_cy.accelQ(Q0, Q1, Q_tmp, s_bat.shape[0])
	if y is not None:
		admix_cy.superQ(Q1, y)

	# 2nd EM step
	admix_cy.accelBatchP(Z, P1, P2, Q1, Q_tmp, k_vec, c_vec, s_bat, L)
	admix_cy.accelQ(Q1, Q2, Q_tmp, s_bat.shape[0])
	if y is not None:
		admix_cy.superQ(Q2, y)

	# Acceleation update
	admix_cy.jumpBatchP(P0, P1, P2, k_vec, c_vec, s_bat, Q0.shape[1])
	admix_cy.jumpQ(Q0, Q1, Q2)
	if y is not None:
		admix_cy.superQ(Q0, y)

# Randomized PCA with dynamic shift for ALS/SVD initialization
def centerSVD(Z, p_vec, k_vec, c_vec, W, K, chunk, power, rng):
	N = Z.shape[1]//2
	D = K - 1
	M = c_vec[W]
	B = ceil(chunk/ceil(M/W))
	C = ceil(W/B)
	a = 0.0
	L = max(D + 10, 20)
	H = np.zeros((N, L), dtype=np.float32)
	X = np.zeros((np.max(k_vec[:W])*B, N), dtype=np.float32)
	A = rng.standard_normal(size=(M, L), dtype=np.float32)

	# Prime iteration
	for c in np.arange(C):
		W_b = c*B
		W_e = min((c + 1)*B, W)
		C_b = c_vec[W_b]
		C_e = c_vec[W_e]
		C_x = C_e - C_b
		shared_cy.centerC(Z[W_b:W_e], X[:C_x], p_vec[C_b:C_e], k_vec[W_b:W_e], c_vec[W_b:W_e])
		H += np.dot(X[:C_x].T, A[C_b:C_e])
	Q, _, _ = eigSVD(H)
	H.fill(0.0)

	# Power iterations
	for _ in np.arange(power):
		for c in np.arange(C):
			W_b = c*B
			W_e = min((c + 1)*B, W)
			C_b = c_vec[W_b]
			C_e = c_vec[W_e]
			C_x = C_e - C_b
			shared_cy.centerC(Z[W_b:W_e], X[:C_x], p_vec[C_b:C_e], k_vec[W_b:W_e], c_vec[W_b:W_e])
			A[C_b:C_e] = np.dot(X[:C_x], Q)
			H += np.dot(X[:C_x].T, A[C_b:C_e])
		H -= a*Q
		Q, S, _ = eigSVD(H)
		H.fill(0.0)
		if S[-1] > a:
			a = 0.5*(S[-1] + a)

	# Extract singular vectors
	for c in np.arange(C):
		W_b = c*B
		W_e = min((c + 1)*B, W)
		C_b = c_vec[W_b]
		C_e = c_vec[W_e]
		C_x = C_e - C_b
		shared_cy.centerC(Z[W_b:W_e], X[:C_x], p_vec[C_b:C_e], k_vec[W_b:W_e], c_vec[W_b:W_e])
		A[C_b:C_e] = np.dot(X[:C_x], Q)
	U, S, V = eigSVD(A)
	U = np.ascontiguousarray(U[:,:D])
	S = np.ascontiguousarray(S[:D])
	V = np.ascontiguousarray(np.dot(Q, V)[:,:D])
	return U, S, V

# Alternating least square (ALS) for initializing Q and P
def factorALS(U, S, V, p_vec, k_vec, c_vec, iter, tole, rng):
	M, D = U.shape
	Y = np.ascontiguousarray(U*S)
	P = rng.random(size=(M, D + 1), dtype=np.float32)
	admix_cy.projectP(P, k_vec, c_vec)
	I = np.dot(P, np.linalg.pinv(np.dot(P.T, P)))
	Q = 0.5*np.dot(V, np.dot(Y.T, I)) + np.sum(I*p_vec.reshape(-1,1), axis=0)
	admix_cy.projectQ(Q)
	Q0 = np.copy(Q)

	# Perform ALS iterations
	for _ in range(iter):
		# Update P
		I = np.dot(Q, np.linalg.pinv(np.dot(Q.T, Q)))
		P = 0.5*np.dot(Y, np.dot(V.T, I)) + np.outer(p_vec, np.sum(I, axis=0))
		admix_cy.projectP(P, k_vec, c_vec)

		# Update Q
		I = np.dot(P, np.linalg.pinv(np.dot(P.T, P)))
		Q = 0.5*np.dot(V, np.dot(Y.T, I)) + np.sum(I*p_vec.reshape(-1,1), axis=0)
		admix_cy.projectQ(Q)

		# Check convergence
		if admix_cy.rmseQ(Q, Q0) < tole:
			break
		memoryview(Q0.ravel())[:] = memoryview(Q.ravel())
	return P.flatten().astype(float), Q.repeat(2, axis=0).astype(float)

# Project remaining clusters for ALS/SVD initialization
def centerSub(Z, S, V, p_vec, k_vec, c_vec, W_sub, chunk):
	N = Z.shape[1]//2
	D = V.shape[1]
	W = Z.shape[0] - W_sub
	A = c_vec[W_sub]
	M = c_vec[Z.shape[0]] - A
	B = ceil(chunk/ceil(M/W))
	C = ceil(W/B)
	Y = np.ascontiguousarray(V*(1.0/S))
	U = np.zeros((M, D), dtype=np.float32)
	X = np.zeros((np.max(k_vec[W_sub:])*B, N), dtype=np.float32)

	# Loop through chunks
	for c in np.arange(C):
		W_b = W_sub + c*B
		W_e = W_sub + min((c + 1)*B, W)
		C_b = c_vec[W_b]
		C_e = c_vec[W_e]
		C_x = C_e - C_b
		shared_cy.centerC(Z[W_b:W_e], X[:C_x], p_vec[C_b:C_e], k_vec[W_b:W_e], c_vec[W_b:W_e])
		U[(C_b - A):(C_e - A)] = np.dot(X[:C_x], Y)
	return U

# Least square (ALS) for subsampled P and Q followed by standard iteration
def factorSub(U_sub, U_rem, S, V, p_vec, k_vec, c_vec, W_sub, iter, tole, rng):
	# Subsampled arrays
	M, D = U_sub.shape
	Y = np.ascontiguousarray(U_sub*S)
	p_sub = p_vec[:c_vec[W_sub]]
	k_sub = k_vec[:W_sub]
	c_sub = c_vec[:W_sub]

	# Initiate P and Q
	P = rng.random(size=(M, D + 1), dtype=np.float32)
	admix_cy.projectP(P, k_sub, c_sub)
	I = np.dot(P, np.linalg.pinv(np.dot(P.T, P)))
	Q = 0.5*np.dot(V, np.dot(Y.T, I)) + np.sum(I*p_sub.reshape(-1,1), axis=0)
	admix_cy.projectQ(Q)
	Q0 = np.copy(Q)

	# Perform ALS iterations
	for _ in range(iter):
		# Update P
		I = np.dot(Q, np.linalg.pinv(np.dot(Q.T, Q)))
		P = 0.5*np.dot(Y, np.dot(V.T, I)) + np.outer(p_sub, np.sum(I, axis=0))
		admix_cy.projectP(P, k_sub, c_sub)

		# Update Q
		I = np.dot(P, np.linalg.pinv(np.dot(P.T, P)))
		Q = 0.5*np.dot(V, np.dot(Y.T, I)) + np.sum(I*p_sub.reshape(-1,1), axis=0)
		admix_cy.projectQ(Q)

		# Check convergence
		if admix_cy.rmseQ(Q, Q0) < tole:
			break
		memoryview(Q0.ravel())[:] = memoryview(Q.ravel())
	del Q0, p_sub, k_sub, c_sub

	# Perform extra full ALS iteration
	Y = np.ascontiguousarray(np.concatenate((U_sub, U_rem), axis=0)*S)
	I = np.dot(Q, np.linalg.pinv(np.dot(Q.T, Q)))
	P = 0.5*np.dot(Y, np.dot(V.T, I)) + np.outer(p_vec, np.sum(I, axis=0))
	admix_cy.projectP(P, k_vec, c_vec)
	I = np.dot(P, np.linalg.pinv(np.dot(P.T, P)))
	Q = 0.5*np.dot(V, np.dot(Y.T, I)) + np.sum(I*p_vec.reshape(-1,1), axis=0)
	admix_cy.projectQ(Q)
	return P.flatten().astype(float), Q.repeat(2, axis=0).astype(float)

# Update for ancestry estimation in projection mode
def proSteps(Z, P, Q, Q_tmp, k_vec, c_vec):
	admix_cy.stepQ(Z, P, Q, Q_tmp, k_vec, c_vec)
	admix_cy.updateQ(Q, Q_tmp, Z.shape[0])

# Accelerated update for ancestry estimation in projection mode
def proQuasi(Z, P, Q0, Q_tmp, Q1, Q2, k_vec, c_vec):
	# 1st EM step
	admix_cy.stepQ(Z, P, Q0, Q_tmp, k_vec, c_vec)
	admix_cy.accelQ(Q0, Q1, Q_tmp, Z.shape[0])

	# 2nd EM step
	admix_cy.stepQ(Z, P, Q1, Q_tmp, k_vec, c_vec)
	admix_cy.accelQ(Q1, Q2, Q_tmp, Z.shape[0])

	# Acceleation update
	admix_cy.jumpQ(Q0, Q1, Q2)

# Batch accelerated update for ancestry estimation in projection mode
def proBatch(Z, P, Q0, Q_tmp, Q1, Q2, k_vec, c_vec, s_bat):
	# 1st EM step
	admix_cy.stepBatchQ(Z, P, Q0, Q_tmp, k_vec, c_vec, s_bat)
	admix_cy.accelQ(Q0, Q1, Q_tmp, s_bat.shape[0])

	# 2nd EM step
	admix_cy.stepBatchQ(Z, P, Q1, Q_tmp, k_vec, c_vec, s_bat)
	admix_cy.accelQ(Q1, Q2, Q_tmp, s_bat.shape[0])

	# Acceleation update
	admix_cy.jumpQ(Q0, Q1, Q2)
