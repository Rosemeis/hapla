# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
from libc.stdint cimport uint8_t, int16_t, uint32_t

ctypedef uint8_t u8
ctypedef uint32_t u32
ctypedef int16_t i16


##### Cython memory efficient functions #####
# Read variant from VCF/BCF into 1-bit integer format
cpdef void readBit(
		u8[::1] G, const i16[:,::1] V, const Py_ssize_t N
	) noexcept nogil:
	cdef:
		Py_ssize_t B = G.shape[0]
		size_t i = 0
		size_t b, bit
	for b in range(B):
		for bit in range(0, 8, 2):
			if V[i,0] == 1:
				G[b] |= (1<<bit)
			if V[i,1] == 1:
				G[b] |= (1<<(bit+1))
			
			# Increase counter and check for break
			i += 1
			if i == N:
				break

# Read variant from VCF/BCF into 2-bit integer format
cpdef void predBit(
		u8[::1] G, const i16[:,::1] V, const Py_ssize_t N
	) noexcept nogil:
	cdef:
		Py_ssize_t B = G.shape[0]
		size_t i = 0
		size_t b, bit
	for b in range(B):
		for bit in range(0, 8, 4):
			if V[i,0] == 1: # Allele 1 (1,1)
				G[b] |= (1<<bit)
				G[b] |= (1<<(bit+1))
			elif V[i,0] == -1: # Missing (1,0)
				G[b] |= (1<<bit)
			if V[i,1] == 1: # Allele 2 (1,1)
				G[b] |= (1<<(bit+2))
				G[b] |= (1<<(bit+3))
			elif V[i,1] == -1: # Missing (1,0)
				G[b] |= (1<<(bit+2))
			
			# Increase counter and check for break
			i += 1
			if i == N:
				break

# Convert 1-bit into full array and initialize cluster mean
cpdef void convertBit(
		const u8[:,::1] G, u8[:,::1] H, u32[:,::1] C, u32[::1] p_vec, u32[::1] d_vec, u32[::1] a_tmp, u32[::1] b_tmp, 
		u32[::1] d_tmp, u32[::1] e_tmp, const u32 S
	) noexcept nogil:
	cdef:
		Py_ssize_t B = G.shape[1]
		Py_ssize_t M = H.shape[0]
		Py_ssize_t N = H.shape[1]
		size_t b, h, i, j, k, s, u, v, bit
		u8 mask = 1
		u8 g, byte
		u32 f, l, p, q
	# Populate haplotype matrix and cluster mean
	for j in range(M):
		h = 0
		s = S + j
		C[0,j] = 0
		for b in range(B):
			byte = G[s,b]
			for bit in range(8):
				g = (byte >> bit) & mask
				H[j,h] = g
				C[0,j] += g
				h += 1
				if h == N:
					break
	
		# Populate suffix arrays
		u = v = 0
		p = q = j + 1
		for i in range(N):
			f = p_vec[i]
			l = d_vec[i]
			if l > p:
				p = l
			if l > q:
				q = l
			if H[j,f] == 0:
				a_tmp[u] = f
				d_tmp[u] = p
				u += 1
				p = 0
			else:
				b_tmp[v] = f
				e_tmp[v] = q
				v += 1
				q = 0
		for k in range(u):
			p_vec[k] = a_tmp[k]
			d_vec[k] = d_tmp[k]
		for k in range(v):
			p_vec[u + k] = b_tmp[k]
			d_vec[u + k] = e_tmp[k]

# Extract unique haplotypes from suffix arrays
cpdef u32 uniqueBit(
		const u8[:,::1] H, u8[:,::1] X, const u32[::1] p_vec, const u32[::1] d_vec, u32[::1] u_vec
	) noexcept nogil:
	cdef:
		Py_ssize_t N = X.shape[0]
		Py_ssize_t M = X.shape[1]
		size_t u = 0
		size_t h, i, j
	for i in range(N):
		if d_vec[i] != 0:
			h = p_vec[i]
			for j in range(M):
				X[u,j] = H[j,h]
			u += 1
		u_vec[u - 1] += 1
	return u

# Convert 2-bit into full array for predicting target clusters
cpdef void predictBit(
		const u8[:,::1] G, u8[:,::1] X, const u32 S
	) noexcept nogil:
	cdef:
		Py_ssize_t B = G.shape[1]
		Py_ssize_t N = X.shape[0]
		Py_ssize_t M = X.shape[1]
		size_t b, j, s, bit
		u8[4] recode = [0, 9, 9, 1]
		u8 mask = 3
		u8 i, byte
	for j in range(M):
		i = 0
		s = S + j
		for b in range(B):
			byte = G[s,b]
			for bit in range(4):
				X[i,j] = recode[byte & mask]
				byte = byte >> 2 # Right shift 2 bits
				i += 1
				if i == N:
					break
