# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
from libc.stdint cimport uint8_t, int16_t, uint32_t

##### Cython memory efficient functions #####
# Read variant from VCF/BCF into 1-bit integer format
cpdef void readBit(
		uint8_t[::1] G, const int16_t[:,::1] V, const uint32_t N
	) noexcept nogil:
	cdef:
		uint32_t B = G.shape[0]
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
		uint8_t[::1] G, const int16_t[:,::1] V, const uint32_t N
	) noexcept nogil:
	cdef:
		uint32_t B = G.shape[0]
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
		const uint8_t[:,::1] G, uint8_t[:,::1] H, uint32_t[:,::1] C, uint32_t[::1] p_vec, uint32_t[::1] d_vec, \
		uint32_t[::1] a_tmp, uint32_t[::1] b_tmp, uint32_t[::1] d_tmp, uint32_t[::1] e_tmp, const size_t S
	) noexcept nogil:
	cdef:
		uint8_t mask = 1
		uint8_t g, byte
		uint32_t B = G.shape[1]
		uint32_t M = H.shape[0]
		uint32_t N = H.shape[1]
		uint32_t f, l, p, q
		size_t b, h, i, j, k, s, u, v, bit
	# Populate haplotype matrix and cluster mean
	for j in range(M):
		h = 0
		s = S+j
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
		p = q = <uint32_t>j + 1
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
			p_vec[u+k] = b_tmp[k]
			d_vec[u+k] = e_tmp[k]

# Extract unique haplotypes from suffix arrays
cpdef uint32_t uniqueBit(
		const uint8_t[:,::1] H, uint8_t[:,::1] X, const uint32_t[::1] p_vec, const uint32_t[::1] d_vec, 
		uint32_t[::1] u_vec
	) noexcept nogil:
	cdef:
		uint32_t N = X.shape[0]
		uint32_t M = X.shape[1]
		size_t u = 0
		size_t h, i, j
	for i in range(N):
		if d_vec[i] != 0:
			h = <size_t>p_vec[i]
			for j in range(M):
				X[u,j] = H[j,h]
			u += 1
		u_vec[u-1] += 1
	return u

# Convert 2-bit into full array for predicting target clusters
cpdef void predictBit(
		const uint8_t[:,::1] G, uint8_t[:,::1] X, const size_t m
	) noexcept nogil:
	cdef:
		uint8_t[4] recode = [0, 9, 9, 1]
		uint8_t mask = 3
		uint8_t i, byte
		uint32_t B = G.shape[1]
		uint32_t N = X.shape[0]
		uint32_t M = X.shape[1]
		size_t b, j, s, bit
	for j in range(M):
		i = 0
		s = m+j
		for b in range(B):
			byte = G[s,b]
			for bit in range(4):
				X[i,j] = recode[byte & mask]
				byte = byte >> 2 # Right shift 2 bits
				i += 1
				if i == N:
					break
