# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np

##### Cython memory efficient functions #####
# Read variant from VCF/BCF into 1-bit integer format
cpdef void readBit(unsigned char[::1] G, const short[:,::1] V, const size_t N) \
		noexcept nogil:
	cdef:
		size_t B = G.shape[0]
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
cpdef void predBit(unsigned char[::1] G, const short[:,::1] V, const size_t N) \
		noexcept nogil:
	cdef:
		size_t B = G.shape[0]
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
cpdef void convertBit(const unsigned char[:,::1] G, unsigned char[:,::1] H, \
		unsigned int[:,::1] C, unsigned int[::1] p_vec, unsigned int[::1] d_vec, \
		unsigned int[::1] a_tmp, unsigned int[::1] b_tmp, unsigned int[::1] d_tmp, \
		unsigned int[::1] e_tmp, const size_t S) noexcept nogil:
	cdef:
		size_t B = G.shape[1]
		size_t M = H.shape[0]
		size_t N = H.shape[1]
		size_t b, h, i, j, k, s, u, v, bit
		unsigned int f, l, p, q
		unsigned char mask = 1
		unsigned char g, byte
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
		p = q = j+1
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
cpdef unsigned int uniqueBit(const unsigned char[:,::1] H, unsigned char[:,::1] X, \
		const unsigned int[::1] p_vec, const unsigned int[::1] d_vec, \
		unsigned int[::1] u_vec) noexcept nogil:
	cdef:
		size_t N = X.shape[0]
		size_t M = X.shape[1]
		size_t h, i, j
		unsigned int u = 0
	for i in range(N):
		if d_vec[i] != 0:
			h = p_vec[i]
			for j in range(M):
				X[u,j] = H[j,h]
			u += 1
		u_vec[u-1] += 1
	return u

# Convert 2-bit into full array for predicting target clusters
cpdef void predictBit(const unsigned char[:,::1] G, unsigned char[:,::1] X, \
		const size_t w_s) noexcept nogil:
	cdef:
		size_t B = G.shape[1]
		size_t N = X.shape[0]
		size_t M = X.shape[1]
		size_t b, i, j, s, bit
		unsigned char[4] recode = [0, 9, 9, 1]
		unsigned char mask = 3
		unsigned char byte
	for j in range(M):
		i = 0
		s = w_s+j
		for b in range(B):
			byte = G[s,b]
			for bit in range(4):
				X[i,j] = recode[byte & mask]
				byte = byte >> 2 # Right shift 2 bit
				i += 1
				if i == N:
					break
