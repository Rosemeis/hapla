# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
# cython: cdivision=True
"""Exact weighted binary medians with packed distances and incremental updates."""

import math
import numpy as np

from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libc.math cimport log

ctypedef uint8_t u8
ctypedef uint32_t u32
ctypedef uint64_t u64

ctypedef fused words:
    u8
    u64

cdef extern from *:
    """
    static inline unsigned hapla_popcount(unsigned long long value) {
        return (unsigned)__builtin_popcountll(value);
    }
    """
    unsigned hapla_popcount(unsigned long long) noexcept nogil


### Count differing packed bits, stopping above the best distance
cdef inline u32 distance(words mode, const u64* a, const u64* b, Py_ssize_t count,
                         u32 best) noexcept nogil:
    cdef Py_ssize_t q
    cdef u32 d = 0
    if words is u8:
        return hapla_popcount(a[0] ^ b[0])
    for q in range(count):
        d += hapla_popcount(a[q] ^ b[q])
        if d > best:
            break
    return d


### Stably sort observed indices by reversed-SNP order
cdef void radix_order(const u64[:, ::1] X_pack, u32[::1] order,
                      u32[::1] tmp, Py_ssize_t bits) noexcept nogil:
    cdef Py_ssize_t cnt[256]
    cdef Py_ssize_t off[256]
    cdef Py_ssize_t i, bit, q, shift, bucket, total, count, H = order.shape[0]
    cdef u32* src = &order[0]
    cdef u32* dst = &tmp[0]
    cdef u32* swap
    for bit in range(0, bits, 8):
        q, shift = bit >> 6, bit & 63
        for bucket in range(256):
            cnt[bucket] = 0
        for i in range(H):
            cnt[(X_pack[src[i], q] >> shift) & 255] += 1
        # Constant digits cannot change the ordering.
        if cnt[(X_pack[src[0], q] >> shift) & 255] == H:
            continue
        total = 0
        for bucket in range(256):
            count = cnt[bucket]
            off[bucket] = total
            total += count
        for i in range(H):
            bucket = (X_pack[src[i], q] >> shift) & 255
            dst[off[bucket]] = src[i]
            off[bucket] += 1
        swap, src, dst = src, dst, src
    if src != &order[0]:
        for i in range(H):
            order[i] = src[i]


### Move weighted allele counts between clusters
cdef void move_counts(const u64* x, u64[:, ::1] C, u64[::1] n,
                      int old, int dst, u64 weight, Py_ssize_t B) noexcept nogil:
    cdef Py_ssize_t j
    cdef u64 value
    n[old] -= weight
    n[dst] += weight
    for j in range(B):
        value = weight * ((x[j >> 6] >> (j & 63)) & 1)
        C[old, j] -= value
        C[dst, j] += value


### Resolve growth ties by canonical reversed-SNP order, independently of deduplication
cdef inline bint precedes(const u64* a, const u64* b, Py_ssize_t Q) noexcept nogil:
    cdef Py_ssize_t q
    for q in range(Q - 1, -1, -1):
        if a[q] != b[q]:
            return a[q] < b[q]
    return False


### Reuse unchanged distances and update only moved haplotypes
cdef u64 assign(words mode, const u64[:, ::1] X, const u64[::1] w_vec, u64[:, ::1] R,
                u64[:, ::1] R_old, u8[::1] a_old, u8[::1] active,
                u32[::1] c_idx, u8[::1] dirty, u32[::1] z, u32[::1] d_vec,
                u64[:, ::1] C, u64[::1] n, int slots, Py_ssize_t B,
                bint cached) noexcept nogil:
    cdef Py_ssize_t i, q, Q = X.shape[1], U = X.shape[0]
    cdef int k, index, changed = 0, dst, old
    cdef u32 best, d
    cdef bint full
    cdef u64 pairs = 0
    for k in range(slots):
        dirty[k] = active[k] and not a_old[k]
        if active[k]:
            for q in range(Q):
                dirty[k] |= R[k, q] != R_old[k, q]
                R_old[k, q] = R[k, q]
            if dirty[k]:
                c_idx[changed] = k
                changed += 1
        a_old[k] = active[k]
    for i in range(U):
        old = z[i]
        full = not cached or not active[old] or dirty[old]
        dst = 0 if full else old
        best = B + 1 if full else d_vec[i]
        for index in range(slots if full else changed):
            k = index if full else c_idx[index]
            if active[k]:
                d = distance(mode, &X[i, 0], &R[k, 0], Q, best)
                pairs += 1
                if d < best or (d == best and k > dst):
                    best, dst = d, k
        if old != dst:
            move_counts(&X[i, 0], C, n, old, dst, w_vec[i], B)
            z[i] = dst
        d_vec[i] = best
    return pairs


### Update strict-majority medians and retire empty clusters
cdef bint medians(u64[:, ::1] R, const u64[:, ::1] C, const u64[::1] n,
                  u8[::1] active, int slots, Py_ssize_t B) noexcept nogil:
    cdef int k
    cdef Py_ssize_t j, q, stop, Q = R.shape[1]
    cdef u64 word
    cdef bint changed = False
    for k in range(slots):
        if active[k] and n[k] == 0:
            active[k] = 0
            changed = True
        if active[k]:
            for q in range(Q):
                word = 0
                stop = min(B, (q + 1) * 64)
                for j in range(q * 64, stop):
                    if C[k, j] > (n[k] >> 1):
                        word |= <u64>1 << (j & 63)
                changed |= R[k, q] != word
                R[k, q] = word
    return changed


### Deduplicate observed haplotypes, grow medians, and prune to convergence
def fit_window(const u8[:, ::1] G, double alpha=0.1, double min_freq=0.005,
               min_mac=None, int K_max=255, int n_iter=1000,
               bint missing=True):
    # Complete windows skip missingness checks inside the clustering kernel
    cdef Py_ssize_t B = G.shape[0], H = G.shape[1], Q = (B + 63) >> 6
    cdef Py_ssize_t i, j, q, h, first, prev = -1, U = 0, H_obs, cand
    cdef int k, slot, it, k_min, n_grow = 0, n_prune = 0
    cdef u32 d_lim, d_max, value
    cdef u64 n_min, total, w_max, pairs = 0, loss = 0
    cdef bint equal, bad = False, full = True, born, changed, capped = False
    cdef bint g_done = False, p_done = False, recode = False
    if B <= 0 or H <= 0 or B >= 2**32 - 1 or H >= 2**32:
        raise ValueError("A window must have positive dimensions below uint32 limits")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if not math.isfinite(min_freq) or not 0 < min_freq < 1:
        raise ValueError("min_freq must lie strictly between 0 and 1")
    if not 1 <= K_max <= 255 or n_iter < 1:
        raise ValueError("Use 1..255 clusters and a positive iteration limit")
    if min_mac is not None and (not isinstance(min_mac, (int, np.integer)) or min_mac < 1):
        raise ValueError("min_mac must be a positive integer")

    # Pack variant-major GT once, reserving byte 255 for missing haplotypes
    cdef u64[:, ::1] X_pack = np.zeros((H, Q), dtype=np.uint64)
    cdef u8[::1] valid = np.ones(H if missing else 0, dtype=np.uint8)
    with nogil:
        if missing:
            for j in range(B):
                for h in range(H):
                    value = G[j, h]
                    if value == 255:
                        valid[h] = 0
                        full = False
                    elif value <= 1:
                        X_pack[h, j >> 6] |= <u64>value << (j & 63)
                    else:
                        bad = True
        else:
            for j in range(B):
                for h in range(H):
                    value = G[j, h]
                    if value > 1:
                        bad = True
                    X_pack[h, j >> 6] |= <u64>(value & 1) << (j & 63)
    if bad:
        raise ValueError("Expected binary GT (or 255 for missing). Complete-data flag must be valid")
    cdef u8[::1] labels = np.full(H, 255, dtype=np.uint8)
    H_obs = H if full else int(np.count_nonzero(np.asarray(valid)))
    if H_obs == 0:
        return dict(labels=np.asarray(labels), medians=np.empty((0, B), dtype=np.uint8),
                    counts=np.empty((0, B), dtype=np.uint64), sizes=np.empty(0, dtype=np.uint64),
                    stats=dict(unique=0, observed=0, missing=H, K=0, growth_passes=0,
                               prune_passes=0, distance_pairs=0, distortion=0, capped=False))
    n_min = int(min_mac) if min_mac is not None else math.ceil(H_obs * min_freq)
    if n_min > <u64>H_obs:
        raise ValueError("Minimum cluster count exceeds the observed haplotypes in this window")
    # Collapse identical haplotypes and retain an inverse map for output
    cdef u32[::1] order = np.empty(H_obs, dtype=np.uint32)
    cdef u32[::1] tmp = np.empty(H_obs, dtype=np.uint32)
    cdef u64[:, ::1] X = np.empty((H_obs, Q), dtype=np.uint64)
    cdef u64[::1] w_vec = np.zeros(H_obs, dtype=np.uint64)
    cdef u32[::1] inverse = np.empty(H, dtype=np.uint32)
    with nogil:
        i = 0
        for h in range(H):
            if full or valid[h]:
                order[i] = h
                i += 1
        first = order[0]
        radix_order(X_pack, order, tmp, B)
        for i in range(H_obs):
            h = order[i]
            equal = prev >= 0
            if equal:
                for q in range(Q):
                    if X_pack[h, q] != X_pack[prev, q]:
                        equal = False
                        break
            if not equal:
                for q in range(Q):
                    X[U, q] = X_pack[h, q]
                U += 1
            w_vec[U - 1] += 1
            inverse[h] = U - 1
            prev = h
    X = X[:U]
    w_vec = w_vec[:U]
    # Drop packing/sort temporaries before allocating iterative working state.
    X_pack = None
    order = tmp = None

    # Maintain counts and cached assignments across growth and pruning
    cdef u64[:, ::1] R = np.zeros((K_max, Q), dtype=np.uint64)
    cdef u64[:, ::1] R_old = np.zeros((K_max, Q), dtype=np.uint64)
    cdef u64[:, ::1] C = np.zeros((K_max, B), dtype=np.uint64)
    cdef u64[::1] n = np.zeros(K_max, dtype=np.uint64)
    cdef u8[::1] active = np.zeros(K_max, dtype=np.uint8)
    cdef u8[::1] a_old = np.zeros(K_max, dtype=np.uint8)
    cdef u8[::1] dirty = np.zeros(K_max, dtype=np.uint8)
    cdef u32[::1] c_idx = np.empty(K_max, dtype=np.uint32)
    cdef u32[::1] z = np.zeros(U, dtype=np.uint32)
    cdef u32[::1] d_vec = np.zeros(U, dtype=np.uint32)
    cdef u8[::1] remap = np.zeros(K_max, dtype=np.uint8)
    cdef u64[::1] flip = np.zeros(Q, dtype=np.uint64)
    d_lim = math.ceil(alpha * B)
    slot = 1
    active[0] = 1
    n[0] = H_obs
    with nogil:
        for i in range(U):
            for j in range(B):
                C[0, j] += w_vec[i] * ((X[i, j >> 6] >> (j & 63)) & 1)
        # Orient the major allele as zero. Balanced sites follow the first complete haplotype.
        for j in range(B):
            if 2*C[0, j] > <u64>H_obs or (2*C[0, j] == <u64>H_obs and G[j, first]):
                flip[j >> 6] |= <u64>1 << (j & 63)
                C[0, j] = H_obs - C[0, j]
                recode = True
        if recode:
            for i in range(U):
                for q in range(Q):
                    X[i, q] ^= flip[q]
        # The initial strict-majority median is now zero at every site.
        for it in range(n_iter):
            if Q == 1:
                pairs += assign[u8](0, X, w_vec, R, R_old, a_old, active, c_idx,
                                     dirty, z, d_vec, C, n, slot, B, it > 0)
            else:
                pairs += assign[u64](0, X, w_vec, R, R_old, a_old, active, c_idx,
                                      dirty, z, d_vec, C, n, slot, B, it > 0)
            n_grow += 1
            cand = 0
            d_max, w_max = d_vec[0], w_vec[0]
            for i in range(1, U):
                if d_vec[i] > d_max or (d_vec[i] == d_max and
                        (w_vec[i] > w_max or (w_vec[i] == w_max and recode and
                         precedes(&X[i, 0], &X[cand, 0], Q)))):
                    cand = i
                    d_max, w_max = d_vec[i], w_vec[i]
            born = d_max >= d_lim and slot < K_max
            if born:
                move_counts(&X[cand, 0], C, n, z[cand], slot, w_vec[cand], B)
                z[cand] = slot
                d_vec[cand] = 0
                active[slot] = 1
                for q in range(Q):
                    R[slot, q] = X[cand, q]
                slot += 1
            elif d_max >= d_lim:
                capped = True
            changed = medians(R, C, n, active, slot, B)
            if not born and not changed:
                g_done = True
                break

        if g_done:
            for it in range(n_iter):
                k_min = -1
                total = H_obs + 1
                for k in range(slot):
                    if active[k] and n[k] <= total:
                        total, k_min = n[k], k
                if total < n_min:
                    active[k_min] = 0
                # A converged growth state with feasible sizes needs no pruning pass.
                elif it == 0:
                    p_done = True
                    break
                if Q == 1:
                    pairs += assign[u8](0, X, w_vec, R, R_old, a_old, active, c_idx,
                                         dirty, z, d_vec, C, n, slot, B, True)
                else:
                    pairs += assign[u64](0, X, w_vec, R, R_old, a_old, active, c_idx,
                                          dirty, z, d_vec, C, n, slot, B, True)
                n_prune += 1
                changed = medians(R, C, n, active, slot, B)
                total = H_obs + 1
                for k in range(slot):
                    if active[k] and n[k] < total:
                        total = n[k]
                if total >= n_min and not changed:
                    p_done = True
                    break
    if not g_done:
        raise RuntimeError(f"Growth did not converge within {n_iter} iterations")
    if not p_done:
        raise RuntimeError(f"Pruning did not converge within {n_iter} iterations")
    keep = np.flatnonzero(np.asarray(active))
    cdef Py_ssize_t K = len(keep)
    cdef u8[:, ::1] R_out = np.empty((K, B), dtype=np.uint8)
    cdef const Py_ssize_t[::1] kept = keep
    with nogil:
        for i in range(K):
            k = kept[i]
            remap[k] = i
            for j in range(B):
                # Export medians and counts in the input REF/ALT coding.
                value = (flip[j >> 6] >> (j & 63)) & 1
                R_out[i, j] = ((R[k, j >> 6] >> (j & 63)) & 1) ^ value
                loss += min(C[k, j], n[k] - C[k, j])
                if value:
                    C[k, j] = n[k] - C[k, j]
        for h in range(H):
            if full or valid[h]:
                labels[h] = remap[z[inverse[h]]]
    return dict(labels=np.asarray(labels), medians=np.asarray(R_out),
                counts=np.asarray(C)[keep], sizes=np.asarray(n)[keep],
                stats=dict(unique=U, observed=H_obs, missing=H-H_obs, K=K,
                           growth_passes=n_grow, prune_passes=n_prune,
                           distance_pairs=int(pairs), distortion=int(loss), capped=bool(capped)))


### Score medians under clipped within-cluster allele frequencies
def likelihoods(const u8[:, ::1] R, const u64[:, ::1] C, const u64[::1] n):
    """Bernoulli scores from final, valid counts, with no threaded BLAS calls."""
    cdef Py_ssize_t K = R.shape[0], B = R.shape[1], a, b, j
    cdef double p, score
    if C.shape[0] != K or C.shape[1] != B or n.shape[0] != K:
        raise ValueError("Median and count dimensions differ")
    cdef double[:, ::1] zero = np.empty((K, B), dtype=np.float64)
    cdef double[:, ::1] one = np.empty((K, B), dtype=np.float64)
    cdef float[:, ::1] result = np.empty((K, K), dtype=np.float32)
    for a in range(K):
        if n[a] == 0:
            raise ValueError("Cannot score an empty cluster")
        for j in range(B):
            if C[a, j] > n[a] or R[a, j] > 1:
                raise ValueError("Invalid median or allele count")
    with nogil:
        for a in range(K):
            for j in range(B):
                # Compute complementary logs from the same minor count under either coding.
                p = max(1e-5, <double>min(C[a, j], n[a] - C[a, j]) / <double>n[a])
                one[a, j], zero[a, j] = log(p), log(1.0 - p)
                if C[a, j] > (n[a] >> 1):
                    one[a, j], zero[a, j] = zero[a, j], one[a, j]
        for a in range(K):
            for b in range(K):
                score = 0.0
                for j in range(B):
                    score += one[b, j] if R[a, j] else zero[b, j]
                result[a, b] = score
    return np.asarray(result)


### Encode diploid cluster dosages as SNP-major PLINK rows
def plink_window(const u8[::1] labels, int K):
    """Encode one window in SNP-major BED order. Either missing haplotype => ./.."""
    cdef Py_ssize_t H = labels.shape[0], N = H >> 1, i, k
    cdef u8 a, b, dosage, code
    if H % 2 or not 0 <= K <= 255:
        raise ValueError("PLINK output requires diploid samples and 0..255 clusters")
    for i in range(H):
        if labels[i] != 255 and labels[i] >= K:
            raise ValueError("Invalid haplotype cluster assignment")
    cdef u8[:, ::1] result = np.zeros((K, (N + 3) >> 2), dtype=np.uint8)
    with nogil:
        for k in range(K):
            for i in range(N):
                a, b = labels[2*i], labels[2*i+1]
                if a == 255 or b == 255:
                    code = 1
                else:
                    dosage = (a == k) + (b == k)
                    code = 3 if dosage == 0 else 2 if dosage == 1 else 0
                result[k, i >> 2] |= code << ((i & 3) * 2)
    return np.asarray(result)


### Compile single-word and multiword scans without a branch inside the distance loop
cdef void nearest(words mode, const u64[:, ::1] X, const u64[:, ::1] R,
                   const u8[::1] valid, u8[::1] labels, Py_ssize_t B) noexcept nogil:
    cdef Py_ssize_t h, k, Q = X.shape[1]
    cdef u32 best, d
    cdef u8 dst
    for h in range(X.shape[0]):
        if valid[h]:
            best, dst = B+1, 0
            for k in range(R.shape[0]):
                d = distance(mode, &X[h, 0], &R[k, 0], Q, best)
                if d <= best:
                    best, dst = d, k
            labels[h] = dst


### Assign complete haplotypes to their nearest packed reference median
def predict_haplotypes(const u8[:, ::1] G, const u8[:, ::1] medians):
    """Exact packed nearest-median assignment. Any missing allele excludes its haplotype."""
    cdef Py_ssize_t B = G.shape[0], H = G.shape[1], K = medians.shape[0]
    cdef Py_ssize_t Q = (B + 63) >> 6, j, h, k
    cdef u8 value
    cdef bint bad = False
    if B < 1 or B >= 2**32 - 1 or H < 1 or medians.shape[1] != B or K > 255:
        raise ValueError("Invalid prediction genotype/median dimensions")
    cdef u64[:, ::1] X = np.zeros((H, Q), dtype=np.uint64)
    cdef u64[:, ::1] R = np.zeros((K, Q), dtype=np.uint64)
    cdef u8[::1] valid = np.ones(H, dtype=np.uint8)
    cdef u8[::1] labels = np.full(H, 255, dtype=np.uint8)
    with nogil:
        for j in range(B):
            for h in range(H):
                value = G[j, h]
                if value == 255:
                    valid[h] = 0
                elif value > 1:
                    bad = True
                else:
                    X[h, j >> 6] |= <u64>value << (j & 63)
            for k in range(K):
                value = medians[k, j]
                if value > 1:
                    bad = True
                else:
                    R[k, j >> 6] |= <u64>value << (j & 63)
        if not bad and K:
            if Q == 1:
                nearest[u8](0, X, R, valid, labels, B)
            else:
                nearest[u64](0, X, R, valid, labels, B)
    if bad:
        raise ValueError("Prediction requires binary medians and binary or missing GT")
    return np.asarray(labels)
