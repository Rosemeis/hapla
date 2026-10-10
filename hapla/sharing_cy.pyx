# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False
# cython: cdivision=True
"""Categorical PBWT matches and sparse, competitive haplotype sharing."""

__author__ = "Jonas Meisner"

from cython.parallel cimport prange, threadid
cimport openmp as omp
import numpy as np
from libc.stdint cimport uint8_t, uint32_t, int64_t
from libc.stdio cimport FILE, fopen, fclose, fwrite
from libc.math cimport sqrt

ctypedef uint8_t u8
ctypedef uint32_t u32
ctypedef int64_t i64
ctypedef double f64


##### Matching


### Exclude identical diploid pairs, allowing chromosome-wide phase swaps
cpdef bint variable(const u8[:, ::1] Z):
    cdef:
        i64 w, i, N = Z.shape[1] // 2
        u8 a, b, x, y
        u8[::1] same = np.full(N, 3, np.uint8)
    with nogil:
        for w in range(Z.shape[0]):
            a, b = Z[w, 0], Z[w, 1]
            if a == 255 or b == 255:
                return True
            for i in range(1, N):
                x, y = Z[w, 2 * i], Z[w, 2 * i + 1]
                if x == 255 or y == 255:
                    return True
                same[i] &= (a == x and b == y) | ((a == y and b == x) << 1)
                if same[i] == 0:
                    return True
    return False


### Select tied matches by fixed sample priority
cdef int _select(
    const u32* tree,
    i64 node,
    i64 l,
    i64 r,
    i64 beg,
    i64 end,
    u32* dst,
    int n,
    int cap,
) noexcept nogil:
    cdef i64 mid, first, j, v = tree[node]
    if r <= beg or l >= end or (n == cap and v >= dst[n - 1]):
        return n
    if r - l == 1:
        j = min(n, cap - 1)
        while j > 0 and dst[j - 1] > v:
            dst[j] = dst[j - 1]
            j -= 1
        dst[j] = v
        return min(n + 1, cap)
    mid = (l + r) // 2
    first = tree[2 * node] <= tree[2 * node + 1]
    if first:
        n = _select(tree, 2 * node, l, mid, beg, end, dst, n, cap)
        return _select(tree, 2 * node + 1, mid, r, beg, end, dst, n, cap)
    n = _select(tree, 2 * node + 1, mid, r, beg, end, dst, n, cap)
    return _select(tree, 2 * node, l, mid, beg, end, dst, n, cap)


### Stream set-maximal matches with O(H) scratch
def matches(
    const u8[:, ::1] Z,
    const u8[::1] cut,
    const u8[::1] info,
    const i64[::1] rank,
    int cap,
    bytes path,
    int bins = 1,
):
    cdef:
        i64 W = Z.shape[0], H = Z.shape[1], base = 1
        i64 w, i, j, h, v, b, l, r, edge, top, low, high, mid
        i64 total = 0, tied = 0, left, right, pos, chunk, slot, cl, cr
        int z, m = 0, t, n, keep, error = 0
        bint stop, extend
        (FILE*)[32] files
    if W < 1 or H < 4 or H % 2 or max(W, H) >= 2**32 or cap < 1:
        raise ValueError("Sharing requires positive windows, matches, and at least two samples")
    if cut.shape[0] != W or info.shape[0] != W or rank.shape[0] != H:
        raise ValueError("Invalid sharing index dimensions")
    if bins < 1 or bins > 32:
        raise ValueError("Sharing cache requires 1..32 partitions")
    if not np.array_equal(np.sort(rank), np.arange(H)):
        raise ValueError("Sharing priorities must be a permutation")
    while base < H:
        base *= 2
    cdef:
        u32[::1] a = np.argsort(rank).astype(np.uint32)
        u32[::1] inv = np.asarray(a).copy()
        u32[::1] nxt = np.empty(H, np.uint32)
        i64[::1] d = np.zeros(H + 1, np.int64)
        i64[::1] nd = np.empty(H + 1, np.int64)
        u32[::1] lo = np.empty(H + 1, np.uint32)
        u32[::1] hi = np.empty(H + 1, np.uint32)
        i64[::1] prev = np.empty(H, np.int64)
        i64[::1] after = np.empty(H, np.int64)
        u32[::1] stack = np.empty(H + 1, np.uint32)
        u32[::1] tree = np.full(2 * base, H, np.uint32)
        u32[::1] pick = np.empty(min(<i64>cap + 2, H), np.uint32)
        i64[::1] count = np.zeros(H, np.int64)
        u32[:, :, ::1] buf = np.empty((bins, 4096, 4), np.uint32)
        i64[::1] used = np.zeros(bins, np.int64)
        i64[256] last
        i64[256] cnt
        i64[256] off
    cap = min(cap, H - 2)
    keep = cap + 2
    chunk = (H // 2 + bins - 1) // bins
    paths = [path if bins == 1 else path + b"." + str(i).encode() for i in range(bins)]
    for i in range(bins):
        files[i] = fopen(paths[i], "wb")
        if files[i] == NULL:
            for j in range(i):
                fclose(files[j])
            raise OSError("Cannot create sharing match cache")
    with nogil:
        for w in range(W + 1):
            stop = w == W
            if not stop:
                stop = cut[w] != 0 or info[w] == 0
            d[0] = d[H] = w + 1

            # Nearest strictly larger divergence bounds each best suffix group
            top = 0
            stack[0] = 0
            for i in range(1, H):
                while top > 0 and d[stack[top]] <= d[i]:
                    top -= 1
                lo[i] = stack[top]
                top += 1
                stack[top] = i
            top = 0
            stack[0] = H
            for i in range(H - 1, 0, -1):
                while top > 0 and d[stack[top]] <= d[i]:
                    top -= 1
                hi[i] = stack[top]
                top += 1
                stack[top] = i

            # Find extendable ties without scanning equal groups
            if not stop:
                for z in range(256):
                    last[z] = -1
                for i in range(H):
                    z = Z[w, a[i]]
                    prev[i] = last[z]
                    last[z] = i
                for z in range(256):
                    last[z] = H
                for i in range(H - 1, -1, -1):
                    z = Z[w, a[i]]
                    after[i] = last[z]
                    last[z] = i
            for i in range(H):
                tree[base + i] = rank[a[i]]
                nxt[a[i]] = i
            for i in range(base - 1, 0, -1):
                tree[i] = min(tree[2 * i], tree[2 * i + 1])

            cl, cr = -1, -1
            for i in range(H):
                h = a[i]
                l, r = i - 1, i + 1
                left, right = d[i], d[i + 1]
                edge = i
                if l >= 0 and a[l] // 2 == h // 2:
                    if d[l] > left:
                        edge = l
                    left = max(left, d[l])
                    l -= 1
                j = i + 1
                if r < H and a[r] // 2 == h // 2:
                    if d[r + 1] > right:
                        j = r + 1
                    right = max(right, d[r + 1])
                    r += 1
                if right < left:
                    edge = j
                b = min(left, right)
                if b >= w:
                    continue
                l, r = lo[edge], hi[edge]
                extend = False
                if not stop and Z[w, h] != 255:
                    j = prev[i]
                    if j >= 0 and a[j] // 2 == h // 2:
                        j = prev[j]
                    extend = j >= l
                    j = after[i]
                    if j < H and a[j] // 2 == h // 2:
                        j = after[j]
                    extend |= j < r
                if extend:
                    continue
                # Keep two extra matches for self-exclusion
                if l != cl or r != cr:
                    cl, cr, m = l, r, 0
                    if r - l <= min(keep, 32):
                        for j in range(l, r):
                            v, t = rank[a[j]], m
                            while t > 0 and pick[t - 1] > v:
                                pick[t] = pick[t - 1]
                                t -= 1
                            pick[t] = v
                            m += 1
                    else:
                        m = _select(&tree[0], 1, 0, base, l, r, &pick[0], 0, keep)
                tied += r - l - 1 - (l <= nxt[h ^ 1] < r) > cap
                slot = (h // 2) // chunk
                n, pos = 0, used[slot]
                for t in range(m):
                    v = inv[pick[t]]
                    if v // 2 == h // 2:
                        continue
                    buf[slot, pos, 0] = h
                    buf[slot, pos, 1] = v
                    buf[slot, pos, 2] = b
                    buf[slot, pos, 3] = w
                    pos += 1
                    n += 1
                    if pos == 4096:
                        if fwrite(&buf[slot, 0, 0], 16, pos, files[slot]) != 4096:
                            error = 1
                            break
                        pos = 0
                    if n == cap:
                        break
                used[slot] = pos
                count[h] += n
                total += n
                if error:
                    break
            if error or w == W:
                break

            # Flush previous runs at cuts and constant windows
            if stop:
                for i in range(H + 1):
                    d[i] = w
            for z in range(256):
                cnt[z], last[z] = 0, -1
            for i in range(H):
                z = Z[w, a[i]] if info[w] else 255
                cnt[z] += 1
            pos = 0
            for z in range(256):
                off[z] = pos
                pos += cnt[z]

            # Update the PBWT using a monotone range-maximum stack
            top = 0
            for i in range(H):
                while top > 0 and d[stack[top - 1]] <= d[i]:
                    top -= 1
                stack[top] = i
                top += 1
                z = Z[w, a[i]] if info[w] else 255
                v = w + 1
                if last[z] >= 0 and z != 255:
                    low, high = 0, top
                    while low < high:
                        mid = (low + high) // 2
                        if stack[mid] <= last[z]:
                            low = mid + 1
                        else:
                            high = mid
                    v = d[stack[low]]
                j = off[z]
                nxt[j], nd[j] = a[i], v
                off[z] += 1
                last[z] = i
            for i in range(H):
                a[i], d[i] = nxt[i], nd[i]
        for slot in range(bins):
            if used[slot] and not error:
                error |= fwrite(&buf[slot, 0, 0], 16, used[slot], files[slot]) != <size_t>used[slot]
            if fclose(files[slot]) != 0:
                error = 1
    if error:
        raise OSError("Cannot write sharing match cache")
    return np.asarray(count), total, tied


##### Sharing profiles


### Group matches by haplotype without sorting
def group(const u32[:, ::1] src, i64[::1] pos, u32[:, ::1] dst):
    cdef i64 e, i, p
    with nogil:
        for e in range(src.shape[0]):
            i = src[e, 0]
            p = pos[i]
            dst[p, 0] = src[e, 1]
            dst[p, 1] = src[e, 2]
            dst[p, 2] = src[e, 3]
            pos[i] += 1


### Integrate run weights without subtracting large moments
cdef f64 _paint(
    const u32[:, ::1] seg,
    const f64* x,
    const f64* left,
    const f64* right,
    i64 W,
    i64 beg,
    i64 end,
    f64* den,
    f64* val,
) noexcept nogil:
    cdef:
        i64 e, w, b, r
        f64 a, z, total = 0, v
    for w in range(W):
        den[w] = 0
    for e in range(beg, end):
        b, r = seg[e, 1], seg[e, 2]
        a, z = left[b], right[r - 1]
        for w in range(b, r):
            den[w] += (x[w] - a) * (z - x[w])
    for w in range(W):
        if den[w] > 0:
            total += right[w] - left[w]
            den[w] = (right[w] - left[w]) / den[w]
    for e in range(beg, end):
        b, r = seg[e, 1], seg[e, 2]
        # Tied matches share the same integral
        if e > beg and b == seg[e - 1, 1] and r == seg[e - 1, 2]:
            val[e] = val[e - 1]
            continue
        a, z, v = left[b], right[r - 1], 0
        for w in range(b, r):
            v += (x[w] - a) * (z - x[w]) * den[w]
        val[e] = v
    return total


### Paint haplotypes with one window buffer per worker
def paint(
    const u32[:, ::1] seg,
    const i64[::1] ptr,
    const f64[::1] x,
    const f64[::1] left,
    const f64[::1] right,
    f64[::1] val,
):
    cdef:
        i64 H = ptr.shape[0] - 1, W = x.shape[0], h, t
        int nt = max(1, min(H, omp.omp_get_max_threads()))
        f64[:, ::1] work = np.empty((nt, W))
        f64[::1] total = np.empty(H)
    for h in prange(H, nogil=True, schedule="dynamic", num_threads=nt):
        t = threadid()
        total[h] = _paint(
            seg, &x[0], &left[0], &right[0], W, ptr[h], ptr[h + 1], &work[t, 0], &val[0]
        )
    return np.asarray(total).reshape(-1, 2).sum(axis=1)


### Sum matching segments one sample at a time
def compact(
    const u32[:, ::1] seg,
    const i64[::1] ptr,
    const f64[::1] val,
    i64[::1] rows,
    u32[::1] col,
    f64[::1] out,
    f64[::1] sums,
    i64[::1] seen,
    i64[::1] hit,
    i64 first,
):
    cdef i64 i, j, e, n, used = 0
    rows[0] = 0
    with nogil:
        for i in range(rows.shape[0] - 1):
            n = 0
            for e in range(ptr[2 * i], ptr[2 * i + 2]):
                j = seg[e, 0] // 2
                if val[e] <= 0:
                    continue
                if seen[j] != first + i:
                    seen[j], sums[j] = first + i, 0
                    hit[n] = j
                    n += 1
                sums[j] += val[e]
            for e in range(n):
                j = hit[e]
                col[used], out[used] = j, sums[j]
                used += 1
            rows[i + 1] = used
    return used


### Combine chromosome rows and normalize, with optional square roots
def combine(
    data,
    const f64[::1] cov,
    i64 first,
    i64[::1] rows,
    u32[::1] dst,
    f64[::1] out,
    f64[::1] sums,
    u32[::1] hit,
    bint root,
):
    cdef:
        i64 i, j, e, n, used = 0
        f64 v
        const i64[::1] ptr
        const u32[::1] col
        const f64[::1] val
    rows[0] = 0
    for i in range(rows.shape[0] - 1):
        n = 0
        for d in data:
            ptr, col, val = d
            with nogil:
                for e in range(ptr[first + i], ptr[first + i + 1]):
                    j = col[e]
                    if sums[j] == 0:
                        hit[n] = j
                        n += 1
                    sums[j] += val[e]
        with nogil:
            for e in range(n):
                j = hit[e]
                v = sums[j] / cov[first + i]
                dst[used], out[used] = j, sqrt(v) if root else v
                sums[j] = 0
                used += 1
            rows[i + 1] = used
    return used


##### Matrix products


### Cache the transpose for independent row products
def transpose(
    const i64[::1] ptr,
    const u32[::1] col,
    const f64[::1] val,
    i64[::1] rows,
    u32[::1] dst,
    f64[::1] out,
):
    cdef:
        i64 N = ptr.shape[0] - 1, i, j, e, p
        i64[::1] pos = np.zeros(N, np.int64)
    rows[0] = 0
    with nogil:
        for e in range(col.shape[0]):
            pos[col[e]] += 1
        for i in range(N):
            rows[i + 1] = rows[i] + pos[i]
            pos[i] = rows[i]
        for i in range(N):
            for e in range(ptr[i], ptr[i + 1]):
                j = col[e]
                p = pos[j]
                dst[p], out[p] = i, val[e]
                pos[j] += 1


### Apply sparse sharing to a narrow sketch
def product(
    const i64[::1] ptr,
    const u32[::1] col,
    const f64[::1] val,
    const f64[:, ::1] Q,
    f64[:, ::1] out,
):
    cdef:
        i64 i, j, e, l, N = ptr.shape[0] - 1, L = Q.shape[1]
        f64 v
    for i in prange(N, nogil=True, schedule="static"):
        for e in range(ptr[i], ptr[i + 1]):
            j, v = col[e], val[e]
            for l in range(L):
                out[i, l] += v * Q[j, l]


### Measure column means and centered variation from the combined profiles
def moments(const i64[::1] ptr, const u32[::1] col, const f64[::1] val):
    cdef:
        i64 N = ptr.shape[0] - 1, i, j, e
        f64 v, s, ss = 0.0
        f64[::1] mean = np.zeros(N)
    with nogil:
        for i in range(N):
            s = 0
            for e in range(ptr[i], ptr[i + 1]):
                j, v = col[e], val[e]
                mean[j] += v
                s += v * v
            ss += s
        for j in range(N):
            mean[j] /= N
            ss -= N * mean[j] * mean[j]
    return np.asarray(mean), ss


### Expand and center only the rows needed by the current kernel tile
def rows(
    const i64[::1] ptr, const u32[::1] col, const f64[::1] val, i64 beg,
    const f64[::1] mean, f64[:, ::1] out,
):
    cdef i64 i, j, e
    for i in prange(out.shape[0], nogil=True, schedule="static"):
        for j in range(out.shape[1]):
            out[i, j] = 0
        for e in range(ptr[beg + i], ptr[beg + i + 1]):
            out[i, col[e]] += val[e]
        for j in range(out.shape[1]):
            out[i, j] -= mean[j]


### Form centered group profiles directly from the cached transpose
def centroids(
    const i64[::1] ptr,
    const u32[::1] col,
    const f64[::1] val,
    const int[::1] Z,
    const f64[::1] inv,
    const f64[::1] mean,
    f64[:, ::1] out,
):
    cdef:
        i64 i, e, g, N = ptr.shape[0] - 1, K = out.shape[1]
    for i in prange(N, nogil=True, schedule="static"):
        for g in range(K):
            out[i, g] = -mean[i]
        for e in range(ptr[i], ptr[i + 1]):
            g = Z[col[e]]
            out[i, g] += val[e] * inv[g]
