# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False
"""Bounded native HTSlib reading of phased, diploid, biallelic GT."""

__author__ = "Jonas Meisner"

import os

from libc.stdint cimport uint8_t, uint32_t, int8_t, int32_t, int64_t
from libc.stdlib cimport free


##### Native declarations


cdef extern from "htslib/kstring.h" nogil:
    ctypedef struct kstring_t:
        size_t l
        size_t m
        char* s
    int kputs(const char*, kstring_t*)
    int kputc(int, kstring_t*)
    int kputll(long long, kstring_t*)

cdef extern from "htslib/hts.h" nogil:
    ctypedef struct htsFile:
        pass
    htsFile* hts_open(const char*, const char*)
    int hts_close(htsFile*)
    int hts_set_threads(htsFile*, int)
    const char* hts_version()

cdef extern from "htslib/vcf.h" nogil:
    ctypedef struct bcf_hdr_t:
        char** samples
        int32_t[3] n
    ctypedef struct bcf_fmt_t:
        int id
        int n
        int size
        int type
        uint8_t* p
        uint32_t p_len
    ctypedef struct bcf_dec_t:
        char** allele
    ctypedef struct bcf1_t:
        int64_t pos
        int32_t rid
        uint32_t n_allele
        uint32_t n_sample
        int errcode
        bcf_dec_t d
    bcf_hdr_t* bcf_hdr_read(htsFile*)
    void bcf_hdr_destroy(bcf_hdr_t*)
    int bcf_hdr_nsamples(const bcf_hdr_t*)
    const char* bcf_hdr_id2name(const bcf_hdr_t*, int)
    int bcf_hdr_id2int(const bcf_hdr_t*, int, const char*)
    bcf1_t* bcf_init()
    void bcf_destroy(bcf1_t*)
    int bcf_read(htsFile*, bcf_hdr_t*, bcf1_t*)
    int bcf_unpack(bcf1_t*, int)
    bcf_fmt_t* bcf_get_fmt_id(bcf1_t*, int)
    int bcf_get_genotypes(const bcf_hdr_t*, bcf1_t*, int32_t**, int*)
    int BCF_BT_INT8
    int BCF_DT_CTG
    int BCF_DT_ID
    int BCF_UN_STR


##### Genotype reader


### Decode phased INT8 GT and diagnose invalid rows with scalar code
cdef bint decode8(const int8_t* raw, uint8_t* output, int samples, uint8_t* missing) noexcept nogil:
    cdef:
        int i, a, b
        unsigned int invalid = 0, absent = 0
    for i in range(samples):
        a, b = raw[2 * i], raw[2 * i + 1]
        invalid |= (a < 0) | (b < 0) | (a > 5) | (b > 5)
        invalid |= ((a >> 1) != (b >> 1)) & ((b & 1) == 0)
        absent |= (a < 2) | (b < 2)
        output[2 * i] = <uint8_t>((a >> 1) - 1)
        output[2 * i + 1] = <uint8_t>((b >> 1) - 1)
    missing[0] = absent != 0
    return invalid == 0


### Own HTSlib pointers and fill a reusable block with the GIL released
cdef class Reader:
    cdef:
        htsFile* handle
        bcf_hdr_t* header
        bcf1_t* record
        int32_t* gt
        int cap
        int N
        int gid
        int rid0
        int64_t pos0
        kstring_t sbuf
        bint failed, busy, save
    cdef readonly bint finished
    cdef readonly object htslib_version
    cdef public object samples, contigs, path, sites
    cdef public unsigned long long variants

    ### Initialize sort sentinels before opening the input
    def __cinit__(self):
        self.rid0 = -1
        self.pos0 = -1

    ### Validate the header and retain one reusable record and GT buffer
    def __init__(self, path, int threads = 0, bint save = False):
        cdef bytes encoded
        if threads < 0:
            raise ValueError("HTSlib threads must be nonnegative")
        self.path = os.fspath(path)
        self.save = save
        self.sites = b""
        self.htslib_version = hts_version().decode("ascii")
        encoded = os.fsencode(path)
        self.handle = hts_open(encoded, b"r")
        if self.handle == NULL:
            raise OSError(f"Cannot open genotype file: {self.path}")
        if threads and hts_set_threads(self.handle, threads) != 0:
            raise OSError("HTSlib could not initialize decompression threads")
        self.header = bcf_hdr_read(self.handle)
        if self.header == NULL:
            raise ValueError(f"Invalid VCF/BCF header: {self.path}")
        self.N = bcf_hdr_nsamples(self.header)
        if self.N <= 0:
            raise ValueError("VCF/BCF contains no samples")
        self.gid = bcf_hdr_id2int(self.header, BCF_DT_ID, b"GT")
        if self.gid < 0:
            raise ValueError("VCF/BCF header does not declare FORMAT/GT")
        self.samples = [self.header.samples[i].decode("utf-8") for i in range(self.N)]
        if len(set(self.samples)) != len(self.samples):
            raise ValueError("VCF/BCF sample IDs must be unique")
        for name in self.samples:
            if not name or any(map(str.isspace, name)):
                raise ValueError("Sample IDs must be nonempty and contain no whitespace")
        self.contigs = [
            bcf_hdr_id2name(self.header, i).decode("utf-8")
            for i in range(self.header.n[BCF_DT_CTG])
        ]
        for name in self.contigs:
            if not name or any(map(str.isspace, name)):
                raise ValueError("Contig names must be nonempty and contain no whitespace")
        self.record = bcf_init()
        if self.record == NULL:
            raise MemoryError("Cannot allocate HTSlib record")

    ### Close the reader when no read is in progress
    def close(self):
        if self.busy:
            raise RuntimeError("Cannot close a reader while readInto is running")
        self._release()

    ### Release native pointers once, including partially initialized readers
    cdef void _release(self) noexcept:
        if self.handle != NULL:
            hts_close(self.handle)
            self.handle = NULL
        if self.header != NULL:
            bcf_hdr_destroy(self.header)
            self.header = NULL
        if self.record != NULL:
            bcf_destroy(self.record)
            self.record = NULL
        if self.gt != NULL:
            free(self.gt)
            self.gt = NULL
        if self.sbuf.s != NULL:
            free(self.sbuf.s)
            self.sbuf.s = NULL
            self.sbuf.l = self.sbuf.m = 0

    ### Release native buffers when the reader is destroyed
    def __dealloc__(self):
        self._release()

    ### Return the reader for context management
    def __enter__(self):
        return self

    ### Close the reader when its context ends
    def __exit__(self, *args):
        self.close()

    ### Fill variant-major arrays, using direct INT8 or reusable INT32 GT decoding
    def readInto(
        self,
        uint8_t[:, ::1] output,
        int64_t[::1] pos,
        int32_t[::1] contigs,
        uint8_t[::1] missing,
    ):
        cdef:
            Py_ssize_t row = 0, i, size = output.shape[0]
            int status = 0, a = 0, b = 0, error = 0
            bcf_fmt_t* fmt
            int8_t* raw = NULL
            bint decoded
        if self.busy:
            raise RuntimeError("Concurrent calls on the same genotype reader are unsupported")
        if self.handle == NULL or self.failed:
            raise ValueError("Genotype reader is closed or failed")
        if output.shape[1] != 2 * self.N:
            raise ValueError("Genotype buffer has the wrong number of haplotypes")
        if pos.shape[0] != size or contigs.shape[0] != size or missing.shape[0] != size:
            raise ValueError("Genotype and metadata buffer lengths differ")
        self.sites = b""
        self.sbuf.l = 0
        if self.finished:
            return 0
        self.busy = True
        with nogil:
            while row < size:
                status = bcf_read(self.handle, self.header, self.record)
                if status == -1:
                    self.finished = True
                    break
                if status < -1 or self.record.errcode:
                    error = 1
                    break
                if self.record.n_allele != 2:
                    error = 2
                    break
                if self.record.n_sample != self.N:
                    error = 3
                    break
                if self.record.rid < 0 or self.record.rid >= self.header.n[BCF_DT_CTG]:
                    error = 4
                    break
                if (
                    self.record.pos < 0
                    or self.record.rid < self.rid0
                    or (self.record.rid == self.rid0 and self.record.pos < self.pos0)
                ):
                    error = 4
                    break
                fmt = bcf_get_fmt_id(self.record, self.gid)
                if fmt == NULL:
                    error = 5
                    break
                if fmt.n != 2:
                    error = 6
                    break
                if fmt.type == BCF_BT_INT8:
                    if fmt.size != 2 or fmt.p_len < 2 * self.N:
                        error = 3
                        break
                    raw = <int8_t*>fmt.p
                    decoded = decode8(raw, &output[row, 0], self.N, &missing[row])
                else:
                    status = bcf_get_genotypes(self.header, self.record, &self.gt, &self.cap)
                    if status != 2 * self.N:
                        error = 6
                        break
                    missing[row] = 0
                    decoded = False
                for i in range(0 if decoded else self.N):
                    if fmt.type == BCF_BT_INT8:
                        a, b = raw[2 * i], raw[2 * i + 1]
                    else:
                        a, b = self.gt[2 * i], self.gt[2 * i + 1]

                    # Missing is 0/1. Negative values include vector-end/ploidy markers.
                    if a < 0 or b < 0:
                        error = 6
                        break
                    if a > 5 or b > 5:
                        error = 7
                        break

                    # Homozygous or wholly missing calls have no phase ambiguity.
                    if (a >> 1) != (b >> 1) and not (b & 1):
                        error = 8
                        break
                    output[row, 2 * i] = 255 if a < 2 else (a >> 1) - 1
                    output[row, 2 * i + 1] = 255 if b < 2 else (b >> 1) - 1
                    if a < 2 or b < 2:
                        missing[row] = 1
                if error:
                    break
                if self.save:
                    if bcf_unpack(self.record, BCF_UN_STR) < 0:
                        error = 1
                        break

                    # Format a whole read block without taking the GIL for each variant.
                    status = kputs(bcf_hdr_id2name(self.header, self.record.rid), &self.sbuf)
                    status |= kputc(9, &self.sbuf)
                    status |= kputll(self.record.pos + 1, &self.sbuf)
                    status |= kputc(9, &self.sbuf)
                    status |= kputs(self.record.d.allele[0], &self.sbuf)
                    status |= kputc(9, &self.sbuf)
                    status |= kputs(self.record.d.allele[1], &self.sbuf)
                    status |= kputc(10, &self.sbuf)
                    if status < 0:
                        error = 9
                        break
                pos[row] = self.record.pos + 1
                contigs[row] = self.record.rid
                self.rid0, self.pos0 = self.record.rid, self.record.pos
                self.variants += 1
                row += 1
        self.busy = False
        if self.sbuf.l:
            self.sites = self.sbuf.s[: self.sbuf.l]
        if error:
            self.failed = True
            if error == 9:
                raise MemoryError("Cannot allocate the variant identity buffer")
            reason = {
                1: "Malformed or truncated VCF/BCF record",
                2: "Only biallelic variants are supported",
                3: "Malformed genotype sample buffer",
                4: "Variants must be sorted by header contig order and position",
                5: "Missing FORMAT/GT",
                6: "Only diploid GT is supported (mixed ploidy/vector-end encountered)",
                7: "GT allele index exceeds the biallelic range",
                8: "Unphased heterozygous or partially missing GT is unsupported",
            }[error]
            raise ValueError(
                f"{reason} at record {self.variants + 1}, position {self.record.pos + 1}"
            )
        return row
