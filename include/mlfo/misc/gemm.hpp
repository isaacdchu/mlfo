#ifndef MLFO_GEMM_HPP
#define MLFO_GEMM_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

#include "concepts.hpp"

// general matrix multiplication on contiguous row-major matrices
// every variant accumulates into c (c += ...), so zero c first to overwrite it
// the n and t in the names say whether a and b are used as is or transposed
//
// all variants share one register tiled kernel:
// - k is split into blocks of KC, so the packed panels of a block stay in cache
// - b is packed into panels of NR columns, laid out (KC, NR) so the kernel reads each row of a
//   panel as one contiguous vector. packing is also where b's transpose happens
// - the microkernel computes an MR x NR tile of c in local accumulators, which the compiler keeps
//   in registers, so each loaded value of a and b is used NR or MR times
// - a is only read one value at a time (broadcast against a row of b), so it is read in place
//   through a row and a column stride instead of being packed, which also covers its transpose
// b panels at the edges are zero padded and a rows past the edge repeat the last row, so the
// kernel always runs a full tile and only the store is clipped

namespace mlfo::misc {

namespace detail {

template <Number T>
struct GemmTiling {
    // rows of a tile
    static constexpr std::size_t MR = 4;
    // columns of a tile, 32 bytes of accumulators per row
    static constexpr std::size_t NR = 32 / sizeof(T);
    // depth of a k block
    static constexpr std::size_t KC = 256;
};

// packing space, reused across calls so they don't allocate
template <Number T>
std::vector<T>& gemm_b_panels(std::size_t size) {
    thread_local std::vector<T> buffer;
    if (buffer.size() < size) {
        buffer.resize(size);
    }
    return buffer;
}

// c (mr x nr, row stride ldc) += A (MR x kc) * b panel (kc, NR)
// row i of A starts at a_rows[i], and consecutive columns are a_col_stride apart
template <Number T>
inline void gemm_microkernel(
    std::size_t kc,
    const T* const (&a_rows)[GemmTiling<T>::MR],
    std::size_t a_col_stride,
    const T* __restrict b,
    T* __restrict c,
    std::size_t ldc,
    std::size_t mr,
    std::size_t nr
) {
    constexpr std::size_t MR = GemmTiling<T>::MR;
    constexpr std::size_t NR = GemmTiling<T>::NR;
    // the tile loops must be fully unrolled for acc to live in registers, which gcc only does
    // on its own from -O3, so it is requested explicitly
    T acc[MR][NR] = {};
    for (std::size_t p = 0; p < kc; p++) {
        const T* b_row = b + p * NR;
        const std::size_t a_offset = p * a_col_stride;
        #pragma GCC unroll 8
        for (std::size_t i = 0; i < MR; i++) {
            const T a_value = a_rows[i][a_offset];
            #pragma GCC unroll 16
            for (std::size_t j = 0; j < NR; j++) {
                acc[i][j] += a_value * b_row[j];
            }
        }
    }
    if (mr == MR && nr == NR) {
        for (std::size_t i = 0; i < MR; i++) {
            for (std::size_t j = 0; j < NR; j++) {
                c[i * ldc + j] += acc[i][j];
            }
        }
        return;
    }
    for (std::size_t i = 0; i < mr; i++) {
        for (std::size_t j = 0; j < nr; j++) {
            c[i * ldc + j] += acc[i][j];
        }
    }
}

// c (m x n) += A * B
// A's element (i, p) is a[i * a_row_stride + p * a_col_stride], b_at(p, j) gives B's (p, j)
template <Number T, class BAt>
void gemm(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const T* a,
    std::size_t a_row_stride,
    std::size_t a_col_stride,
    BAt b_at,
    T* c
) {
    constexpr std::size_t MR = GemmTiling<T>::MR;
    constexpr std::size_t NR = GemmTiling<T>::NR;
    constexpr std::size_t KC = GemmTiling<T>::KC;
    if (m == 0 || n == 0 || k == 0) {
        return;
    }
    const std::size_t n_panels = (n + NR - 1) / NR;
    T* b_panels = gemm_b_panels<T>(n_panels * KC * NR).data();
    for (std::size_t p0 = 0; p0 < k; p0 += KC) {
        const std::size_t kc = std::min(KC, k - p0);
        // pack every column panel of B's rows [p0, p0 + kc)
        for (std::size_t panel = 0; panel < n_panels; panel++) {
            const std::size_t j0 = panel * NR;
            const std::size_t nr = std::min(NR, n - j0);
            T* dst = b_panels + panel * KC * NR;
            for (std::size_t p = 0; p < kc; p++) {
                for (std::size_t j = 0; j < nr; j++) {
                    dst[p * NR + j] = b_at(p0 + p, j0 + j);
                }
                for (std::size_t j = nr; j < NR; j++) {
                    dst[p * NR + j] = T{0};
                }
            }
        }
        for (std::size_t i0 = 0; i0 < m; i0 += MR) {
            const std::size_t mr = std::min(MR, m - i0);
            // rows past the edge repeat the last row, so they read valid memory
            // their results are computed but never stored
            const T* a_rows[MR];
            for (std::size_t i = 0; i < MR; i++) {
                a_rows[i] = a + std::min(i0 + i, m - 1) * a_row_stride + p0 * a_col_stride;
            }
            for (std::size_t panel = 0; panel < n_panels; panel++) {
                const std::size_t j0 = panel * NR;
                gemm_microkernel<T>(
                    kc, a_rows, a_col_stride, b_panels + panel * KC * NR,
                    c + i0 * n + j0, n, mr, std::min(NR, n - j0)
                );
            }
        }
    }
}

} // namespace detail

// c (m x n) += a (m x k) * b (k x n)
template <Number T>
void gemm_nn(std::size_t m, std::size_t n, std::size_t k, const T* a, const T* b, T* c) {
    detail::gemm<T>(
        m, n, k, a, k, 1,
        [b, n](std::size_t p, std::size_t j) { return b[p * n + j]; },
        c
    );
}

// c (m x n) += a (m x k) * b^T, where b is stored as (n x k)
template <Number T>
void gemm_nt(std::size_t m, std::size_t n, std::size_t k, const T* a, const T* b, T* c) {
    detail::gemm<T>(
        m, n, k, a, k, 1,
        [b, k](std::size_t p, std::size_t j) { return b[j * k + p]; },
        c
    );
}

// c (m x n) += a^T * b (k x n), where a is stored as (k x m)
template <Number T>
void gemm_tn(std::size_t m, std::size_t n, std::size_t k, const T* a, const T* b, T* c) {
    detail::gemm<T>(
        m, n, k, a, 1, m,
        [b, n](std::size_t p, std::size_t j) { return b[p * n + j]; },
        c
    );
}

} // namespace mlfo::misc

#endif // MLFO_GEMM_HPP
