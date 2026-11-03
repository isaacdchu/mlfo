#ifndef MLFO_IM2COL_HPP
#define MLFO_IM2COL_HPP

#include <algorithm>
#include <cstddef>

#include "concepts.hpp"

// lowers a channels last 2D convolution to a matrix multiplication
// for one sample with input (H, W, C), the column matrix has one row per output position (P * Q)
// and one column per kernel element (R * S * C), in the same (r, s, c) order as a (R, S, C, K) kernel
// so col (P * Q, R * S * C) * kernel (R * S * C, K) = output (P * Q, K)

namespace mlfo::misc {

struct Conv2DGeometry {
    std::size_t h;
    std::size_t w;
    std::size_t c;
    std::size_t r;
    std::size_t s;
    std::size_t p;
    std::size_t q;
    std::size_t stride_h;
    std::size_t stride_w;
    std::size_t padding_h;
    std::size_t padding_w;

    // row length of the column matrix
    std::size_t patch_size() const {
        return r * s * c;
    }

    // rows of the column matrix per sample
    std::size_t positions() const {
        return p * q;
    }

    // the column matrix is exactly the input, so im2col can be skipped
    bool is_pointwise() const {
        return r == 1 && s == 1 && stride_h == 1 && stride_w == 1 && padding_h == 0 && padding_w == 0;
    }

    // the kernel columns [first, last) that land inside the input for output column qi
    // the input columns they read are adjacent, so with channels last a kernel row reads one
    // contiguous run of (last - first) * c values
    struct ColumnRange {
        std::size_t first;
        std::size_t last;
    };

    ColumnRange columns_inside(std::size_t qi) const {
        // kernel column si reads padded input column qi * stride_w + si, which is inside the
        // input when it is in [padding_w, padding_w + w)
        const std::size_t start = qi * stride_w;
        const std::size_t first = start >= padding_w ? 0 : std::min(s, padding_w - start);
        const std::size_t last = start >= padding_w + w ? 0 : std::min(s, padding_w + w - start);
        return {first, std::max(first, last)};
    }
};

// writes the column matrix (P * Q, R * S * C) of one sample's input (H, W, C) into col
// positions in the padding are written as zero
template <Number T>
void im2col(const Conv2DGeometry& g, const T* __restrict input, T* __restrict col) {
    const std::size_t row_size = g.s * g.c;
    for (std::size_t pi = 0; pi < g.p; pi++) {
        for (std::size_t qi = 0; qi < g.q; qi++) {
            const auto [first, last] = g.columns_inside(qi);
            // input column of kernel column first, only used when the range is not empty
            const std::size_t wi = qi * g.stride_w + first - g.padding_w;
            T* row = col + (pi * g.q + qi) * g.patch_size();
            for (std::size_t ri = 0; ri < g.r; ri++) {
                // row in the padded input, all zero if it is in the padding
                const std::size_t hi = pi * g.stride_h + ri;
                T* dst = row + ri * row_size;
                if (hi < g.padding_h || hi - g.padding_h >= g.h || first == last) {
                    std::fill(dst, dst + row_size, T{0});
                    continue;
                }
                const T* src = input + ((hi - g.padding_h) * g.w + wi) * g.c;
                std::fill(dst, dst + first * g.c, T{0});
                std::copy(src, src + (last - first) * g.c, dst + first * g.c);
                std::fill(dst + last * g.c, dst + row_size, T{0});
            }
        }
    }
}

// the adjoint of im2col, adds each element of col back into the input position it was copied from
// input elements covered by several windows receive the sum, elements in the padding are dropped
template <Number T>
void col2im(const Conv2DGeometry& g, const T* __restrict col, T* __restrict input) {
    const std::size_t row_size = g.s * g.c;
    for (std::size_t pi = 0; pi < g.p; pi++) {
        for (std::size_t qi = 0; qi < g.q; qi++) {
            const auto [first, last] = g.columns_inside(qi);
            if (first == last) {
                continue;
            }
            const std::size_t wi = qi * g.stride_w + first - g.padding_w;
            const std::size_t run = (last - first) * g.c;
            const T* row = col + (pi * g.q + qi) * g.patch_size();
            for (std::size_t ri = 0; ri < g.r; ri++) {
                const std::size_t hi = pi * g.stride_h + ri;
                if (hi < g.padding_h || hi - g.padding_h >= g.h) {
                    continue;
                }
                const T* src = row + ri * row_size + first * g.c;
                T* dst = input + ((hi - g.padding_h) * g.w + wi) * g.c;
                for (std::size_t i = 0; i < run; i++) {
                    dst[i] += src[i];
                }
            }
        }
    }
}

} // namespace mlfo::misc

#endif // MLFO_IM2COL_HPP
