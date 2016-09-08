// Copyright (c) 2016 Roman Beránek. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <complex>

#include "common.hpp"

namespace reFFT {
template <typename T, std::ptrdiff_t N, FFT_Direction Direction>
class FFT
{
    using int_t = std::ptrdiff_t;
    using real_t = T;
    using cpx_t = std::complex<T>;

    static_assert(std::is_floating_point<T>::value);
    static_assert((N & (N - 1)) == 0, "N must be a power of 2.");

    static constexpr int_t stride(int_t n_out) {
        return (n_out > 0) ? N / n_out : 0;
    }
    static constexpr int_t radix(int_t n_out) {
        return (n_out % 8 == 0) ? 8 : ((n_out % 4 == 0) ? 4 : 2);
    }
    static constexpr int_t remainder(int_t n_out) {
        return n_out / radix(n_out);
    }

public:
    FFT()
    noexcept {
        auto const step = (is_inverse(Direction) ? 2 : -2) * pi<real_t> / N;
        for (auto i = 0u; i < std::size(twiddles); ++i) {
            twiddles[i] = std::polar(real_t{1}, i * step);
        }
    }

    void
    operator()(gsl::span<cpx_t const, N> in, gsl::span<cpx_t, N> out)
    const noexcept {
        step_in(in, out);
    }

private:

#pragma clang diagnostic push
#pragma ide diagnostic ignored "InfiniteRecursion"
    template <int_t N_in, int_t N_out>
    inline typename std::enable_if_t<(N_out > radix(N_out))>
    step_in(gsl::span<cpx_t const, N_in> in, gsl::span<cpx_t, N_out> out)
    const noexcept {
        auto const m_out = remainder(N_out);
        auto const s = stride(N_out);
        auto const m_in = N_in - s;

        step_in(subspan<0*s, m_in>(in), subspan<0*m_out, m_out>(out));
        step_in(subspan<1*s, m_in>(in), subspan<1*m_out, m_out>(out));

        if (radix(N_out) > 2) {
            step_in(subspan<2*s, m_in>(in), subspan<2*m_out, m_out>(out));
            step_in(subspan<3*s, m_in>(in), subspan<3*m_out, m_out>(out));
        }

        if (radix(N_out) > 4) {
            step_in(subspan<4*s, m_in>(in), subspan<4*m_out, m_out>(out));
            step_in(subspan<5*s, m_in>(in), subspan<5*m_out, m_out>(out));
            step_in(subspan<6*s, m_in>(in), subspan<6*m_out, m_out>(out));
            step_in(subspan<7*s, m_in>(in), subspan<7*m_out, m_out>(out));
        }

        butterfly(out);
    }

    template <int_t N_in, int_t N_out>
    inline typename std::enable_if_t<(N_out == radix(N_out))>
    step_in(gsl::span<cpx_t const, N_in> in, gsl::span<cpx_t, N_out> out)
    const noexcept {
        copy_input(in, out);
        butterfly(out);
    }

    template <int_t N_in, int_t N_out>
    inline typename std::enable_if_t<(N_out > 2)>
    copy_input(
        gsl::span<cpx_t const, N_in> input,
        gsl::span<cpx_t, N_out> output
    ) const noexcept {
        copy_input(
            subspan<0, N_in - stride(N_out)>(input),
            subspan<0, N_out/2>(output)
        );
        copy_input(
            subspan<stride(N_out), N_in - stride(N_out)>(input),
            subspan<N_out/2, N_out/2>(output)
        );
    }
#pragma clang diagnostic pop

    template <int_t N_in, int_t N_out>
    inline typename std::enable_if_t<(N_out == 2)>
    copy_input(
        gsl::span<cpx_t const, N_in> input,
        gsl::span<cpx_t, N_out> output
    ) const noexcept {
        output[0] = input[0];
        output[1] = input[stride(N_out)];
    }

    template <int_t N_out>
    inline void
    butterfly(gsl::span<cpx_t, N_out> output)
    const noexcept {
        switch (radix(N_out)) {
            case 8:
                butterfly_radix8(output);
                break;
            case 4:
                butterfly_radix4(output);
                break;
            case 2:
                butterfly_radix2(output);
                break;
            default:
                assert(false);
                break;
        }
    }

    template <int_t N_out>
    inline void
    butterfly_radix2(gsl::span<cpx_t, N_out> out)
    const noexcept {
        auto const m = remainder(N_out);
        auto const s = stride(N_out);

        scissors(out[0], out[m]);
        for (auto i = 1; i < m; ++i) {
            out[i + m] = multiply_fast(out[i + m], twiddles[i * s]);
            scissors(out[i], out[i + m]);
        }
    }

    template <int_t N_out>
    inline void
    butterfly_radix4(gsl::span<cpx_t, N_out> out)
    const noexcept {
        auto const m = remainder(N_out);
        auto const s = stride(N_out);

        for (auto i = 0; i < m; ++i) {
            if (i > 0) {
                out[i + 1*m] = multiply_fast(out[i + 1*m], twiddles[2*i*s]);
                out[i + 2*m] = multiply_fast(out[i + 2*m], twiddles[1*i*s]);
                out[i + 3*m] = multiply_fast(out[i + 3*m], twiddles[3*i*s]);
            }
            scissors(out[i + 0*m], out[i + 1*m]);
            scissors(out[i + 2*m], out[i + 3*m]);
            scissors(out[i + 0*m], out[i + 2*m]);
            out[i + 3*m] = flip<is_inverse(Direction)>(out[i + 3*m]);
            scissors(out[i + 1*m], out[i + 3*m]);
        }
    }

    template <int_t N_out>
    inline void
    butterfly_radix8(gsl::span<cpx_t, N_out> out)
    const noexcept {
        auto const m = remainder(N_out);
        auto const s = stride(N_out);

        for (auto i = 1; i < m; ++i) {
            if (i > 0) {
                out[i + 1*m] = multiply_fast(out[i + 1*m], twiddles[4*i*s]);
                out[i + 2*m] = multiply_fast(out[i + 2*m], twiddles[2*i*s]);
                out[i + 3*m] = multiply_fast(out[i + 3*m], twiddles[6*i*s]);
                out[i + 4*m] = multiply_fast(out[i + 4*m], twiddles[1*i*s]);
                out[i + 5*m] = multiply_fast(out[i + 5*m], twiddles[5*i*s]);
                out[i + 6*m] = multiply_fast(out[i + 6*m], twiddles[3*i*s]);
                out[i + 7*m] = multiply_fast(out[i + 7*m], twiddles[7*i*s]);
            }

            scissors(out[i + 0*m], out[i + 1*m]);
            scissors(out[i + 2*m], out[i + 3*m]);
            scissors(out[i + 4*m], out[i + 5*m]);
            scissors(out[i + 6*m], out[i + 7*m]);

            out[i + 3*m] = flip<is_inverse(Direction)>(out[i + 3*m]);
            out[i + 5*m] = multiply_fast(out[i + 5*m], twiddles[m]);
            out[i + 7*m] = multiply_fast(out[i + 7*m], twiddles[m]);
            out[i + 7*m] = flip<is_inverse(Direction)>(out[i + 7*m]);

            scissors(out[i + 0*m], out[i + 2*m]);
            scissors(out[i + 1*m], out[i + 3*m]);
            scissors(out[i + 4*m], out[i + 6*m]);
            scissors(out[i + 5*m], out[i + 7*m]);

            out[i + 6*m] = flip<is_inverse(Direction)>(out[i + 6*m]);
            out[i + 7*m] = flip<is_inverse(Direction)>(out[i + 7*m]);

            scissors(out[i + 0*m], out[i + 4*m]);
            scissors(out[i + 1*m], out[i + 5*m]);
            scissors(out[i + 2*m], out[i + 6*m]);
            scissors(out[i + 3*m], out[i + 7*m]);
        }
    }

    std::array<cpx_t, N/2> twiddles;
};
}
