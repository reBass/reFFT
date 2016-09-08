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

#include <algorithm>
#include <array>
#include <complex>
#include <gsl/span>

#include "Real_FFT.hpp"

namespace reFFT {
template <typename T, int N>
/// Computes autocorrelation function of a given input
class ACF
{
    static_assert(std::is_floating_point<T>::value);
public:
    void
    operator()(gsl::span<T, N> data)
    noexcept {
        operator()(data, data);
    }

    void
    operator()(gsl::span<T const, N> input, gsl::span<T, N> output)
    noexcept {
        std::array<T, 2*N> time_domain;
        std::copy(
            std::cbegin(input),
            std::cend(input),
            std::begin(time_domain)
        );
        std::fill(
            std::end(time_domain) - N,
            std::end(time_domain),
            0
        );

        std::array<std::complex<T>, N + 1> frequency_domain;
        fft(time_domain, frequency_domain);

        std::transform(
            std::cbegin(frequency_domain),
            std::cend(frequency_domain),
            std::begin(frequency_domain),
            [] (auto const& value) {
                return std::norm(value);
            }
        );

        ifft(frequency_domain, time_domain);

        auto lag = N;
        std::transform(
            std::cbegin(time_domain),
            std::cbegin(time_domain) + N,
            std::begin(output),
            [this, &lag] (auto value) {
                return std::abs(value) / (N * lag--);
            }
        );
    }

private:
    Real_FFT<T, 2*N, FFT_Direction::Forward> const fft;
    Real_FFT<T, 2*N, FFT_Direction::Inverse> const ifft;
};
}
