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
#include <tuple>
#include <utility>
#include <gsl/span>

namespace reFFT {
enum class FFT_Direction: bool {
    Forward = false,
    Inverse = true
};

template<typename T> constexpr T pi = T(3.141592653589793238462643L);

template <typename T>
constexpr std::complex<T>
multiply_fast(std::complex<T> a, std::complex<T> b)
noexcept {
    return {
        a.real() * b.real() - a.imag() * b.imag(),
        a.real() * b.imag() + a.imag() * b.real()
    };
}

template <bool IsInverse, typename T>
constexpr std::complex<T>
flip(std::complex<T> const value)
noexcept {
    return IsInverse
           ? std::complex<T>{ -value.imag(), value.real() }
           : std::complex<T>{ value.imag(), -value.real() };
}

template <typename T>
constexpr void
scissors(std::complex<T>& a, std::complex<T>& b) {
    std::tie(a, b) = std::make_pair(a + b, a - b);
}

constexpr bool
is_inverse(FFT_Direction const direction) {
    return direction == FFT_Direction::Inverse;
}


template <
    std::ptrdiff_t Count,
    typename ElementType,
    std::ptrdiff_t Extent,
    typename = std::enable_if_t<std::is_const<ElementType>::value>
>
constexpr gsl::span<ElementType const, Count>
subspan(const gsl::span<ElementType, Extent>& s, std::ptrdiff_t offset) {
    return {
        s.data() + offset,
        Count == gsl::dynamic_extent ? s.size() - offset : Count
    };
}

template <
    std::ptrdiff_t Count,
    typename ElementType,
    std::ptrdiff_t Extent,
    typename = std::enable_if_t<!std::is_const<ElementType>::value>
>
constexpr gsl::span<ElementType, Count>
subspan(const gsl::span<ElementType, Extent>& s, std::ptrdiff_t offset) {
    Expects((offset == 0 || (offset > 0 && offset <= s.size())) &&
            (Count == dynamic_extent || (Count >= 0 && offset + Count <= s.size())));
    return {
        s.data() + offset,
        Count == gsl::dynamic_extent ? s.size() - offset : Count
    };
}

template <
    std::ptrdiff_t Offset,
    std::ptrdiff_t Count,
    typename ElementType,
    std::ptrdiff_t Extent,
    typename = std::enable_if_t<std::is_const<ElementType>::value>
>
constexpr gsl::span<ElementType const, Count>
subspan(const gsl::span<ElementType, Extent>& s) {
    return {
        s.data() + Offset,
        Count == gsl::dynamic_extent ? s.size() - Offset : Count
    };
}

template <
    std::ptrdiff_t Offset,
    std::ptrdiff_t Count,
    typename ElementType,
    std::ptrdiff_t Extent,
    typename = std::enable_if_t<!std::is_const<ElementType>::value>
>
constexpr gsl::span<ElementType, Count>
subspan(const gsl::span<ElementType, Extent>& s) {
    Expects((Offset == 0 || (Offset > 0 && Offset <= s.size())) &&
            (Count == dynamic_extent || (Count >= 0 && Offset + Count <= s.size())));
    return {
        s.data() + Offset,
        Count == gsl::dynamic_extent ? s.size() - Offset : Count
    };
}
}
