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

#include <cmath>

#include <algorithm>
#include <array>
#include <iterator>
#include <functional>

#include "common.hpp"

namespace reFFT
{
template<typename T, std::ptrdiff_t N>
class Hann_window {
    static_assert(std::is_floating_point<T>::value);
public:
    Hann_window()
    noexcept {
        encache();
    }

    template <class InputIt, class OutputIt>
    void
    cut(InputIt in, OutputIt out)
    const noexcept {
        std::transform(
            std::cbegin(cache),
            std::cend(cache),
            in,
            out,
            std::multiplies<>()
        );
    }

    static constexpr T
    norm_correction() {
        return 0.5f;
    }

private:
    void
    encache()
    noexcept {
        for (auto i = 0; i < N; ++i) {
            cache[i] = window_function(i);
        }
    }

    static constexpr T
    window_function(std::ptrdiff_t position)
    noexcept {
        auto relative_position = static_cast<T>(position) / N;
        return (1 - std::cos(relative_position * 2 * pi<T>)) / 2;
    }

    std::array<T, N> cache;
};

}
