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

#include <array>
#include <complex>

#include <re_fft/FFT.hpp>
#include <re_fft/Real_FFT.hpp>
#include <re_fft/ACF.hpp>

#include <benchmark/benchmark.h>
#include <kiss_fft/kiss_fftr.h>
#include <kiss_fft/kiss_fft.h>

using namespace reFFT;

static void BM_FFT_float_512real(benchmark::State& state) {
    std::array<float, 512> input;
    input.fill(0);
    input[1] = 1;

    std::array<std::complex<float>, 257> output;
    Real_FFT<float, 512, FFT_Direction::Forward> fft;

    while (state.KeepRunning()) {
        fft({input}, {output});
    }
}
BENCHMARK(BM_FFT_float_512real);

static void BM_Kiss_FFT_float_512real(benchmark::State& state) {
    std::array<float, 512> input;
    input.fill(0);
    input[1] = 1;

    std::array<std::complex<float>, 257> output;
    auto plan = kiss_fftr_alloc(512, false, nullptr, nullptr);

    while (state.KeepRunning()) {
        kiss_fftr(plan, input.data(), reinterpret_cast<kiss_fft_cpx*>(output.data()));
    }
}
BENCHMARK(BM_Kiss_FFT_float_512real);

static void BM_FFT_float_512cpx_bi_direct(benchmark::State& state) {
    std::array<std::complex<float>, 512> input;
    input.fill(0);
    input[1] = 1;

    std::array<std::complex<float>, 512> output;
    FFT<float, 512, FFT_Direction::Forward> fft;
    FFT<float, 512, FFT_Direction::Inverse> ifft;

    while (state.KeepRunning()) {
        fft(input, output);
        ifft(output, input);
        input[1] = 1;
    }
}
BENCHMARK(BM_FFT_float_512cpx_bi_direct);

static void BM_Kiss_FFT_float_512cpx_bi_direct(benchmark::State& state) {
    std::array<std::complex<float>, 512> input;
    input.fill(0);
    input[1] = 1;

    std::array<std::complex<float>, 512> output;
    auto plan = kiss_fft_alloc(512, false, nullptr, nullptr);
    auto inv_plan = kiss_fft_alloc(512, true, nullptr, nullptr);

    while (state.KeepRunning()) {
        kiss_fft(
            plan,
            reinterpret_cast<kiss_fft_cpx const*>(input.data()),
            reinterpret_cast<kiss_fft_cpx*>(output.data())
        );
        kiss_fft(
            inv_plan,
            reinterpret_cast<kiss_fft_cpx const*>(output.data()),
            reinterpret_cast<kiss_fft_cpx*>(input.data())
        );
        input[1] = 1;
    }
}
BENCHMARK(BM_Kiss_FFT_float_512cpx_bi_direct);

static void BM_ACF_float_512(benchmark::State& state) {
    std::array<float, 512> input;
    input.fill(0);
    input[1] = 1;

    std::array<float, 512> output;
    ACF<float, 512> acf;

    while (state.KeepRunning()) {
        acf(input, output);
        input[1] = 1;
    }
}
BENCHMARK(BM_ACF_float_512);

BENCHMARK_MAIN();
