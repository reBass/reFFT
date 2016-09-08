add_subdirectory(external/kiss_fft)
include(cmake/FindBenchmark.cmake)

add_executable(re_fft_benchmark benchmark.cpp ../include)
target_link_libraries(re_fft_benchmark re_fft)
target_link_libraries(re_fft_benchmark kiss_fft)
target_link_libraries(re_fft_benchmark ${benchmark_LIBRARIES})
