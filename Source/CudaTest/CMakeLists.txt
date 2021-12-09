
cmake_policy(SET CMP0104 OLD)

add_executable(CudaTest cudatest.cu)

target_compile_options(CudaTest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIONS}>)
set_target_properties(CudaTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
