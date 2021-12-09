#include <iostream>
#include <random>

constexpr int numElements = 1e8;

void __global__ copyArray(const float* in, float* out, int num)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = id; i < num; i += stride) {
        out[i] = in[i];
    }
}

template<typename itT>
void genRandomData(itT begin, itT end) {
    std::random_device seed;
    std::default_random_engine rng(seed());
    std::uniform_real_distribution<float> dist(0, 100);
    for (auto it = begin; it != end; it++) {
        *it = dist(rng);
    }
}

int main() {
    float* input;
    float* output;
    cudaMallocManaged(&input, numElements * sizeof(float));
    cudaMallocManaged(&output, numElements * sizeof(float));

    genRandomData(input, input + numElements);

    copyArray << <4096, 256 >> > (input, output, numElements);
    cudaPeekAtLastError();
    cudaDeviceSynchronize();

    for (int i = 0; i < 10; i++)
        std::cout << output[i] << std::endl;

    cudaFree(input);
    cudaFree(output);
    return 0;
}