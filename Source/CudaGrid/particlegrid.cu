#include <iostream>
#include <random>
#include <cub/cub.cuh>
#include <chrono>

#include "../Shared/cudaErrorCheck.h"

typedef unsigned int uint;
constexpr int numElements = int(1e4);

template<typename itT>
void genRandomData(itT begin, itT end, int maxSize) {
    std::random_device seed;
    std::default_random_engine rng(seed());
    std::uniform_real_distribution<float> dist(0, maxSize);
    for (auto it = begin; it != end; it++) {
        *it = make_float3(dist(rng),dist(rng),dist(rng));
    }
}

__constant__ float3 worldOrigin;
__constant__ float3 cellSize;
__constant__ uint3  gridSize;

// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floorf((p.x - worldOrigin.x) / cellSize.x);
    gridPos.y = floorf((p.y - worldOrigin.y) / cellSize.y);
    gridPos.z = floorf((p.z - worldOrigin.z) / cellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (gridSize.y-1);
    gridPos.z = gridPos.z & (gridSize.z-1);
    return gridPos.z * gridSize.y * gridSize.x + gridPos.y * gridSize.x + gridPos.x;
}

// calculate grid hash value for each particle
void __global__ calcHashD(uint  *gridParticleHash,  uint   *gridParticleIndex, float3 *pos, uint numParticles)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numParticles){
        return;
    }
    volatile float3 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

//Round a / b to nearest higher integer value
uint iDivUp(uint a, uint b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
    numThreads = min(blockSize, n);
    numBlocks = iDivUp(n, numThreads);
}


void printParticles(float3 * particles, int n){
    for (int i = 0; i < n; i++)
        std::cout << "Particle "<<particles[i].x<<", "<<particles[i].y<<", "<<particles[i].z<<std::endl;
}

void calcHash(uint  *gridParticleHash,
                uint  *gridParticleIndex,
                float *pos,
                int    numParticles)
{
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);
    std::cout<<"Num Blocks: "<<numBlocks<<", Num Threads: "<<numThreads<<std::endl;
    // printParticles((float3 *)pos, 30);
    // execute the kernel
    calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,gridParticleIndex,(float3 *) pos, numParticles);

    gpuErrchk( cudaPeekAtLastError());
}

void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint* dSortedHash, uint* dSortedIndex, uint numParticles)
{
    // Determine temporary device storage requirements
    // this is done by passing 0 as the temp storage
    void     *tempStorage_d = NULL;
    size_t   tempStorageSize = 0;
    gpuErrchk( cub::DeviceRadixSort::SortPairs(tempStorage_d, tempStorageSize, dGridParticleHash,dSortedHash, dGridParticleIndex,dSortedIndex, numParticles));

    // Allocate temporary storage
    gpuErrchk( cudaMalloc(&tempStorage_d, tempStorageSize));
    // Run sorting operation
    gpuErrchk( cub::DeviceRadixSort::SortPairs(tempStorage_d, tempStorageSize, dGridParticleHash, dSortedHash, dGridParticleIndex, dSortedIndex, numParticles));
}

int main() {

    float3 hostWorldOrigin = make_float3(0.f,0.f,0.f);
    gpuErrchk(cudaMemcpyToSymbol(worldOrigin, &hostWorldOrigin, sizeof(float3)));
    float3 hostCellSize = make_float3(2.f,2.f,2.f);
    gpuErrchk(cudaMemcpyToSymbol(cellSize, &hostCellSize, sizeof(float3)));
    uint3  hostGridSize = make_uint3(64,64,64); // must be power of 2
    gpuErrchk(cudaMemcpyToSymbol(gridSize, &hostGridSize, sizeof(uint3)));

    float3* particles;
    float3* sortedParticles;
    gpuErrchk(cudaMallocManaged(&particles, numElements * sizeof(float3)));
    gpuErrchk(cudaMallocManaged(&sortedParticles, numElements * sizeof(float3)));

    genRandomData(particles, particles + numElements, hostGridSize.x*hostCellSize.x); //assumes all dims to be same size
    // printParticles(particles, 30);

    uint* m_dGridParticleHash;
    uint* m_dGridParticleIndex;
    gpuErrchk(cudaMallocManaged(&m_dGridParticleHash, numElements * sizeof(uint)));
    gpuErrchk(cudaMallocManaged(&m_dGridParticleIndex, numElements * sizeof(uint)));

    calcHash(m_dGridParticleHash,m_dGridParticleIndex, (float *) particles, numElements);
    // int *num_out;
    // gpuErrchk(cudaMallocManaged(&num_out, sizeof(int)));

    gpuErrchk( cudaDeviceSynchronize());
    std::cout<<"------------\n";
    for (int i = 0; i < 100; i++){
        std::cout << "Particle "<<particles[i].x<<", "<<particles[i].y<<", "<<particles[i].z<<", "<<"Hash: "<<m_dGridParticleHash[i]<<", Index: "<<m_dGridParticleIndex[i]<<std::endl;
    }

    uint* m_dSortedParticleHash;
    uint* m_dSortedParticleIndex;
    gpuErrchk(cudaMallocManaged(&m_dSortedParticleHash, numElements * sizeof(uint)));
    gpuErrchk(cudaMallocManaged(&m_dSortedParticleIndex, numElements * sizeof(uint)));

    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_dSortedParticleHash, m_dSortedParticleIndex, numElements);

    gpuErrchk( cudaDeviceSynchronize());
    std::cout<<"------AFTER SORTING------\n";
    for (int i = 0; i < 100; i++){
        std::cout << "Particle "<<particles[m_dSortedParticleIndex[i]].x<<", "<<particles[m_dSortedParticleIndex[i]].y<<", "<<particles[m_dSortedParticleIndex[i]].z
                    <<", "<<"Hash: "<<m_dSortedParticleHash[i]<<", Index: "<<m_dSortedParticleIndex[i]<<std::endl;
    }

    cudaFree(particles);
    cudaFree(sortedParticles);
    return 0;
}