#include <iostream>
#include <random>
#include <cub/cub.cuh>
#include <chrono>
#include <cooperative_groups.h>

#include "particlegrid.cuh"
#include "../Shared/cudaErrorCheck.h"

namespace cg = cooperative_groups;


constexpr int numElements = int(1e5);

template<typename itT>
void genRandomData(itT begin, itT end, int maxSize) {
    std::random_device seed;
    std::default_random_engine rng(seed());
    std::uniform_real_distribution<float> dist(0, maxSize);
    for (auto it = begin; it != end; it++) {
        *it = make_float3(dist(rng),dist(rng),dist(rng));
    }
}

inline __host__ __device__ int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
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
void __global__ calcHashD(uint  *gridParticleHash,  uint   *gridParticleIndex, Particle *particles, uint numParticles)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numParticles){
        return;
    }
    volatile float3 p = particles[index].position;

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint   *cellStart,        // output: cell start index
                                  uint   *cellEnd,          // output: cell end index
                                  Particle *sortedParticles,        // output: sorted particles
                                  uint   *gridParticleHash, // input: sorted grid hashes
                                  uint   *gridParticleIndex,// input: sorted particle indices
                                  Particle *oldParticles,       // input: particle array
                                  uint    numParticles)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = blockIdx.x *blockDim.x + threadIdx.x;

    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x+1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index-1];
        }
    }

    cg::sync(cta);

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

        // Now use the sorted index to reorder the pos and vel data
        uint sortedIndex = gridParticleIndex[index];
        Particle pos = oldParticles[sortedIndex];
        // float4 vel = oldVel[sortedIndex];

        sortedParticles[index] = pos;
        // sortedVel[index] = vel;
    }


}

__global__
void calcSphD(Particle *particleArray,
            Particle *oldParticles, // sorted particle array
            uint* cellStart,
            uint* cellEnd,
            uint* gridParticleIndex,
            uint numParticles)
    {
        uint index = blockIdx.x *blockDim.x + threadIdx.x;

        if (index >= numParticles) return;

        Particle part = oldParticles[index]; // Particle for which sph equations need to be computed

        // get address in grid
        int3 gridPos = calcGridPos(part.position);
    	
        //TODO: initialize values to be computed
        float density = 0.f;

        // go over all surrounding cells
        for (int z=-1; z<=1; z++)
        {
            for (int y=-1; y<=1; y++)
            {
                for (int x=-1; x<=1; x++)
                {
                    int3 neighbourPos = gridPos + make_int3(x, y, z);
                    uint gridHash = calcGridHash(neighbourPos);

                    uint startIndex = cellStart[gridHash];

                    if(startIndex != 0xffffffff) // cell not empty
                    {
                        uint endIndex = cellEnd[gridHash];

                        // TODO: Calculate SPH equations here
                        for(uint j = startIndex; j < endIndex; j++)
                        {
                            if(j != index) // exclude the particle itself from neighbors
                            {
                                Particle neighborParticle = oldParticles[j];

                                // density += neighborParticle.mass * ...

                            }
                        }
                    }
                }
            }
        }

        uint originalIndex = gridParticleIndex[index];
        // TODO: Update particle properties in the original array
        particleArray[originalIndex].density = density;
    }   

int3 calcGridPos(float3 p, float3 worldOrigin, float3 cellSize)
{
    int3 gridPos;
    gridPos.x = floorf((p.x - worldOrigin.x) / cellSize.x);
    gridPos.y = floorf((p.y - worldOrigin.y) / cellSize.y);
    gridPos.z = floorf((p.z - worldOrigin.z) / cellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
uint calcGridHash(int3 gridPos, uint3 gridSize)
{
    gridPos.x = gridPos.x & (gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (gridSize.y-1);
    gridPos.z = gridPos.z & (gridSize.z-1);
    return gridPos.z * gridSize.y * gridSize.x + gridPos.y * gridSize.x + gridPos.x;
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


void getNeighbors(float3 pos, uint* dSortedHash, uint* dSortedIndex, std::vector<uint> &neighborIndex, uint numParticles){
    // uint numThreads, numBlocks;
    // computeGridSize(numParticles, 64, numBlocks, numThreads);
    
    float3 cellSize = make_float3(2.f,2.f,2.f);
    uint3  gridSize = make_uint3(64,64,64);

    int3 gridPos;
    gridPos.x = floorf((pos.x) / cellSize.x);
    gridPos.y = floorf((pos.y) / cellSize.y);
    gridPos.z = floorf((pos.z) / cellSize.z);

    printf("(%d,%d,%d)\n", gridPos.x, gridPos.y, gridPos.z);
    for (int z=-1; z<=1; z++)
    {
        for (int y=-1; y<=1; y++)
        {
            for (int x=-1; x<=1; x++)
            {
                int3 neighborPos = gridPos + make_int3(x,y,z);
                
                neighborPos.x = neighborPos.x & (gridSize.x-1);  // wrap grid, assumes size is power of 2
                neighborPos.y = neighborPos.y & (gridSize.y-1);
                neighborPos.z = neighborPos.z & (gridSize.z-1);
                uint neighborHash = neighborPos.z * gridSize.y * gridSize.x + neighborPos.y * gridSize.x + neighborPos.x;
                // printf("\nNeighbor hash %d\n", neighborHash);
                for(uint i = 0; i<numParticles;i++){
                    if(dSortedHash[i] > neighborHash)
                        break;
                    else if(dSortedHash[i] == neighborHash){
                        // printf(" %d ", dSortedIndex[i]);
                        neighborIndex.push_back(dSortedIndex[i]);
                    }
                }
            }
        }
    }
}

void getSortedNeighbors(float3 pos, uint* cellStart, uint* cellEnd, std::vector<uint> &neighborIndex, uint numParticles){
    // uint numThreads, numBlocks;
    // computeGridSize(numParticles, 64, numBlocks, numThreads);
    
    float3 cellSize = make_float3(2.f,2.f,2.f);
    uint3  gridSize = make_uint3(64,64,64);

    int3 gridPos;
    gridPos.x = floorf((pos.x) / cellSize.x);
    gridPos.y = floorf((pos.y) / cellSize.y);
    gridPos.z = floorf((pos.z) / cellSize.z);

    printf("(%d,%d,%d)\n", gridPos.x, gridPos.y, gridPos.z);
    for (int z=-1; z<=1; z++)
    {
        for (int y=-1; y<=1; y++)
        {
            for (int x=-1; x<=1; x++)
            {
                int3 neighborPos = gridPos + make_int3(x,y,z);
                
                neighborPos.x = neighborPos.x & (gridSize.x-1);  // wrap grid, assumes size is power of 2
                neighborPos.y = neighborPos.y & (gridSize.y-1);
                neighborPos.z = neighborPos.z & (gridSize.z-1);
                uint neighborHash = neighborPos.z * gridSize.y * gridSize.x + neighborPos.y * gridSize.x + neighborPos.x;
                // printf("\nNeighbor hash %d\n", neighborHash);
                uint startIndex = cellStart[neighborHash];
                if(startIndex != 0xffffffff){
                    for(uint i = cellStart[neighborHash]; i < cellEnd[neighborHash];i++){
                        neighborIndex.push_back(i);
                        // printf(" %d ", i);
                    }
                }
             
            }
        }
    }
}



ParticleSystem::ParticleSystem(uint numParticles, float3 worldOrigin, uint3 gridSize, float h):
    m_numParticles(numParticles),
    m_particleArray(0),
    m_worldOrigin(worldOrigin),
    m_sortedParticleArray(0),
    m_gridSize(gridSize),
    m_cellSize(make_float3(2*h,2*h,2*h))
{
    m_numGridCells = gridSize.x * gridSize.y * gridSize.z;
    _init(numParticles);
}

ParticleSystem::~ParticleSystem(){
    _free();
}

void ParticleSystem::calcHash(uint  *gridParticleHash,
                uint  *gridParticleIndex,
                Particle *particles,
                int    numParticles)
{
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);
    std::cout<<"Num Blocks: "<<numBlocks<<", Num Threads: "<<numThreads<<std::endl;
    // printParticles((float3 *)pos, 30);
    // execute the kernel
    calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,gridParticleIndex, particles, numParticles);

    gpuErrchk( cudaPeekAtLastError());
}

void ParticleSystem::sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint* dSortedHash, uint* dSortedIndex, uint numParticles)
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


void ParticleSystem::reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     Particle *sortedParticles,
                                    //  float *sortedPos,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                    //  float *oldPos,
                                     Particle *oldParticles,
                                     uint   numParticles,
                                     uint   numCells)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // set all cells to empty
        gpuErrchk(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

        uint smemSize = sizeof(uint)*(numThreads+1);
        reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
            cellStart,
            cellEnd,
            sortedParticles,
            gridParticleHash,
            gridParticleIndex,
            oldParticles,
            numParticles);

    }

void ParticleSystem::calcSph(Particle *particleArray, //write new properties to this array
             Particle *sortedParticles,
             uint* cellStart,
             uint* cellEnd,
             uint* gridParticleIndex,
             uint numParticles
             )         
    {

        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

                // execute the kernel
        calcSphD<<< numBlocks, numThreads >>>(particleArray,
                                              sortedParticles,
                                              cellStart,
                                              cellEnd,
                                              gridParticleIndex,
                                              numParticles);

    }

void ParticleSystem::update(){

    //TODO: 
    //timeIntegration()

    calcHash(m_dGridParticleHash, m_dGridParticleIndex, m_particleArray, m_numParticles);

    gpuErrchk( cudaDeviceSynchronize());
    std::cout<<"------------\n";
    for (int i = 0; i < 100; i++){
        float3 p = m_particleArray[i].position;
        std::cout << "Particle "<<p.x<<", "<<p.y<<", "<<p.z<<", "
        <<"Hash: "<<m_dGridParticleHash[i]<<
        ", Index: "<<m_dGridParticleIndex[i]<<std::endl;
    }


    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_dSortedParticleHash, m_dSortedParticleIndex, m_numParticles);

    gpuErrchk( cudaDeviceSynchronize());
    std::cout<<"------ HASHING ------\n";
    for (int i = 0; i < 100; i++){
        float3 p = m_particleArray[m_dSortedParticleIndex[i]].position;
        std::cout << "Particle "<<p.x<<", "<<p.y<<", "<<p.z
                    <<", "<<"Hash: "<<m_dSortedParticleHash[i]<<", Index: "<<m_dSortedParticleIndex[i]<<std::endl;
    }


    reorderDataAndFindCellStart(m_cellStart,
                                m_cellEnd,
                                m_sortedParticleArray,
                                m_dSortedParticleHash,
                                m_dSortedParticleIndex,
                                m_particleArray,
                                m_numParticles,
                                m_numGridCells);

    gpuErrchk( cudaDeviceSynchronize());
    std::cout<<"------ REORDERING AND SORTING ------\n";
    for (int i = 0; i < 100; i++){
        float3 p = m_sortedParticleArray[i].position;
        int3 gridPos = calcGridPos(p, m_worldOrigin, m_cellSize);
        uint gridHash = calcGridHash(gridPos, m_gridSize);
        std::cout << "Particle "<<p.x<<", "<<p.y<<", "<<p.z
                    <<", GridPos: "<<gridPos.x<<", "<<gridPos.y<<", "<<gridPos.z<<", Hash: "<<gridHash
                    <<", Cell Start: "<<m_cellStart[gridHash]<<", Cell End: "<<m_cellEnd[gridHash]
                    <<", Index: "<<m_dSortedParticleIndex[i]<<std::endl;
    }

    //TODO: call calcSph

}

void ParticleSystem::checkNeighbors(uint index){

    Particle testParticle = m_particleArray[index];
    float3 p = testParticle.position;
    std::cout<<" ### NEIGHBORS FOR PARTICLE "<<p.x<<","<<p.y<<","<<p.z<<" ###\n"<<std::endl;
    
    std::vector<uint> neighbors;

    getNeighbors(p, m_dSortedParticleHash, m_dSortedParticleIndex, neighbors, numElements);
    printf("Without sorting: \n");
    for(int i = 0; i < neighbors.size(); i++){
        uint index = neighbors[i];
        float3 neighPos = m_particleArray[index].position;
        printf("Neighbor %d, pos (%f,%f,%f)\n", index, neighPos.x, neighPos.y, neighPos.z);
    }

    neighbors.clear();
    getSortedNeighbors(p, m_cellStart, m_cellEnd, neighbors, numElements);
    printf("\nWith sorting: \n");
    for(int i = 0; i < neighbors.size(); i++){
        uint index = neighbors[i];
        float3 neighPos = m_sortedParticleArray[index].position;
        printf("Neighbor %d, pos (%f,%f,%f)\n", index, neighPos.x, neighPos.y, neighPos.z);
    }
}

void ParticleSystem::_initParticles(int numParticles){
    std::random_device seed;
    std::default_random_engine rng(seed());
    std::uniform_real_distribution<float> pos(0, m_gridSize.x*m_cellSize.x); //assumes all dims to be same size
    
    for (auto it = m_particleArray; it != m_particleArray + numParticles; it++) {
        it->position = make_float3(pos(rng),pos(rng),pos(rng));
        it->velocity = make_float3(0.f,0.f,0.f);
        it->mass = 1.f;
    }
}

void ParticleSystem::_init(int numParticles){
   

    gpuErrchk(cudaMallocManaged(&m_particleArray, numParticles * sizeof(Particle)));
    gpuErrchk(cudaMallocManaged(&m_sortedParticleArray, numParticles * sizeof(Particle)));
    _initParticles(numParticles);

    // gpuErrchk(cudaMallocManaged(&m_particles, numParticles * sizeof(float3)));
    // gpuErrchk(cudaMallocManaged(&m_sortedParticles, numParticles * sizeof(float3)));
    // genRandomData(m_particles, m_particles + numParticles, m_gridSize.x*m_cellSize.x); //assumes all dims to be same size
    
    gpuErrchk(cudaMallocManaged(&m_dGridParticleHash, numParticles * sizeof(uint)));
    gpuErrchk(cudaMallocManaged(&m_dGridParticleIndex, numParticles * sizeof(uint)));

    gpuErrchk(cudaMallocManaged(&m_dSortedParticleHash, numParticles * sizeof(uint)));
    gpuErrchk(cudaMallocManaged(&m_dSortedParticleIndex, numParticles * sizeof(uint)));

    gpuErrchk(cudaMallocManaged(&m_cellStart, m_numGridCells * sizeof(uint)));
    gpuErrchk(cudaMallocManaged(&m_cellEnd, m_numGridCells * sizeof(uint)));
}

void ParticleSystem::_free(){
    cudaFree(m_particleArray);
    cudaFree(m_sortedParticleArray);
    cudaFree(m_dGridParticleHash);
    cudaFree(m_dGridParticleIndex);
    cudaFree(m_dSortedParticleHash);
    cudaFree(m_dSortedParticleIndex);
    cudaFree(m_cellStart);
    cudaFree(m_cellEnd);
}

int main() {

    float3 hostWorldOrigin = make_float3(0.f,0.f,0.f);
    gpuErrchk(cudaMemcpyToSymbol(worldOrigin, &hostWorldOrigin, sizeof(float3)));
    float3 hostCellSize = make_float3(2.f,2.f,2.f);
    gpuErrchk(cudaMemcpyToSymbol(cellSize, &hostCellSize, sizeof(float3)));
    uint3  hostGridSize = make_uint3(64,64,64); // must be power of 2
    gpuErrchk(cudaMemcpyToSymbol(gridSize, &hostGridSize, sizeof(uint3)));

    ParticleSystem* psystem = new ParticleSystem(numElements, hostWorldOrigin, hostGridSize, 1);

    psystem->update();

    //for testing purposes
    psystem->checkNeighbors(5);
    return 0;
}