#include <iostream>
#include <iomanip>
#include <random>
#include <cub/util_allocator.cuh>
#include <cub/cub.cuh>
#include <chrono>
#include <cooperative_groups.h>

#include "particlegrid.cuh"

#include "imgui/imgui.h"

#define EPS 1e-12

namespace cg = cooperative_groups;

cub::CachingDeviceAllocator  g_allocator(true);

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
__constant__ float smoothingRadius;
__constant__ float3 gravity;
__constant__ float restingDensity;
__constant__ float mu; // how viscous is the fluid
__constant__ float V_i; // Volume of a particle
__constant__ float gamma1;
__constant__ float gamma2;

__global__ void timeIntegrationD(Particle* particles,
                                float deltaTime,
                                uint numParticles,
                                BoundaryMode mode)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numParticles){
        return;
    }

    float3 pos = particles[index].position;
    float3 vel = particles[index].velocity;

    //float3 velh = particles[index].velh;
   
    // equation (1)
    float3 accel = gravity + particles[index].pressureGradient + particles[index].viscosity;

    // if(isnan(length(accel))){
    //     printf("pos before integrating: (%f,%f,%f)\n", pos.x, pos.y, pos.z);
    //     printf("vel before integrating: (%f,%f,%f)\n", vel.x, vel.y, vel.z);
    //     printf("-- Density: %f\n", particles[index].density);
    //     printf("-- pressureGradient:  (%f,%f,%f)\n", particles[index].pressureGradient.x, particles[index].pressureGradient.y, particles[index].pressureGradient.z);
    //     printf("-- viscosity:  (%f,%f,%f)\n", particles[index].viscosity.x, particles[index].viscosity.y, particles[index].viscosity.z);
    //     assert(0);
    // }
    // float3 velh = vel + particles[index].acceleration * deltaTime/2.f;
    // pos  += deltaTime * velh;
    // vel  = velh + accel * deltaTime/2.f;

    // // rosswog Strömer-Verlet
    // pos += deltaTime * vel + accel/2.f * pow(deltaTime,2);
    // vel += deltaTime*(particles[index].acceleration + accel)/2.f;

    // implicit euler
    vel += accel * deltaTime;
    pos += vel * deltaTime;


    if(mode == REFLECT || mode == BP_MOVING){
        // Reflection boundary handling
        float MIN = 0.0f;
        float MAX = cellSize.x * gridSize.x; // assumes equal dims
        const float DAMP = 0.75;
        if (pos.x < MIN){
            float tbounce = (pos.x - MIN)/vel.x;
            pos -= vel * (1-DAMP)*tbounce;

            pos.x = 2*MIN - pos.x;
            vel.x = -vel.x;

            vel *= DAMP;
        }
        if (pos.x > MAX){
            float tbounce = (pos.x - MAX)/vel.x;
            pos -= vel * (1-DAMP)*tbounce;

            pos.x = 2*MAX - pos.x;
            vel.x = -vel.x;

            vel *= DAMP;
        }
        if (pos.y < MIN){
            float tbounce = (pos.y - MIN)/vel.y;
            pos -= vel * (1-DAMP)*tbounce;

            pos.y = 2*MIN - pos.y;
            vel.y = -vel.y;

            vel *= DAMP;
        }
        // if (pos.y > MAX){ // leave commented out for open ceiling
        //     float tbounce = (pos.y - MAX)/vel.y;
        //     pos -= vel * (1-DAMP)*tbounce;

        //     pos.y = 2*MAX - pos.y;
        //     vel.y = -vel.y;

        //     vel *= DAMP;
        // }
        if (pos.z < MIN){
            float tbounce = (pos.z - MIN)/vel.z;
            pos -= vel * (1-DAMP)*tbounce;

            pos.z = 2*MIN - pos.z;
            vel.z = -vel.z;

            vel *= DAMP;
        }
        if (pos.z > MAX){
            float tbounce = (pos.z - MAX)/vel.z;
            pos -= vel * (1-DAMP)*tbounce;

            pos.z = 2*MAX - pos.z;
            vel.z = -vel.z;

            vel *= DAMP;
        }

    }

    particles[index].position = pos;
    particles[index].velocity = vel;
    particles[index].acceleration = accel;
    
}


__global__  void moveObjectD(Particle *particles, float deltaTime, float3 velocity, int numFluidParticles, int numObjectParticles){

    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numObjectParticles){
        return;
    }
    index += numFluidParticles; //offset to access boundary particles

    float3 pos = particles[index].position;
    float3 vel = velocity;
    pos += deltaTime * vel;

    float MAX = 1.5f*(cellSize.x * gridSize.x); // assumes equal dims
    float MIN = - (MAX - cellSize.x * gridSize.x);
    if(pos.x > MAX){
        pos.x = MIN + (pos.x - MAX);
    }

    particles[index].position = pos;
    particles[index].velocity = vel;
}

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
        Particle particle = oldParticles[sortedIndex];

        sortedParticles[index] = particle;
    }


}

// ##### KERNELS #####

// __host__ __device__ float kernelW(float r, float h){

//     if(r >= h)
//         return 0.f;

//     float denom = static_cast<float>(64 * M_PI * pow(h, 9));
//     float W = (315 / denom) * static_cast<float>(pow(pow(h, 2) - pow(r,2),3));

//     if(isnan(W)){
//         printf("W = %f, r = %f, h = %f.\n",W,r,h);
//         assert(0); // Error: q negative or nan
//         return -1;
//     }
//     return W;
// }

// //from rosswog2015 or sim_10_sph.pdf
__host__ __device__ float kernelW(float r, float h){ 

    float q = r/h;

    if(q >= 2.0f)
        return 0.f;

    float alpha = 1.f/static_cast<float>(4* M_PI* pow(h,3));

    if(0.f <= q && q < 1.f){

        return alpha*static_cast<float>(pow(2-q,3) - 4* pow(1-q,3));
    }
    else if(1.f <= q && q < 2.0f){

        return alpha*static_cast<float>(pow(2-q,3));
    }
    else{
        printf("q = %f, r = %f, h = %f.\n",q,r,h);
        assert(0); // Error: q negative or nan
        return -1;
    }
}


// //from 2019-EG-SPH
// __host__ __device__ float kernelW(float r, float h){ 

//     float q = r/h;

//     if(q > 1.0f)
//         return 0.f;

//     float sigma = 8.f/static_cast<float>(M_PI* pow(h,3));

//     if(0.f <= q && q <= 0.5f){

//         return sigma*static_cast<float>(6 * (pow(q,3)-pow(q,2)) + 1);
//     }
//     else if(0.5f < q && q <= 1.0f){

//         return sigma*static_cast<float>(2 * pow(1-q,3));
//     }
//     else{
//         printf("q = %f, r = %f, h = %f.\n",q,r,h);
//         assert(0); // Error: q negative or nan
//         return -1;
//     }
// }


// //springel
// __host__ __device__ float kernelW(float r, float h){ 

//     float q = r/(2*h);

//     if(q > 1.0f)
//         return 0.f;

//     float c = 8.f/static_cast<float>(M_PI);

//     if(0.f <= q && q <= 1.f/2.f){

//         return c*static_cast<float>(1 - 6 * q*q + 6 * q*q*q);
//     }
//     else if(1.f/2.f < q && q <= 1.0f){

//         return c*static_cast<float>(2*pow(1-q,3));
//     }
//     else{
//         // printf("q = %f, r = %f, h = %f.\n",q,r,h);
//         assert(0); // Error: q negative or nan
//         return -1;
//     }
// }

// __host__ __device__ float3 kernelNablaW(float3 xij, float h){
//     float r = length(xij);

//     if(r >= h){
//         return make_float3(0.f,0.f,0.f);
//     }
//     return(-45 / (M_PI * pow(h,6))) * pow(h - r,2) * xij/r;
// }

__host__ __device__ float3 kernelNablaW(float3 xij, float h){

    float r = length(xij);
    float q = r/h;
    float W = 0.f;

    if(q >= 2.0f)
        return make_float3(0.f,0.f,0.f);

    float alpha = 1.f/static_cast<float>(4* M_PI* pow(h,3));

    if(0.f <= q && q < 1.f){

        W = static_cast<float>(-3*pow(2-q,2) + 12* pow(1-q,2));
    }
    else if(1.f <= q && q < 2.0f){

        W = static_cast<float>(-3*pow(2-q,2));
    }
    else{
        assert(0); // Error: q negative
    }

    return alpha * W * xij/(r*h);
}

// // derivative from 2019-EG-SPH
// __host__ __device__ float3 kernelNablaW(float3 xij, float h){

//     float r = length(xij);
//     float q = r/h;
//     float W = 0.f;

//     if(q > 1.0f)
//         return make_float3(0.f,0.f,0.f);

//     float sigma = 8.f/static_cast<float>(M_PI* pow(h,3));

//     if(0.f <= q && q <= 0.5f){

//         W = static_cast<float>(6*(3*pow(q,2) - 2*q));
//     }
//     else if(0.5f < q && q <= 1.0f){

//         W = static_cast<float>(-6*pow(1-q,2));
//     }
//     else{
//         assert(0); // Error: q negative
//     }

//     return sigma * W * xij/(r*h);
// }

// #### SPH EQUATIONS ####

__device__ float sphDensity(uint* cellStart,
                uint* cellEnd,
                int3 gridPos,
                Particle *particles,
                uint index,
                uint originalIndex){

        float sumWFluid = 0.f;
        float sumWBound = 0.f;

        Particle particle = particles[index];
        // if(originalIndex == 0){
        //     printf("in density equation for Particle (%f,%f,%f) with density %f\n", particle.position.x,particle.position.y,particle.position.z,particle.density);
        // }
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

                        for(uint j = startIndex; j < endIndex; j++)
                        {
                            if(j != index) // exclude the particle itself from neighbors
                            {
                                Particle neighborParticle = particles[j];

                                float3 dVec = particle.position - neighborParticle.position;
                                float dist = length(dVec);

                                if(dist > cellSize.x){
                                    //Dismiss
                                    continue;
                                }
                                // if(isnan(dist)){

                                //     printf("DENSITY at pos (%f,%f,%f) -- neighbor (%f,%f,%f). Distance: %f, density %f: \n", particle.position.x, particle.position.y, particle.position.z, neighborParticle.position.x,neighborParticle.position.y,neighborParticle.position.z, dist, neighborParticle.density);
                                //     if(neighborParticle.isBoundary){
                                //         printf("-- BOUNDARY\n");
                                //     }
                                // }
                                float W = kernelW(dist,smoothingRadius);

                                if(neighborParticle.isBoundary){
                                    sumWBound += neighborParticle.mass * W;
                                }
                                else{
                                    sumWFluid += W;
                                }
                                // if(originalIndex == 0){

                                //     printf("DENSITY -- neighbor (%f,%f,%f). Distance: %f, density %f: kernel W = %f -> Contributing %f\n", neighborParticle.position.x,neighborParticle.position.y,neighborParticle.position.z, dist, neighborParticle.density, W, particle.mass*W);
                                //     if(neighborParticle.isBoundary){
                                //         printf("-- BOUNDARY\n");
                                //     }
                                // }
                            
                            }
                        }

                    }
                }
            }
        }
        float density = (particle.mass * sumWFluid + gamma1 * sumWBound);
        // if(isnan(density)){
        //     assert(0);
        // }
        // float density = (particle.mass * sumWFluid + sumWBound);
        // float density = (particle.mass * sumWFluid + particle.mass * sumWBound);
        // float density = particle.mass * sumWFluid;
        return density;
}


__device__ void sphPGradVisc(
                uint* cellStart,
                uint* cellEnd,
                int3 gridPos,
                Particle *particles,
                uint index,
                uint originalIndex,
                float3 &pGradient,
                float3 &viscosity){

        // for Gradient
        float3 sumFluid = make_float3(0.0f,0.0f,0.0f);
        float3 sumNablaWb = make_float3(0.0f,0.0f,0.0f);

        Particle particle = particles[index];

        float pi = 0.f;
        if(particle.density > EPS){
            pi = (particle.pressure/ pow(particle.density,2));
        }

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

                        for(uint j = startIndex; j < endIndex; j++)
                        {
                            if(j != index) // exclude the particle itself from neighbors
                            {
                                Particle neighborParticle = particles[j];

                                float3 xij = particle.position - neighborParticle.position;
                                float r = length(xij);

                                if(r > cellSize.x || r < EPS){
                                    //Dismiss
                                    continue;
                                }

                                float3 NablaW = kernelNablaW(xij, smoothingRadius);

                                if(neighborParticle.isBoundary){

                                    sumNablaWb += neighborParticle.mass * NablaW;
                                }
                                else{

                                    float pj = 0.f;
                                    if(neighborParticle.density > EPS){
                                        pj = (neighborParticle.pressure/ pow(neighborParticle.density,2));
                                    }
                                    sumFluid += (pi + pj) * NablaW;

                                    // if(isnan(length(sumFluid))){
                                    //     printf("NaN SumFluid: pi = %f, pj = %f, NablaW = (%f,%f,%f)\n", pi, pj, NablaW.x,NablaW.y,NablaW.z);
                                    //     printf("Neighbor: pressure = %f, Density = %f\n", neighborParticle.pressure, neighborParticle.density);
                                    //     assert(0);
                                    // }
                                }

                                float3 vij = particle.velocity - neighborParticle.velocity;
                                if(neighborParticle.density > EPS){
                                    viscosity += (neighborParticle.mass/neighborParticle.density) * (dot(vij,xij)/(dot(xij,xij) + 0.01*pow(smoothingRadius,2))) * NablaW;
                                }
                                
                                // if(originalIndex == 0){
                                //     printf("adding (%f,%f,%f).\n", add.x,add.y,add.z);
                                // }
                            }
                        }
                    }
                }
            }
        }
        float3 gradientFluid = - particle.mass * sumFluid; 
        // float3 gradientBound = -(2*pi* particle.mass * gamma2) * sumNablaWb; // one boundary layer
        float3 gradientBound = -(2*pi* gamma2) * sumNablaWb;

        pGradient = gradientFluid + gradientBound;
        viscosity = mu * 10 * viscosity;

        // if(isnan(length(pGradient))){
        //     printf("NaN Gradient: Pos (%f,%f,%f) gradientFluid = (%f,%f,%f), gradientBound = (%f,%f,%f)\n",particle.position.x,particle.position.y,particle.position.z, gradientFluid, gradientBound);
        //     assert(0);
        // }
        // if(isnan(length(viscosity))){
        //     printf("NaN viscosity: Pos (%f,%f,%f)\n",particle.position.x,particle.position.y,particle.position.z);
        //     assert(0);
        // }
}



__global__
void calcDensityPressureD(Particle *oldParticles, // sorted particle array
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
    
    uint originalIndex = gridParticleIndex[index];

    // SPH: density 
    float density = 0.0;
    density = sphDensity(cellStart, cellEnd,gridPos, oldParticles, index, originalIndex);

    // SPH: Pressure
    float K = 1481.f;
    float pressure = K * max(density - restingDensity,0.f); // CLAMPING at 0 to consider particle defiancy at the fluid surface
    
    oldParticles[index].density = density;
    oldParticles[index].pressure = pressure;
    
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
    	uint originalIndex = gridParticleIndex[index];

        float3 gradient = make_float3(0.f,0.f,0.f);
        float3 viscosity = make_float3(0.f,0.f,0.f);
        // Calculate pressure gradient and viscosity
        sphPGradVisc(cellStart, cellEnd, gridPos, oldParticles, index, originalIndex, gradient, viscosity);

        // Update particle properties in the original array
        particleArray[originalIndex].density = part.density;
        particleArray[originalIndex].pressure = part.pressure;
        particleArray[originalIndex].pressureGradient = gradient;
        particleArray[originalIndex].viscosity = viscosity;
        
    }   


__global__ void gatherPositionsD(float4* positions, Particle* particleArray, uint numParticles)
{
    uint index = blockIdx.x *blockDim.x + threadIdx.x;

    if (index >= numParticles){
        return;
    }

    positions[index] = make_float4(particleArray[index].position, 1.0);
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


// ########## PARTICLE SYSTEM CLASS ################

ParticleSystem::ParticleSystem(uint numParticles, float3 hostWorldOrigin, uint3 hostGridSize, float h):
    m_numParticles(numParticles),
    m_numBoundary(0),
    m_numAllParticles(0),
    m_particleArray(0),
    m_worldOrigin(hostWorldOrigin),
    m_sortedParticleArray(0),
    m_gridSize(hostGridSize),
    m_cellSize(make_float3(2*h,2*h,2*h)), //cell size equals kernel support
    m_uniform_mass(0.f),
    m_h(h),
    objVel(make_float3(5.f,0.f,0.f)),
    m_boundaryMode(REFLECT)
{
    gpuErrchk(cudaMemcpyToSymbol(worldOrigin, &m_worldOrigin, sizeof(float3)));
    gpuErrchk(cudaMemcpyToSymbol(smoothingRadius, &h, sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(cellSize, &m_cellSize, sizeof(float3)));
    gpuErrchk(cudaMemcpyToSymbol(gridSize, &m_gridSize, sizeof(uint3)));

    // copy equation constants to gpu
    gpuErrchk(cudaMemcpyToSymbol(gravity, &m_gravity, sizeof(float3)));
    gpuErrchk(cudaMemcpyToSymbol(restingDensity, &m_restingDensity, sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(mu, &m_mu, sizeof(float)));

    m_spacing = 0.5f*m_cellSize.x;
    m_numGridCells = hostGridSize.x * hostGridSize.y * hostGridSize.z;

    m_particleVolume = static_cast<float>(pow(m_spacing,3));
    m_fluidVolume = m_particleVolume * numParticles;
   
    gpuErrchk(cudaMemcpyToSymbol(V_i, &m_particleVolume, sizeof(float)));

    m_uniform_mass = m_restingDensity * m_particleVolume; // rho_0 * V / n


    glGenBuffers(1, &m_vbo);

    _init(numParticles);
}


void ParticleSystem::_resetProperties(Particle *it){
    it->velocity = make_float3(0.f,0.f,0.f);
    it->density = 0.f;
    it->pressure = 0.f;
    it->pressureGradient = make_float3(0.f,0.f,0.f);
    it->viscosity = make_float3(0.f,0.f,0.f);
    it->acceleration = make_float3(0.f,0.f,0.f);
}

void ParticleSystem::_initParticles(int numParticles)
{
    
    Particle* it = m_particleArray;

    m_partConstellation = "Four Pillars";

    if(strcmp(m_partConstellation, "Sphere") == 0)
    {

    }
    //4 groups of particles at each corner of the volume
    else if (strcmp(m_partConstellation, "Four Pillars") == 0)
    {
        float spacingLocal = m_spacing;
        int width = m_cellSize.x * m_gridSize.x;
        float particlesPerLine = (width / spacingLocal) + 1;
        int height = numParticles/(particlesPerLine*particlesPerLine);
        int count = 0;
        float quarterW = width / 4;
        int y = 0;
        float3 offset = make_float3(m_spacing,5*m_spacing,m_spacing);

        
        while(count < numParticles){    
            y++;

            //fill from z = beginning,x = beginning
            //float3 offset = make_float3(m_cellSize.x,2.0f,m_cellSize.z);
            for(auto z = 0; z < quarterW; z++){
                for(auto x = 0; x < quarterW; x++){
                    it->position = m_spacing*make_float3(x,y,z) + offset;
                    it->mass = m_uniform_mass;
                    it->isBoundary = false;
                    it++;
                    count++;
                }
            }
            //fill from z = end, x = beginning
            //float3 offset = make_float3(m_cellSize.x,2.0f,-m_cellSize.z);
            for (auto z = width; z > width - quarterW; z = z-1)
            {
                for (auto x = 0; x < quarterW; x++)
                {
                    it->position = m_spacing*make_float3(x,y,z) + (offset * make_float3(1.f,1.f,-1.f));
                    it->mass = m_uniform_mass;
                    it->isBoundary = false;
                    it++;
                    count++;
                }
            }
            //fill from x = beginning, x = end
            //float3 offset = make_float3(-m_cellSize.x,2.0f,m_cellSize.z);
            for (auto z = 0; z < quarterW; z++)
            {
                for (auto x = width; x > width - quarterW; x = x-1)
                {
                    it->position = m_spacing*make_float3(x,y,z) + (offset * make_float3(-1.f,1.f,1.f));
                    it->mass = m_uniform_mass;
                    it->isBoundary = false;
                    it++;
                    count++;
                }
            }
            //fill from z = end, x = end
            //float3 offset = make_float3(-m_cellSize.x,2.0f,-m_cellSize.z);
            for (auto z = width; z > width - quarterW; z = z-1)
            {
                for (auto x = width; x > width - quarterW; x = x-1)
                {
                    it->position = m_spacing*make_float3(x,y,z) + (offset * make_float3(-1.f,1.f,-1.f));
                    it->mass = m_uniform_mass;
                    it->isBoundary = false;
                    it++;
                    count++;
                }
            }
        }
    }
    //default configuration
    else
    {
        int width = 15;//;(m_cellSize.x*m_gridSize.x/m_spacing) - 1;
        int height = numParticles/(width*width);
        float3 offset = make_float3(m_spacing,5*m_spacing,m_spacing);
        // float3 offset = make_float3(5*m_spacing,m_spacing,5*m_spacing);
        int count = 0;
        for(auto y = 0; y < height; y++){
            for(auto z = 0; z < width; z++){
                for(auto x = 0; x < width; x++){
                    
                    it->position = m_spacing*make_float3(x,y,z) + offset;
                    _resetProperties(it);
                    //it->velocity = make_float3(10.f,0.f,0.f);
                    it->mass = m_uniform_mass;
                    it->isBoundary = false;
                    it++;
                    count++;
                }
            }
        }

        int rest = numParticles - (height*width*width);
        int y = height;
        for(auto i = 0; i < rest; i++){
            int x = (i%width);
            int z = (i/width);
            it->position = m_spacing*make_float3(x,y,z) + offset;
            _resetProperties(it);
            //it->velocity = make_float3(10.f,0.f,0.f);
            it->mass = m_uniform_mass;
            it->isBoundary = false;
            it++;
            count++;
        }
    }

    // dumpParticleInfo(0,1000);
}

void ParticleSystem::_initGammas(){

    std::vector<Particle> perfectNeighbors;

    Particle p;
    p.position = make_float3(0,m_spacing,0); // Particle with ideal spacing over boundary
    p.mass = m_uniform_mass;
    float3 pos = p.position;

    float kernelSupport = m_cellSize.x; // assumes all dims to be the same (here: 2*h)
    float maxExtend = m_spacing* ceil(kernelSupport/m_spacing);

    float sumWf = 0.f; //Fluid contribution
    float3 sumNablaWf = make_float3(0.f,0.f,0.f);

    for(auto x = -maxExtend; x <= maxExtend; x+=m_spacing){
        for(auto z = -maxExtend; z <= maxExtend; z+=m_spacing){
            for(auto y = 0.f + m_spacing; y <= maxExtend; y+=m_spacing){

                float3 neighborPos = make_float3(x,y,z);
                float3 xij = pos-neighborPos;
                float r = length(xij);
                if(r < EPS)
                    continue;
                sumWf += kernelW(r,m_h);
                sumNablaWf += kernelNablaW(pos-neighborPos, m_h);
            }
        }
    }

    float sumWb = 0.f; //Boundary contribution
    float3 sumNablaWb = make_float3(0.f,0.f,0.f);

    for(auto x = -maxExtend; x <= maxExtend; x+=m_spacing){
        for(auto z = -maxExtend; z <= maxExtend; z+=m_spacing){

            float3 boundaryPos = make_float3(x,0.f,z); //one y layer of boundary particles
            float3 xij = pos-boundaryPos;
            float r = length(xij);
            if(r < EPS)
                continue;
            
            float add = kernelW(r, m_h);
            sumWb += add;
            sumNablaWb += kernelNablaW(xij, m_h);
            // if(add > 0.f)
            //     printf("Gamma Boundary contrib: relative pos (%f,%f,%f) - r = %f. Adding %f to sumWb (%f)\n",x-pos.x,0.f-pos.y,z-pos.z, r, add, sumWb);
        }
    }

    float sumWbb = 0.f; //Gamma3: Boundary to boundary contrib used to calculate boundary masses
    float3 bPos = make_float3(0.f,0.f,0.f);
    for(auto x = -maxExtend; x <= maxExtend; x+=m_spacing){
        for(auto z = -maxExtend; z <= maxExtend; z+=m_spacing){

            float3 boundaryPos = make_float3(x,0.f,z); //one y layer of boundary particles
            float3 xij = bPos-boundaryPos;
            float r = length(xij);
            if(r < EPS)
                continue;
            
            float add = kernelW(r, m_h);
            sumWbb += add;
        }
    }

    float g1 = ((1.f/m_particleVolume) - sumWf)/sumWb;
    float g2 = - dot(sumNablaWf, sumNablaWb)/dot(sumNablaWb,sumNablaWb);
    m_g3 = g1 * m_particleVolume *sumWbb;

    // printf("Checking Gammas:\n");
    // float sumW = sumWf+g1*sumWb;
    // printf("Gamma1 %f: (1/Vi = %f). sumWf + g1*sumWb = %f -> must be ~ equal\n",g1, 1.f/m_particleVolume, sumW);
    // float3 constraint = sumNablaWf + g2 * sumNablaWb;
    // printf("Gamma2 %f: SumNablaWf + gamma2*SumNablaWb = (%f,%f,%f) -> must be ~ 0\n", g2,constraint.x,constraint.y,constraint.z);

    gpuErrchk(cudaMemcpyToSymbol(gamma1, &g1, sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(gamma2, &g2, sizeof(float)));
}

void ParticleSystem::_initBoundary(int extend, float spacing, BoundaryMode bMode)
{
    std::vector<float3> boundaryPos;
    int i = 0;

    if(bMode == REFLECT) // No boundary particles here
        return;

    else if(bMode == BP_BOX){
        // Ground
        for(auto x = 0; x < extend; x++){
            for(auto z = 0; z < extend; z++){
                // for(auto y = 0; y < 4; y++){
                    auto pos =  m_spacing*make_float3(x, 0.f, z);
                    boundaryPos.push_back(pos);
                    i++;
                    // it->position = m_spacing*make_float3(x, 0.f, z);
                    // //printf("Boundary particle (%f,%f,%f) added.\n",it->position.x,0.f,it->position.z);
                    // it->mass = m_uniform_mass;
                    // it->isBoundary = true;
                    // it++;
                // }
            }  
        }
        
        // Wall 1 (back)
        for(auto x = 0; x < extend; x++){
            for(auto y = 0; y < extend; y++){
                // printf("Boundary particle (%f,%f,%f) added.\n",x,y,z);
                i++;
                auto pos = m_spacing * make_float3(x, y + 1, 0.f);
                boundaryPos.push_back(pos);
            }
        }

        // Wall 2 (front)
        for(auto x = 0; x < extend; x++){
            for(auto y = 0; y < extend; y++){
                // printf("Boundary particle (%f,%f,%f) added.\n",x,y,z);
                i++;
                auto pos = m_spacing * make_float3(x, y + 1, extend - 1);
                boundaryPos.push_back(pos);
            }
        }

        // Wall 3 (left)
        for(auto y = 0; y < extend; y++){
            for(auto z = 0; z < extend; z++){
                // printf("Boundary particle (%f,%f,%f) added.\n",x,y,z);
                i++;
                auto pos = m_spacing * make_float3(0.f, y + 1, z);
                boundaryPos.push_back(pos);
            }
        }

        // Wall 4 (right)
        for(auto y = 0; y < extend; y++){
            for(auto z = 0; z < extend; z++){
                // printf("Boundary particle (%f,%f,%f) added.\n",x,y,z);
                i++;
                auto pos = m_spacing * make_float3(extend - 1, y + 1, z);
                boundaryPos.push_back(pos);
            }
        }

    }
    else if(bMode == BP_GROUND){

        float3 offset = make_float3(-m_cellSize.x*m_gridSize.x, 0.f, -m_cellSize.x*m_gridSize.x)/2.f;

        for(auto x = 0; x < extend; x++){
            for(auto z = 0; z < extend; z++){
                // for(auto y = 0; y < 4; y++){
                    auto pos =  m_spacing*make_float3(x, 0.f, z) + offset;
                    boundaryPos.push_back(pos);
                    i++;
            }  
        }
    }
    else if(bMode == BP_MOVING){
        int depth = 5;
        float3 offset = make_float3(0.f, 0.f, m_cellSize.x*m_gridSize.x)/2.f - make_float3(0,0,extend/2);

            for(auto x = 0; x < depth; x++){
                for(auto y = 0; y < extend; y++){
                    for(auto z = 0; z < extend; z++){

                        auto pos =  m_spacing*make_float3(x, y, z) + offset; //back side         
                        boundaryPos.push_back(pos);
                        i++;  
                    }
                }
            }
    }
    
    // Insert into particle array
    for(int i = 0; i < m_numBoundary; i++){

        auto pos = boundaryPos[i];
        
        //printf("Boundary particle (%f,%f,%f) added.\n",it->position.x,0.f,it->position.z);

        //Calculate Mass for boundary
        // float sumWbb = 0.0f;
        // for(int j = 0; j < m_numBoundary; j++){
        //     if(i == j)
        //         continue;

        //     float3 neighborBoundaryPos = boundaryPos[j];
        //     float r = length(neighborBoundaryPos - pos);
        //     if(r > m_cellSize.x)
        //         continue;
        //     float add = kernelW(r, m_h);
        //     sumWbb += add;
        // }
        
        // float boundaryMass = m_restingDensity * (m_g3/sumWbb);
        float boundaryMass = m_uniform_mass;

        m_particleArray[i + m_numParticles].position = pos;
        m_particleArray[i + m_numParticles].mass = boundaryMass;
        m_particleArray[i + m_numParticles].isBoundary = true;
    }
    
    printf("%u boundary particles added!\n",i);
}

void ParticleSystem::_setGLArray(uint numParticles){
    // glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    // glBufferData(GL_ARRAY_BUFFER, m_numParticles*sizeof(float4), 0, GL_DYNAMIC_DRAW);
    glBufferData(GL_ARRAY_BUFFER, numParticles*sizeof(float4), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&m_cuda_vbo_resource, m_vbo, cudaGraphicsRegisterFlagsNone));
}

void ParticleSystem::_init(int numParticles)
{
    printf("Mode = %u\n", m_boundaryMode);
    int bExtend;

    if (m_boundaryMode == REFLECT)
    {
        bExtend = 0;
        m_numBoundary = 0;
    }
    else if (m_boundaryMode == BP_BOX)
    {
        bExtend = static_cast<int>(1.0f * m_gridSize.x * m_cellSize.x / m_spacing) + 1;
        m_numBoundary = static_cast<uint>(pow(bExtend, 2)) * 5; // Boundary particles for 1 uniform ground square + 4 walls
    }
    else if (m_boundaryMode == BP_GROUND)
    {
        bExtend = static_cast<int>(2.0f * m_gridSize.x * m_cellSize.x / m_spacing) + 1;
        m_numBoundary = static_cast<uint>(pow(bExtend, 2));
    }
    else if(m_boundaryMode == BP_MOVING){
        bExtend = static_cast<int>(10/m_spacing); // here extend means the width/height of the moving object
        int depth = 5; // must be the same as in initBoundary

        // m_numBoundary = static_cast<uint>(2*pow(bExtend, 2) + (depth-2)*4*(bExtend-1)); hollow object
        m_numBoundary = static_cast<uint>(depth*pow(bExtend, 2));
    }

    m_numAllParticles = numParticles + m_numBoundary;
    uint size = (m_numAllParticles) * sizeof(Particle);

    gpuErrchk(cudaMallocManaged(&m_particleArray, size));
    gpuErrchk(cudaMallocManaged(&m_sortedParticleArray, size));

    _setGLArray(m_numAllParticles);
    _initParticles(numParticles);
    gpuErrchk( cudaDeviceSynchronize());
    _initGammas();
    _initBoundary(bExtend, m_spacing, m_boundaryMode);
    gpuErrchk( cudaDeviceSynchronize());

    gpuErrchk(cudaMallocManaged(&m_dGridParticleHash, m_numAllParticles * sizeof(uint)));
    gpuErrchk(cudaMallocManaged(&m_dGridParticleIndex, m_numAllParticles * sizeof(uint)));

    gpuErrchk(cudaMallocManaged(&m_dSortedParticleHash, m_numAllParticles * sizeof(uint)));
    gpuErrchk(cudaMallocManaged(&m_dSortedParticleIndex, m_numAllParticles * sizeof(uint)));

    gpuErrchk(cudaMallocManaged(&m_cellStart, m_numGridCells * sizeof(uint)));
    gpuErrchk(cudaMallocManaged(&m_cellEnd, m_numGridCells * sizeof(uint)));
}

void ParticleSystem::_free()
{
    gpuErrchk(cudaFree(m_particleArray));
    gpuErrchk(cudaFree(m_sortedParticleArray));
    gpuErrchk(cudaFree(m_dGridParticleHash));
    gpuErrchk(cudaFree(m_dGridParticleIndex));
    gpuErrchk(cudaFree(m_dSortedParticleHash));
    gpuErrchk(cudaFree(m_dSortedParticleIndex));
    gpuErrchk(cudaFree(m_cellStart));
    gpuErrchk(cudaFree(m_cellEnd));
    // glDeleteBuffers(1, (const GLuint *)&m_vbo);
    // gpuErrchk(cudaGraphicsUnregisterResource(m_cuda_vbo_resource));
}

ParticleSystem::~ParticleSystem(){
    _free();
    m_numParticles = 0;
}

void ParticleSystem::timeIntegration(Particle* particles,
                float deltaTime,
                int numParticles)
{
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

    timeIntegrationD<<< numBlocks, numThreads >>>(particles, deltaTime, numParticles, m_boundaryMode);

    gpuErrchk( cudaPeekAtLastError());
}

void ParticleSystem::calcHash(uint  *gridParticleHash,
                uint  *gridParticleIndex,
                Particle *particles,
                int    numParticles)
{
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);

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
    CubDebugExit(g_allocator.DeviceAllocate(&tempStorage_d, tempStorageSize));
    // Run sorting operation
    gpuErrchk( cub::DeviceRadixSort::SortPairs(tempStorage_d, tempStorageSize, dGridParticleHash, dSortedHash, dGridParticleIndex, dSortedIndex, numParticles));

    // Free temporary storage
    CubDebugExit(g_allocator.DeviceFree(tempStorage_d));
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

        auto start = std::chrono::steady_clock ::now();
        //calculate density and pressure first
        calcDensityPressureD<<< numBlocks, numThreads >>>(sortedParticles,
                                                            cellStart,
                                                            cellEnd,
                                                            gridParticleIndex,
                                                            numParticles);
        gpuErrchk( cudaPeekAtLastError());
        gpuErrchk( cudaDeviceSynchronize());

        auto duration = std::chrono::steady_clock ::now() - start;
        // std::cout << "calcSph(Density): \t Kernel took "
        // << std::chrono::duration_cast<std::chrono::milliseconds>( duration ).count() << "ms" << std::endl;

        start = std::chrono::steady_clock ::now();
        // calculate pressure gradient and viscosity
        calcSphD<<< numBlocks, numThreads >>>(particleArray,
                                              sortedParticles,
                                              cellStart,
                                              cellEnd,
                                              gridParticleIndex,
                                              numParticles);
        gpuErrchk( cudaPeekAtLastError());
        gpuErrchk( cudaDeviceSynchronize());
        duration = std::chrono::steady_clock ::now() - start;
        // std::cout << "calcSph(PGradVisc): \t Kernel took "
        // << std::chrono::duration_cast<std::chrono::milliseconds>( duration ).count() << "ms" << std::endl;

    }

void ParticleSystem::gatherPositions(float4* positions, Particle* particleArray, uint numParticles){
    // thread per particle
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 64, numBlocks, numThreads);

    gatherPositionsD<<< numBlocks, numThreads >>>(positions, particleArray, numParticles);
}

void ParticleSystem::moveObject(Particle *particleArray, float deltaTime, float3 velocity, int numFluidParticles, int numObjectParticles){
    // thread per particle
    uint numThreads, numBlocks;
    computeGridSize(numObjectParticles, 64, numBlocks, numThreads);

    moveObjectD<<< numBlocks, numThreads >>>(particleArray, deltaTime, velocity, numFluidParticles, numObjectParticles);
}


void ParticleSystem::update(float deltaTime)
{
    // std::cout<<"---------------\n";
    auto start = std::chrono::steady_clock ::now();

    calcHash(m_dGridParticleHash, m_dGridParticleIndex, m_particleArray, m_numAllParticles);
    gpuErrchk( cudaDeviceSynchronize());

    auto duration = std::chrono::steady_clock ::now() - start;
    // std::cout << "calcHash: \t\t Kernel took "
    // << std::chrono::duration_cast<std::chrono::milliseconds>( duration ).count() << "ms" << std::endl;
    
    
    // std::cout<<"------------\n";
    // for (int i = 0; i < 100; i++){
    //     float3 p = m_particleArray[i].position;
    //     std::cout << "Particle "<<p.x<<", "<<p.y<<", "<<p.z<<", "
    //     <<"Hash: "<<m_dGridParticleHash[i]<<
    //     ", Index: "<<m_dGridParticleIndex[i]<<std::endl;
    // }
    start = std::chrono::steady_clock ::now();

    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_dSortedParticleHash, m_dSortedParticleIndex, m_numAllParticles);
    gpuErrchk( cudaDeviceSynchronize());

    duration = std::chrono::steady_clock ::now() - start;
    // std::cout << "sortParticles: \t\t Kernel took "
    // << std::chrono::duration_cast<std::chrono::milliseconds>( duration ).count() << "ms" << std::endl;
    
    // std::cout<<"------ HASHING ------\n";
    // for (int i = 0; i < 100; i++){
    //     float3 p = m_particleArray[m_dSortedParticleIndex[i]].position;
    //     std::cout << "Particle "<<p.x<<", "<<p.y<<", "<<p.z
    //                 <<", "<<"Hash: "<<m_dSortedParticleHash[i]<<", Index: "<<m_dSortedParticleIndex[i]<<std::endl;
    // }
    start = std::chrono::steady_clock ::now();

    reorderDataAndFindCellStart(m_cellStart,
                                m_cellEnd,
                                m_sortedParticleArray,
                                m_dSortedParticleHash,
                                m_dSortedParticleIndex,
                                m_particleArray,
                                m_numAllParticles,
                                m_numGridCells);
   gpuErrchk( cudaDeviceSynchronize());
    duration = std::chrono::steady_clock ::now() - start;
    // std::cout << "reorderAndFindCStart: \t Kernel took "
    // << std::chrono::duration_cast<std::chrono::milliseconds>( duration ).count() << "ms" << std::endl;
    
    // std::cout<<"------ REORDERING AND SORTING ------\n";
    // for (int i = 0; i < 100; i++){
    //     float3 p = m_sortedParticleArray[i].position;
    //     int3 gridPos = calcGridPos(p, m_worldOrigin, m_cellSize);
    //     uint gridHash = calcGridHash(gridPos, m_gridSize);
    //     std::cout << "Particle "<<p.x<<", "<<p.y<<", "<<p.z
    //                 <<", GridPos: "<<gridPos.x<<", "<<gridPos.y<<", "<<gridPos.z<<", Hash: "<<gridHash
    //                 <<", Cell Start: "<<m_cellStart[gridHash]<<", Cell End: "<<m_cellEnd[gridHash]
    //                 <<", Index: "<<m_dSortedParticleIndex[i]<<std::endl;
    // }
    // start = std::chrono::steady_clock ::now();

    //TODO: call calcSph
    calcSph(m_particleArray, 
            m_sortedParticleArray,
            m_cellStart,
            m_cellEnd, 
            m_dSortedParticleIndex, 
            m_numAllParticles);
    gpuErrchk( cudaDeviceSynchronize());   
    // dumpParticleInfo(0,3);

    // duration = std::chrono::steady_clock ::now() - start;
    // std::cout << "calcSph: \t\t Kernel took "
    // << std::chrono::duration_cast<std::chrono::milliseconds>( duration ).count() << "ms" << std::endl;

    start = std::chrono::steady_clock ::now();

    timeIntegration(m_particleArray, deltaTime, m_numParticles);    
    gpuErrchk( cudaDeviceSynchronize());  
    // std::cout<<"======== AFTER TIME INTEGRATION =========\n"; 
    // dumpParticleInfo(0,1000);
    duration = std::chrono::steady_clock ::now() - start;
    // std::cout << "timeIntegration: \t Kernel took "
    // << std::chrono::duration_cast<std::chrono::milliseconds>( duration ).count() << "ms" << std::endl;

    if(m_boundaryMode == BP_MOVING){
        moveObject(m_particleArray, deltaTime, objVel, m_numParticles, m_numBoundary);
        gpuErrchk( cudaDeviceSynchronize()); 
    }


    //map m_positions to be used with cuda
    float4* m_positions;
    gpuErrchk(cudaGraphicsMapResources(1, &m_cuda_vbo_resource));
    size_t num_bytes;
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void **) &m_positions, &num_bytes, m_cuda_vbo_resource));

    gatherPositions(m_positions, m_particleArray, m_numAllParticles);
    // unmap
    gpuErrchk(cudaGraphicsUnmapResources(1, &m_cuda_vbo_resource, 0));

    drawGUIConstellation();
}


void ParticleSystem::checkNeighbors(uint index, int numElements){

    Particle testParticle = m_particleArray[index];
    float3 p = testParticle.position;
    std::cout<<" ### NEIGHBORS FOR PARTICLE "<<p.x<<","<<p.y<<","<<p.z<<" ###"<<std::endl;
    
    std::vector<uint> neighbors;

    // getNeighbors(p, m_dSortedParticleHash, m_dSortedParticleIndex, neighbors, numElements);
    // printf("Without sorting: \n");
    // for(int i = 0; i < neighbors.size(); i++){
    //     uint index = neighbors[i];
    //     float3 neighPos = m_particleArray[index].position;
    //     printf("Neighbor %d, pos (%f,%f,%f)\n", index, neighPos.x, neighPos.y, neighPos.z);
    // }
    gpuErrchk( cudaDeviceSynchronize());  
    neighbors.clear();
    getSortedNeighbors(p, neighbors, numElements);
    // printf("\nWith sorting: \n");
    // for(int i = 0; i < neighbors.size(); i++){
    //     uint index = neighbors[i];
    //     float3 neighPos = m_sortedParticleArray[index].position;
    //     printf("Neighbor %d, pos (%f,%f,%f). Dist is %f.", index, neighPos.x, neighPos.y, neighPos.z,length(p-neighPos));
    //     if(m_sortedParticleArray[index].isBoundary){
    //         printf("- BOUNDARY");
    //     }
    //     printf("\n");
        
    // }
    printf("num: %u\n", int(neighbors.size()));

    gpuErrchk( cudaDeviceSynchronize());  
}

Particle* ParticleSystem::getParticleArray() 
{
    return m_particleArray;

}

GLuint ParticleSystem::getVBO(){
    return m_vbo;
}

void ParticleSystem::getSortedNeighbors(float3 pos, std::vector<uint> &neighborIndex, uint numParticles){
    // uint numThreads, numBlocks;
    // computeGridSize(numParticles, 64, numBlocks, numThreads);

    int3 gridPos;
    gridPos.x = floorf((pos.x) / m_cellSize.x);
    gridPos.y = floorf((pos.y) / m_cellSize.y);
    gridPos.z = floorf((pos.z) / m_cellSize.z);

    printf("Position (%f,%f,%f) in GridPos (%d,%d,%d)\n", pos.x, pos.y, pos.z,gridPos.x, gridPos.y, gridPos.z);
    for (int z=-1; z<=1; z++)
    {
        for (int y=-1; y<=1; y++)
        {
            for (int x=-1; x<=1; x++)
            {
                int3 neighborPos = gridPos + make_int3(x,y,z);
                
                neighborPos.x = neighborPos.x & (m_gridSize.x-1);  // wrap grid, assumes size is power of 2
                neighborPos.y = neighborPos.y & (m_gridSize.y-1);
                neighborPos.z = neighborPos.z & (m_gridSize.z-1);
                uint neighborHash = neighborPos.z * m_gridSize.y * m_gridSize.x + neighborPos.y * m_gridSize.x + neighborPos.x;
                // printf("\nNeighbor hash %d\n", neighborHash);
                uint startIndex = m_cellStart[neighborHash];
                if(startIndex != 0xffffffff){
                    for(uint i = m_cellStart[neighborHash]; i < m_cellEnd[neighborHash];i++){
                        float3 nPos = m_sortedParticleArray[i].position;
                        if(length(pos-nPos) <= m_cellSize.x){ //compare with h
                            neighborIndex.push_back(i);
                        }
                        // printf(" %d ", i);
                    }
                }
             
            }
        }
    }
}

void ParticleSystem::dumpParticleInfo(uint start, uint end){
    gpuErrchk( cudaDeviceSynchronize());  
    for(auto i = start; i<end;i++){

        Particle p = m_particleArray[i];
        std::cout << std::setprecision(6) << "["<<i<<"]: " << "Position: (" << p.position.x <<","
                                                        <<p.position.y<<","
                                                        <<p.position.z <<")\n"
                                    << "Velocity: (" << p.velocity.x <<","
                                                        <<p.velocity.y<<","
                                                        <<p.velocity.z <<")\n"
                                    << "Density: " << p.density << "\n"
                                    << "Pressure: " <<  p.pressure << "\n"
                                    << "PressureGradient: (" << p.pressureGradient.x <<","
                                                            <<p.pressureGradient.y<<","
                                                            <<p.pressureGradient.z <<")\n"
                                    << "Viscosity: (" << p.viscosity.x <<","
                                                            <<p.viscosity.y<<","
                                                            <<p.viscosity.z <<")\n"
                                    << "Mass: "<< p.mass << "\n"
                                    << "Is Boundary: "<< p.isBoundary << std::endl;
    }
    gpuErrchk( cudaDeviceSynchronize());  
}

void ParticleSystem::resetParticles(uint numParticles, BoundaryMode bMode)
{
    gpuErrchk(cudaDeviceSynchronize());
    m_boundaryMode = bMode;
    _init(numParticles);
}

void ParticleSystem::resetParticles(uint numParticles)
{
    gpuErrchk(cudaDeviceSynchronize());
    _init(numParticles);
}

void ParticleSystem::drawGUIConstellation()
{
    ImGui::Begin("SPH Constellation");

    m_partConstellation = "Default";

    //Combo Box to select particle array constellation
    const char* constellations[] = { "Default", "Sphere", "Four Pillars" };

    if (ImGui::BeginCombo("##combo", m_partConstellation))
    {
        for (int n = 0; n < IM_ARRAYSIZE(constellations); n++)
        {
            bool is_selected = (m_partConstellation == constellations[n]);
            if (ImGui::Selectable(constellations[n], is_selected))
                m_partConstellation = constellations[n];
            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::Text("Particle Constellation");

    ImGui::SliderFloat3("Gravity (m/s^2)", &m_gravity.x, -20.0f, 20.0f);
    gpuErrchk(cudaMemcpyToSymbol(gravity, &m_gravity, sizeof(float3)));

    // density from hydrogen to glycerol
    ImGui::SliderFloat("Density (kg/m^3)", &m_restingDensity, 0.09f, 1260.0f);
    gpuErrchk(cudaMemcpyToSymbol(restingDensity, &m_restingDensity, sizeof(float)));

    // large range because it gives really interesting results
    ImGui::SliderFloat("Viscosity (mPa*s)", &m_mu, 0.005f, 1.0f);
    gpuErrchk(cudaMemcpyToSymbol(mu, &m_mu, sizeof(float)));

    ImGui::End();
}