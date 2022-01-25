#include <iostream>
#include <iomanip>
#include <random>
#include <cub/cub.cuh>
#include <chrono>
#include <cooperative_groups.h>
#include "helper_math.h"

#include "particlegrid.cuh"

namespace cg = cooperative_groups;

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
__constant__ float nu; // how viscous is the fluid
__constant__ float V_i; // Volume of a particle

__global__ void timeIntegrationD(Particle* particles,
                                float deltaTime,
                                uint numParticles)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numParticles){
        return;
    }

    float3 pos = particles[index].position;
    float3 vel = particles[index].velocity;

    // if(index == 0){
    //     printf("pos before integrating: (%f,%f,%f)\n", pos.x, pos.y, pos.z);
    //     printf("vel before integrating: (%f,%f,%f)\n", vel.x, vel.y, vel.z);
    //     printf("-- pressureGradient:  (%f,%f,%f)\n", particles[index].pressureGradient.x, particles[index].pressureGradient.y, particles[index].pressureGradient.z);
    //     printf("-- viscosity:  (%f,%f,%f)\n", particles[index].viscosity.x, particles[index].viscosity.y, particles[index].viscosity.z);
    // }
    // equation (1)
    float3 acceleration = gravity + particles[index].pressureGradient + particles[index].viscosity;

    // if(isnan(length(acceleration))){
    //     printf("Acceleration - NaN: index [%u]\n", index);
    // }
    // if(index == 0){
    //     printf("Acceleration = (%f,%f,%f).\n", acceleration.x, acceleration.y, acceleration.z);
    // }

    //TODO: write better integration scheme
    vel += acceleration * deltaTime;
    pos += vel * deltaTime;

    //Boundary Handling
    

    const float ground = -0.5f;
    const float edge = gridSize.x * cellSize.x;//100.f;
    const float ceiling = edge*2;
    const float groundDamping = 0.99;
    const float edgeDamping = 0.6;

    // if(index == 0){
    //     printf("pos before boundary handling: (%f,%f,%f)\n", pos.x, pos.y, pos.z);
    //     printf("vel before boundary handling: (%f,%f,%f)\n", vel.x, vel.y, vel.z);
    // }
    //Ground
    if(pos.y < ground){
        pos.y = ground;
        vel *= groundDamping; //Reduce velocity when "hitting" the ground 
    }
    // //Ceiling
    // if(pos.y > ceiling){
    //     pos.y = ceiling;
    //     vel *= edgeDamping; //Reduce velocity when hitting the ceiling
    // }
    // // Edges
    // if( pos.x > edge){
    //     pos.x = edge;
    //     vel *= edgeDamping; //Reduce velocity when hitting the edge 
    // }
    // else if(pos.x < 0){
    //     pos.x = 0;
    //     vel *= edgeDamping;
    // }
    // if( pos.z > edge){
    //     pos.z = edge;
    //     vel *= edgeDamping;
    // }
    // else if(pos.z < 0){
    //     pos.z = 0;
    //     vel *= edgeDamping; 
    // }

    // if(index == 0){
    //     printf("pos after boundary handling: (%f,%f,%f)\n", pos.x, pos.y, pos.z);
    //     printf("vel after boundary handling: (%f,%f,%f)\n", vel.x, vel.y, vel.z);
    // }
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
        Particle pos = oldParticles[sortedIndex];

        sortedParticles[index] = pos;
    }


}

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

                                if(dist > smoothingRadius){
                                    //Dismiss
                                    continue;
                                }
                                
                                
                                //Density equation
                                float denom = static_cast<float>(64 * M_PI * pow(smoothingRadius, 9));
                                float W = (315 / denom) * pow(pow(smoothingRadius, 2) - pow(dist,2),3);

                                // if(neighborParticle.isBoundary){
                                //     sumWBound += W;
                                // }
                                // else{
                                //     sumWFluid += W;
                                // }
                                sumWFluid += W;
                                // if(originalIndex == 0){

                                //     printf("DENSITY -- neighbor (%f,%f,%f). Distance: %f, density %f: Contributing %f\n", neighborParticle.position.x,neighborParticle.position.y,neighborParticle.position.z, dist, neighborParticle.density, particle.mass*W);
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
        // float gamma1 = 1.f;
        // if(sumWBound > 0.f){
        //     gamma1 = ((1.f/V_i) - sumWFluid)/sumWBound;
        // }
        // float density = (particle.mass * sumWFluid + gamma1 * particle.mass * sumWBound);
        float density = particle.mass * sumWFluid;
        // if(isnan(density)){
        //     printf("DENSITY - NaN: index [%u]\n", originalIndex);
        // }
        // if(originalIndex == 0){
        //     printf("Density = %f. \n",density);
        // }
        return density;
}


__device__ float3 sphPressureGradient(
                uint* cellStart,
                uint* cellEnd,
                int3 gridPos,
                Particle *particles,
                uint index,
                uint originalIndex){

        float3 sumFluid = make_float3(0.0f,0.0f,0.0f);
        float3 sumBound = make_float3(0.0f,0.0f,0.0f);
        // float3 sumNablaWFluid = make_float3(0.0f,0.0f,0.0f);
        // float3 sumNablaWBound = make_float3(0.0f,0.0f,0.0f);

        Particle particle = particles[index];
        // go over all surrounding cells
        int num = 0;

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
                                if(dist > smoothingRadius || dist == 0.f){ //check for particle overlap
                                    //Dismiss
                                    continue;
                                }
                                // if(originalIndex == 0){
                                //     printf("GRADIENT -- neighbor (%f,%f,%f). Distance: %f, Pressure %f, density %f.\n", neighborParticle.position.x,neighborParticle.position.y,neighborParticle.position.z, dist, neighborParticle.pressure, neighborParticle.density);
                                //     if(neighborParticle.isBoundary){
                                //         printf("-> BOUNDARY\n");
                                //     }
                                // }
                                
                                float3 NablaW = (-45 / (M_PI * pow(smoothingRadius,6))) * pow(smoothingRadius - dist,2) * dVec/dist;
                                // if(originalIndex == 0){
                                //     printf("NablaW = (%f,%f,%f).\n", NablaW.x,NablaW.y,NablaW.z);
                                // }

                                // if(neighborParticle.isBoundary){

                                //     float pi = (particle.pressure/ pow(particle.density,2));
                                //     float pj = pi; // mirroring pressure at boundary
                                //     sumNablaWBound += NablaW;
                                //     sumBound += (pi + pj) * NablaW;
                                // }
                                // else{

                                //     float pi = (particle.pressure/ pow(particle.density,2));
                                //     float pj = (neighborParticle.pressure / pow(neighborParticle.density,2));
                                //     sumNablaWFluid += NablaW;
                                //     sumFluid += (pi + pj) * NablaW;
                                
                                // }
                                float pi = (particle.pressure/ pow(particle.density,2));
                                float pj = (neighborParticle.pressure / pow(neighborParticle.density,2));
                                // if(originalIndex == 0){
                                //     printf("IN GRADIENT: pi %f, pj %f.\n",pi,pj);
                                // }
                                // sumNablaWFluid += NablaW;
                                float3 add = (pi + pj) * NablaW;
                                // if(isnan(length(add))){
                                //     float3 pos = particle.position;
                                //     printf("Adding to Gradient - NaN: index [%u] at pos (%f,%f,%f). Density %f. pi = %f. pj = %f >>\n", originalIndex, pos.x,pos.y,pos.z, particle.density,pi,pj);
                                //     printf(">> Distance = %f, NablaW = (%f,%f,%f).\n", dist, NablaW.x,NablaW.y,NablaW.z);
                                // }

                                sumFluid += add;
                                num++;
                            }
                        }
                    }
                }
            }
        }

        float3 gradient = - particle.mass * sumFluid;
        // if(isnan(gradient.x) || isnan(gradient.y) || isnan(gradient.z)){
        //     float3 pos = particle.position;
        //     printf("GRADIENT - NaN: index [%u] at pos (%f,%f,%f). Density %f. Number of neigbors = %u\n", originalIndex, pos.x,pos.y,pos.z, particle.density,num);
        // }
        return gradient;

        // float3 gamma2 = make_float3(0.f,0.f,0.f);
        // float pi = 0.f;
        // if(length(sumNablaWFluid) > 0 && length(sumNablaWBound) > 0){ // only if neighbours for boundary exist
        //     pi = (particle.pressure/ pow(particle.density,2));
        //     gamma2 = -(sumNablaWFluid*sumNablaWBound)/(sumNablaWBound*sumNablaWBound);
        // }
        // float3 gradientFluid = - particle.mass * sumFluid; 
        // // if(originalIndex == 0){
        // //     printf("IN GRADIENT: Density %f, Pressure %f, sumNablaWBound (%f,%f,%f).\n",particle.density, particle.pressure, sumNablaWBound.x,sumNablaWBound.y,sumNablaWBound.z);
        // // }
        
        // float3 gradientBound = -((2*pi*particle.mass) * gamma2) * sumNablaWBound;
        // // float3 gradientBound =  - particle.mass * sumBound;

        // float3 gradient = gradientFluid + gradientBound;

        // // if(originalIndex == 0){
        // //     printf("GradientFluid = (%f,%f,%f), GradientBound = (%f,%f,%f)\n", gradientFluid.x,gradientFluid.y,gradientFluid.z,gradientBound.x,gradientBound.y,gradientBound.z);
        // //     printf("Gradient = (%f,%f,%f), SumFluid = (%f,%f,%f).\n\n", gradient.x, gradient.y, gradient.z, sumFluid.x,sumFluid.y,sumFluid.z);
        // //     // printf("Gradient = (%f,%f,%f), Gamma2 = (%f,%f,%f). SumFluid = (%f,%f,%f).\n", gradient.x, gradient.y, gradient.z,gamma2.x,gamma2.y,gamma2.z, sumFluid.x,sumFluid.y,sumFluid.z);

        // // }
        // return gradient;
}


__device__ float3 sphViscosity(
                uint* cellStart,
                uint* cellEnd,
                int3 gridPos,
                Particle *particles,
                uint index,
                uint originalIndex){

        float3 viscosity = make_float3(0.0f,0.0f,0.0f);
        Particle particle = particles[index];
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
                                if(dist > smoothingRadius){
                                    //Dismiss
                                    continue;
                                }

                                // if(originalIndex == 0){
                                //     printf("VISCOSITY -- neighbor (%f,%f,%f). Distance: %f, density %f. ", neighborParticle.position.x,neighborParticle.position.y,neighborParticle.position.z, dist, neighborParticle.density);
                                // }
                                float3 add = ((neighborParticle.velocity - particle.velocity) / neighborParticle.density)
                                            * (neighborParticle.mass * (45/(M_PI * pow(smoothingRadius,6))) * (smoothingRadius - dist));
                                viscosity += add;
                                // if(originalIndex == 0){
                                //     printf("adding (%f,%f,%f).\n", add.x,add.y,add.z);
                                // }
                            }
                        }
                    }
                }
            }
        }
        // if(isnan(length(viscosity))){
        //     printf("Viscosity - NaN: index [%u]\n", originalIndex);
        // }
        // return (mu / particle.density) * viscosity;
        return 2* nu * viscosity;
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
    float K = 0.4f;
    float pressure = K * (density - restingDensity);
    
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

        // SPH: Pressure gradient
        float3 gradient = sphPressureGradient(cellStart, cellEnd, gridPos,oldParticles,index, originalIndex);

        // SPH: viscosity
        float3 viscosity = sphViscosity(cellStart,cellEnd,gridPos,oldParticles,index,originalIndex);

        
        
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

void ParticleSystem::getSortedNeighbors(float3 pos, std::vector<uint> &neighborIndex, uint numParticles){
    // uint numThreads, numBlocks;
    // computeGridSize(numParticles, 64, numBlocks, numThreads);

    int3 gridPos;
    gridPos.x = floorf((pos.x) / m_cellSize.x);
    gridPos.y = floorf((pos.y) / m_cellSize.y);
    gridPos.z = floorf((pos.z) / m_cellSize.z);

    printf("(%d,%d,%d)\n", gridPos.x, gridPos.y, gridPos.z);
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



ParticleSystem::ParticleSystem(uint numParticles, float3 hostWorldOrigin, uint3 hostGridSize, float h):
    m_numParticles(numParticles),
    m_numBoundary(0),
    m_numAllParticles(0),
    m_particleArray(0),
    m_worldOrigin(hostWorldOrigin),
    m_sortedParticleArray(0),
    m_gridSize(hostGridSize),
    m_cellSize(make_float3(h,h,h)), //cell size equals smoothing length
    m_uniform_mass(0.f)
{
    gpuErrchk(cudaMemcpyToSymbol(worldOrigin, &m_worldOrigin, sizeof(float3)));
    gpuErrchk(cudaMemcpyToSymbol(smoothingRadius, &h, sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(cellSize, &m_cellSize, sizeof(float3)));
    gpuErrchk(cudaMemcpyToSymbol(gridSize, &m_gridSize, sizeof(uint3)));

    // copy equation constants to gpu
    gpuErrchk(cudaMemcpyToSymbol(gravity, &m_gravity, sizeof(float3)));
    gpuErrchk(cudaMemcpyToSymbol(restingDensity, &m_restingDensity, sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(nu, &m_nu, sizeof(float)));

    m_spacing = 0.3f;
    m_numGridCells = hostGridSize.x * hostGridSize.y * hostGridSize.z;
    //m_fluidVolume = m_gridSize.x*m_cellSize.x * m_gridSize.y*m_cellSize.y * m_gridSize.z*m_cellSize.z;
    m_fluidVolume = pow(m_spacing,3) * numParticles;
    m_particleVolume = m_fluidVolume/numParticles;
    gpuErrchk(cudaMemcpyToSymbol(V_i, &m_particleVolume, sizeof(float)));
    // Option 1
    m_uniform_mass = m_restingDensity * m_particleVolume; // rho_0 * V / n
    // Option 2
    //m_uniform_mass = m_restingDensity * (m_cellSize.x * m_cellSize.y *m_cellSize.z); // rho_0 * h^3

    _init(numParticles);
}


void ParticleSystem::_initParticles(int numParticles)
{
    // std::random_device seed;
    // std::default_random_engine rng(seed());
    // std::uniform_real_distribution<float> pos(0, m_gridSize.x*m_cellSize.x); //assumes all dims to be same size
    
    Particle* it = m_particleArray;

    int width = 20;
    int height = numParticles/(width*width);
    int count = 0;
    for(auto y = 0; y < height; y++){
        for(auto z = 0; z < width; z++){
            for(auto x = 0; x < width; x++){
                
                it->position = m_spacing*make_float3(x,y,z);
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
        it->position = m_spacing*make_float3(x,y,z);
        it->mass = m_uniform_mass;
        it->isBoundary = false;
        it++;
        count++;
    }

}

void ParticleSystem::_initBoundary(float extend, uint numLayers, float spacing)
{
    
    float height = -0.5f; //ground height
    float middle = (m_cellSize.x*m_gridSize.x)/2;
    //for now: only Ground considered
    Particle* it = m_particleArray + m_numParticles;
    int i = 0;
    for(auto x = middle-extend/2; x < middle+extend/2; x += spacing){
        for(auto z = middle-extend/2; z < middle+extend/2; z += spacing){
            for(uint y = 0; y < numLayers; y++){
                // printf("Boundary particle (%f,%f,%f) added.\n",x,y,z);
                i++;
                it->position = make_float3(x,height-spacing*y,z);
                it->mass = m_uniform_mass;
                it->isBoundary = true;
                it++;
            }
        }
    }
    printf("%u boundary particles added!\n",i);
}

void ParticleSystem::_setGLArray(uint numParticles){
    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    // glBufferData(GL_ARRAY_BUFFER, m_numParticles*sizeof(float4), 0, GL_DYNAMIC_DRAW);
    glBufferData(GL_ARRAY_BUFFER, numParticles*sizeof(float4), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&m_cuda_vbo_resource, m_vbo, cudaGraphicsRegisterFlagsNone));
}

void ParticleSystem::_init(int numParticles)
{
    float boundaryExtend = 0;// 1.2f * m_gridSize.x *m_cellSize.x;
    m_numBoundary = 0; //static_cast<uint>(pow(boundaryExtend/m_spacing,2)); // Boundary particles for uniform ground square

    m_numAllParticles = numParticles + m_numBoundary;
    uint size = (m_numAllParticles) * sizeof(Particle);

    gpuErrchk(cudaMallocManaged(&m_particleArray, size));
    gpuErrchk(cudaMallocManaged(&m_sortedParticleArray, size));

    _setGLArray(m_numAllParticles);
    _initParticles(numParticles);
    gpuErrchk( cudaDeviceSynchronize());
    // _initBoundary(boundaryExtend, 1, m_spacing);
    // gpuErrchk( cudaDeviceSynchronize());

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
    glDeleteBuffers(1, (const GLuint *)&m_vbo);
    gpuErrchk(cudaGraphicsUnregisterResource(m_cuda_vbo_resource));
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

    timeIntegrationD<<< numBlocks, numThreads >>>(particles, deltaTime, numParticles);

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

        //calculate density and pressure first
        calcDensityPressureD<<< numBlocks, numThreads >>>(sortedParticles,
                                                            cellStart,
                                                            cellEnd,
                                                            gridParticleIndex,
                                                            numParticles);
        gpuErrchk( cudaPeekAtLastError());
        gpuErrchk( cudaDeviceSynchronize());

        // calculate pressure gradient and viscosity
        calcSphD<<< numBlocks, numThreads >>>(particleArray,
                                              sortedParticles,
                                              cellStart,
                                              cellEnd,
                                              gridParticleIndex,
                                              numParticles);
    }

void ParticleSystem::gatherPositions(float4* positions, Particle* particleArray, uint numParticles){
    // thread per particle
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 64, numBlocks, numThreads);

    gatherPositionsD<<< numBlocks, numThreads >>>(positions, particleArray, numParticles);
}

void ParticleSystem::update(float deltaTime)
{

    calcHash(m_dGridParticleHash, m_dGridParticleIndex, m_particleArray, m_numAllParticles);

    gpuErrchk( cudaDeviceSynchronize());
    // std::cout<<"------------\n";
    // for (int i = 0; i < 100; i++){
    //     float3 p = m_particleArray[i].position;
    //     std::cout << "Particle "<<p.x<<", "<<p.y<<", "<<p.z<<", "
    //     <<"Hash: "<<m_dGridParticleHash[i]<<
    //     ", Index: "<<m_dGridParticleIndex[i]<<std::endl;
    // }


    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_dSortedParticleHash, m_dSortedParticleIndex, m_numAllParticles);

    gpuErrchk( cudaDeviceSynchronize());
    // std::cout<<"------ HASHING ------\n";
    // for (int i = 0; i < 100; i++){
    //     float3 p = m_particleArray[m_dSortedParticleIndex[i]].position;
    //     std::cout << "Particle "<<p.x<<", "<<p.y<<", "<<p.z
    //                 <<", "<<"Hash: "<<m_dSortedParticleHash[i]<<", Index: "<<m_dSortedParticleIndex[i]<<std::endl;
    // }


    reorderDataAndFindCellStart(m_cellStart,
                                m_cellEnd,
                                m_sortedParticleArray,
                                m_dSortedParticleHash,
                                m_dSortedParticleIndex,
                                m_particleArray,
                                m_numAllParticles,
                                m_numGridCells);

    gpuErrchk( cudaDeviceSynchronize());
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

    //TODO: call calcSph
    calcSph(m_particleArray, 
            m_sortedParticleArray,
            m_cellStart,
            m_cellEnd, 
            m_dSortedParticleIndex, 
            m_numAllParticles);
    gpuErrchk( cudaDeviceSynchronize());   
    // dumpParticleInfo(0,3);

    timeIntegration(m_particleArray, deltaTime, m_numParticles);    
    gpuErrchk( cudaDeviceSynchronize());  
    // std::cout<<"======== AFTER TIME INTEGRATION =========\n"; 
    // dumpParticleInfo(0,3);


    //map m_positions to be used with cuda
    float4* m_positions;
    gpuErrchk(cudaGraphicsMapResources(1, &m_cuda_vbo_resource));
    size_t num_bytes;
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void **) &m_positions, &num_bytes, m_cuda_vbo_resource));

    gatherPositions(m_positions, m_particleArray, m_numAllParticles);
    // unmap
    gpuErrchk(cudaGraphicsUnmapResources(1, &m_cuda_vbo_resource, 0));
}

void ParticleSystem::checkNeighbors(uint index, int numElements){

    Particle testParticle = m_particleArray[index];
    float3 p = testParticle.position;
    std::cout<<" ### NEIGHBORS FOR PARTICLE "<<p.x<<","<<p.y<<","<<p.z<<" ###\n"<<std::endl;
    
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
    printf("\nWith sorting: \n");
    for(int i = 0; i < neighbors.size(); i++){
        uint index = neighbors[i];
        float3 neighPos = m_sortedParticleArray[index].position;
        printf("Neighbor %d, pos (%f,%f,%f). Dist is %f.", index, neighPos.x, neighPos.y, neighPos.z,length(p-neighPos));
        if(m_sortedParticleArray[index].isBoundary){
            printf("- BOUNDARY");
        }
        printf("\n");
        
    }
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


// int main() {

//     float3 hostWorldOrigin = make_float3(0.f,0.f,0.f);
//     float h = 1.f;
//     uint3  hostGridSize = make_uint3(64,64,64); // must be power of 2

//     ParticleSystem* psystem = new ParticleSystem(numElements, hostWorldOrigin, hostGridSize, h);

//     psystem->update();

//     //for testing purposes
//     psystem->checkNeighbors(5);
//     psystem->dumpParticleInfo(0,6);

//     return 0;
// }
