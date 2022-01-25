#ifndef __PARTICLEGRID_H__
#define __PARTICLEGRID_H__

#include <iostream>
#include <random>
#define _USE_MATH_DEFINES
#include <math.h>
#include "../Engine/Defs.h"
#include <cuda_gl_interop.h>
#include "../Shared/cudaErrorCheck.h"

class Particle 
{
    public:
        float3 pressureGradient;
        float pressure;
        float3 position;
        float density;
        float3 velocity;
        float mass;
        // float3 convecAccel; wird bei timeIntegration berechnet
        float3 viscosity;
        bool isBoundary;
};

typedef unsigned int uint;

class ParticleSystem{
 public:
     ParticleSystem(uint numParticles, float3 worldOrigin, uint3 gridSize, float h);
    ~ParticleSystem();

    void timeIntegration(Particle* particles,
                float deltaTime,
                int numParticles);

    void calcHash(uint  *gridParticleHash,
                uint  *gridParticleIndex,
                //float *pos,
                Particle* particles,
                int    numParticles);
    
    void sortParticles(uint *dGridParticleHash, 
                uint *dGridParticleIndex, 
                uint* dSortedHash, 
                uint* dSortedIndex, 
                uint numParticles);

    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     Particle *sortedParticles,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     Particle *oldParticles,
                                     uint   numParticles,
                                     uint   numCells);

    void calcDensityPressure(Particle *sortedParticles,
             uint* cellStart,
             uint* cellEnd,
             uint* gridParticleIndex,
             uint numParticles
             );
   
    void calcSph(Particle *particleArray, //write new properties to this array
             Particle *sortedParticles,
             uint* cellStart,
             uint* cellEnd,
             uint* gridParticleIndex,
             uint numParticles
             );

    void gatherPositions(float4 * positions, //output
                        Particle *particleArray,
                        uint numParticles);

    void update(float deltaTime);

    // testing method to get all neighboring particles for a given indexed particle
    void checkNeighbors(uint index, int numParticles);
    void getSortedNeighbors(float3 pos, std::vector<uint> &neighborIndex, uint numParticles);
    void dumpParticleInfo(uint start, uint end);

    Particle* getParticleArray();
    GLuint getVBO();
    uint numParticles(){
        return m_numAllParticles;
    }

 protected:
    void _init(int numParticles);
    void _initParticles(int numParticles);
    void _initBoundary(float extend, uint numLayers, float spacing);
    void _free();
    
    uint m_numParticles;
    uint m_numBoundary;
    uint m_numAllParticles;

 private:

    const float3 m_gravity = make_float3(0.f,-9.81, 0.f);
    const float m_restingDensity = 1000.f;
    // Option 1
    //const float m_mu = float(10e-6) * m_restingDensity;
    // Option 2
    const float m_nu = float(10e-6);
    float m_spacing;
    float m_fluidVolume;
    float m_particleVolume;
    float m_uniform_mass;

    Particle* m_particleArray;
    Particle* m_sortedParticleArray;

    uint* m_dGridParticleHash;
    uint* m_dGridParticleIndex;
    uint* m_dSortedParticleHash;
    uint* m_dSortedParticleIndex;

    float3 m_worldOrigin;
    uint m_numGridCells;
    uint3 m_gridSize;
    float3 m_cellSize;

    uint* m_cellStart;
    uint* m_cellEnd;

    //OpenGL
    void _setGLArray(uint numParticles);
    GLuint m_vbo;
    struct cudaGraphicsResource *m_cuda_vbo_resource;

};

#endif // __PARTICLEGRID_H__
