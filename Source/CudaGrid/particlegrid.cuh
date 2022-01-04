#ifndef __PARTICLEGRID_H__
#define __PARTICLEGRID_H__

#include <iostream>
#include <random>

typedef unsigned int uint;

class ParticleSystem{
 public:
     ParticleSystem(uint numParticles, float3 worldOrigin, uint3 gridSize, float h);
    ~ParticleSystem();

    void calcHash(uint  *gridParticleHash,
                uint  *gridParticleIndex,
                float *pos,
                int    numParticles);
    
    void sortParticles(uint *dGridParticleHash, 
                uint *dGridParticleIndex, 
                uint* dSortedHash, 
                uint* dSortedIndex, 
                uint numParticles);

    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     float *sortedPos,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     float *oldPos,
                                     uint   numParticles,
                                     uint   numCells);
    void update();

    void checkNeighbors(uint index);

 protected:
    void _init(int numParticles);
    void _free();

 private:
    uint m_numParticles;

    float3* m_particles;
    float3* m_sortedParticles;

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

};

#endif // __PARTICLEGRID_H__
