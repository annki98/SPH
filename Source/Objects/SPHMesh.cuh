//
// Created by maxbr on 10.05.2020.
//

#pragma once

#include "../Engine/Drawable.h"
#include "../CudaGrid/particlegrid.cuh"
#include <random>

class SPHMesh : public Drawable
{
public:
    SPHMesh();
    ~SPHMesh();
    void draw();
    void updateParticles(float deltaTime);

    void createBuffers();
private:

    ParticleSystem* m_psystem;
    float time;
};

