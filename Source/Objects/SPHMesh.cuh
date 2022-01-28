//
// Created by maxbr on 10.05.2020.
//

#pragma once

#include "../Engine/Drawable.h"
#include "../CudaGrid/particlegrid.cuh"
#include "../Engine/State.h"
#include "../Engine/Shader.h"
#include "../Engine/ShaderProgram.h"
#include <random>

class SPHMesh : public Drawable
{
public:
    SPHMesh(std::shared_ptr<State> state);
    void draw();
    void updateParticles(float deltaTime);

    void createBuffers();
private:
    void drawGUI();
    std::unique_ptr<ParticleSystem> m_psystem;
    float time;

    std::shared_ptr<State> m_state;

    // GUI specific
    const char* m_renderingMode;
    float m_sphereRadius;
    bool m_renderBoundaries;

    // Rendering specific
    std::shared_ptr<Shader> m_vertexBasicShader;
    std::shared_ptr<Shader> m_fragmentBasicShader;
    std::unique_ptr<ShaderProgram> m_basicShaderProgram;

    std::shared_ptr<Shader> m_vertexSphereShader;
    std::shared_ptr<Shader> m_fragmentSphereShader;
    std::unique_ptr<ShaderProgram> m_shpereShaderProgram;
};

