//
// Created by maxbr on 10.05.2020.
//

#pragma once

#include "../Engine/Drawable.h"
#include "../Objects/Cube.h"
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
    void setupSphere(glm::vec3 center, float radius, int resolution);
    void setDepthtexture(GLuint depthTexture);
private:
    void drawGUI();
    std::unique_ptr<ParticleSystem> m_psystem;
    float time;
    float timeSpeed;
    uint3 m_hostGridSize;

    std::shared_ptr<State> m_state;

    // GUI specific
    const char* m_renderingMode;
    float m_sphereRadius;
    bool m_renderBoundaries;

    // Rendering specific
    std::unique_ptr<Cube> m_cube;

    std::shared_ptr<Shader> m_vertexBasicShader;
    std::shared_ptr<Shader> m_fragmentBasicShader;
    std::unique_ptr<ShaderProgram> m_basicShaderProgram;

    std::shared_ptr<Shader> m_vertexBasicWithModelShader;
    std::shared_ptr<Shader> m_fragmentBasicWithModelShader;
    std::unique_ptr<ShaderProgram> m_basicWithModelShaderProgram;

    std::shared_ptr<Shader> m_vertexSphereShader;
    std::shared_ptr<Shader> m_fragmentSphereShader;
    std::unique_ptr<ShaderProgram> m_shpereShaderProgram;

    std::vector<glm::vec4> m_sphereVertices;
    std::vector<glm::vec3> m_sphereNormals;
    std::vector<GLuint> m_sphereIndices;

    GLuint m_depthTexture;
    GLuint m_vaoSphere;
    GLuint m_vboSphere;
    GLuint m_indexBufferSphere;
};

