//
// Created by maxbr on 25.05.2020.
//

#include "ScreenFillingQuad.h"

ScreenFillingQuad::ScreenFillingQuad(std::shared_ptr<State> state)
{
    m_state = state;
    m_quad = std::make_unique<Quad>();

    m_fragmentShader = std::make_shared<Shader>(std::string(SHADERPATH) + "/DrawFBO.frag");
    m_drawFBOShader = std::make_shared<Shader>(std::string(SHADERPATH) + "/DrawFBO.vert");
    m_shaderProgram = std::make_shared<ShaderProgram>("DrawFBO");
    m_shaderProgram->addShader(m_fragmentShader);
    m_shaderProgram->addShader(m_drawFBOShader);
    m_shaderProgram->link();
}

ScreenFillingQuad::~ScreenFillingQuad()
{

}

std::shared_ptr<ShaderProgram> ScreenFillingQuad::getShaderProgram()
{
    return m_shaderProgram;
}

void ScreenFillingQuad::draw(const GLuint fboBufferID)
{
    m_shaderProgram->use();
    m_shaderProgram->setVec2("resolution", glm::vec2(m_state->getWidth(), m_state->getHeight()));
    m_shaderProgram->setSampler2D("fbo", fboBufferID, 0);
    m_quad->draw();
}
