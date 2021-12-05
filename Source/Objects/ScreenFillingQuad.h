//
// Created by maxbr on 25.05.2020.
//

#pragma once

#include "../Engine/Shader.h"
#include "../Engine/ShaderProgram.h"
#include "../Engine/Drawable.h"
#include "../Engine/State.h"
#include "../Objects/Quad.h"
#include <string>

class ScreenFillingQuad : public Drawable
{
public:
    ScreenFillingQuad(std::shared_ptr<State> state);

    ~ScreenFillingQuad();

    std::shared_ptr<ShaderProgram> getShaderProgram();

    void draw(const GLuint fboBufferID);

private:
    std::shared_ptr<State> m_state;
    std::shared_ptr<Shader> m_fragmentShader;
    std::shared_ptr<Shader> m_drawFBOShader;
    std::shared_ptr<ShaderProgram> m_shaderProgram;
    std::unique_ptr<Quad> m_quad;

};