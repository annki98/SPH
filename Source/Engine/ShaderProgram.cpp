//
// Created by maxbr on 22.05.2020.
//

#include "ShaderProgram.h"

ShaderProgram::ShaderProgram(std::string name)
{
    m_linked = false;
    m_isCompute = false;
    m_name = name;
    m_id = glCreateProgram();
}

ShaderProgram::~ShaderProgram()
{
    glDeleteProgram(m_id);
}

void ShaderProgram::link()
{
    glLinkProgram(m_id);

    // if errors occured, write them to log
    GLint success = 0;
    glGetProgramiv(m_id, GL_LINK_STATUS, &success);
    if(!success)
    {
        char infoLog[1024];
        glGetProgramInfoLog(m_id, 1024, NULL, infoLog);
        LOG_SHADER_ERROR("Shader program", "Shader program " + m_name + " linking failed: " + infoLog);
        glDeleteProgram(m_id);
    }

    m_linked = true;
}

void ShaderProgram::use()
{
    if(m_linked)
        glUseProgram(m_id);
    else
        LOG_WARNING(" Shader program " + m_name + " can't be used before being linked.");
}

GLuint ShaderProgram::getID()
{
    return m_id;
}

void ShaderProgram::addShader(std::shared_ptr<Shader> shader)
{
    // make sure that a compute shader program always just contains exactly one compute shader
    if(shader->getType().name == "Compute")
    {
        if(m_isCompute)
        {
            LOG_ERROR("Compute shader already attached on '" + m_name + "'!");
            return;
        }
        if(shaders.size() > 0)
        {
            LOG_ERROR(
                    "Can't add compute shader to program '" + m_name + "', which already contains different shaders!");
            return;
        }

        m_isCompute = true;
    }

    shaders.push_back(shader->getID());
    glAttachShader(m_id, shader->getID());
}

void ShaderProgram::setFloat(std::string name, float value) const
{
    GLuint uniformLocation = glGetUniformLocation(m_id, name.c_str());
    glUniform1f(uniformLocation, value);
}

void ShaderProgram::setBool(std::string name, bool value) const
{
    GLuint uniformLocation = glGetUniformLocation(m_id, name.c_str());
    glUniform1i(uniformLocation, value != 0);
}

void ShaderProgram::setInt(std::string name, int value) const
{
    GLuint uniformLocation = glGetUniformLocation(m_id, name.c_str());
    glUniform1i(uniformLocation, value);
}

void ShaderProgram::setVec2(std::string name, glm::vec2 value) const
{
    GLuint uniformLocation = glGetUniformLocation(m_id, name.c_str());
    glUniform2fv(uniformLocation, 1, glm::value_ptr(value));
}

void ShaderProgram::setVec3(std::string name, glm::vec3 value) const
{
    GLuint uniformLocation = glGetUniformLocation(m_id, name.c_str());
    glUniform3fv(uniformLocation, 1, glm::value_ptr(value));
}

void ShaderProgram::setVec4(std::string name, glm::vec4 value) const
{
    GLuint uniformLocation = glGetUniformLocation(m_id, name.c_str());
    glUniform4fv(uniformLocation, 1, glm::value_ptr(value));
}

void ShaderProgram::setMat2(std::string name, glm::mat2 value) const
{
    GLuint uniformLocation = glGetUniformLocation(m_id, name.c_str());
    glUniformMatrix2fv(uniformLocation, 1, false, glm::value_ptr(value));
}

void ShaderProgram::setMat3(std::string name, glm::mat3 value) const
{
    GLuint uniformLocation = glGetUniformLocation(m_id, name.c_str());
    glUniformMatrix3fv(uniformLocation, 1, false, glm::value_ptr(value));
}

void ShaderProgram::setMat4(std::string name, glm::mat4 value) const
{
    GLuint uniformLocation = glGetUniformLocation(m_id, name.c_str());
    glUniformMatrix4fv(uniformLocation, 1, false, glm::value_ptr(value));
}

void ShaderProgram::setSampler2D(std::string name, GLuint texture, int idGl) const
{
    glActiveTexture(GL_TEXTURE0 + idGl);
    glBindTexture(GL_TEXTURE_2D, texture);
    this->setInt(name, idGl);
}

void ShaderProgram::setSampler3D(std::string name, GLuint texture, int idGl) const
{
    glActiveTexture(GL_TEXTURE0 + idGl);
    glBindTexture(GL_TEXTURE_3D, texture);
    this->setInt(name, idGl);
}
