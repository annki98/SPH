//
// Created by maxbr on 22.05.2020.
//

#pragma once

#include "Defs.h"
#include <list>
#include "Shader.h"

class ShaderProgram
{
public:
    explicit ShaderProgram(std::string name);

    ~ShaderProgram();

    void link();

    void use();

    GLuint getID();

    void addShader(std::shared_ptr<Shader> shader);

    void setFloat(std::string name, float value) const;

    void setBool(std::string name, bool value) const;

    void setInt(std::string name, int value) const;

    void setVec2(std::string name, glm::vec2 value) const;

    void setVec3(std::string name, glm::vec3 value) const;

    void setVec4(std::string name, glm::vec4 value) const;

    void setMat2(std::string name, glm::mat2 value) const;

    void setMat3(std::string name, glm::mat3 value) const;

    void setMat4(std::string name, glm::mat4 value) const;

    void setSampler2D(std::string name, GLuint texture, int idGl) const;

    void setSampler3D(std::string name, GLuint texture, int idGl) const;

private:
    GLuint m_id;
    std::list<GLuint> shaders;

    bool m_linked;
    bool m_isCompute;
    std::string m_name;
};