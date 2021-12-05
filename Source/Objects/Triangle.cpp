//
// Created by maxbr on 10.05.2020.
//

#include "Triangle.h"

Triangle::Triangle()
{
    create(glm::vec3(0.5f, 0.5f, 0.0f), glm::vec3(0.5f, -0.5f, 0.0f), glm::vec3(-0.5f, -0.5f, 0.0f),
           glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 0.0f, 1.0f),
           glm::vec2(1.0f, 0.0f), glm::vec2(0.0f, 1.0f), glm::vec2(0.0f, 0.0f));
}

void Triangle::create(glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 na, glm::vec3 nb, glm::vec3 nc, glm::vec2 tca,
                      glm::vec2 tcb, glm::vec2 tcc)
{
    m_vertices.push_back(glm::vec4(a, 1.0f));
    m_vertices.push_back(glm::vec4(b, 1.0f));
    m_vertices.push_back(glm::vec4(c, 1.0f));

    m_normals.push_back(na);
    m_normals.push_back(nb);
    m_normals.push_back(nc);

    m_uvs.push_back(tca);
    m_uvs.push_back(tcb);
    m_uvs.push_back(tcc);

    m_numberOfPoints = 3;

    m_index.push_back(0);
    m_index.push_back(1);
    m_index.push_back(2);

    m_numberOfIndices = 3;

    createBuffers();
}