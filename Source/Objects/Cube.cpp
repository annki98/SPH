//
// Created by maxbr on 10.05.2020.
//

#include "Cube.h"

Cube::Cube(float size)
{
    create(size);
}

void Cube::create(float size)
{
    GLfloat vertices[] = { 
		-1.0f,  1.0f,  1.0f,  -1.0f, -1.0f,  1.0f,   1.0f, -1.0f,  1.0f,   1.0f,  1.0f,  1.0f,
		-1.0f,  1.0f, -1.0f,  -1.0f, -1.0f, -1.0f,  -1.0f, -1.0f,  1.0f,  -1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,   1.0f, -1.0f,  1.0f,   1.0f, -1.0f, -1.0f,   1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,  -1.0f,  1.0f,  1.0f,   1.0f,  1.0f,  1.0f,   1.0f,  1.0f, -1.0f,
		-1.0f, -1.0f,  1.0f,  -1.0f, -1.0f, -1.0f,   1.0f, -1.0f, -1.0f,   1.0f, -1.0f,  1.0f,
		 1.0f,  1.0f, -1.0f,   1.0f, -1.0f, -1.0f,  -1.0f, -1.0f, -1.0f,  -1.0f,  1.0f, -1.0f
    	};
    
    GLfloat normals[] = {        
         0.0f,  0.0f,  1.0f,    0.0f,  0.0f,  1.0f,    0.0f,  0.0f,  1.0f,    0.0f,  0.0f,  1.0f,
        -1.0f,  0.0f,  0.0f,   -1.0f,  0.0f,  0.0f,   -1.0f,  0.0f,  0.0f,   -1.0f,  0.0f,  0.0f,
         1.0f,  0.0f,  0.0f,    1.0f,  0.0f,  0.0f,    1.0f,  0.0f,  0.0f,    1.0f,  0.0f,  0.0f,
         0.0f,  1.0f,  0.0f,    0.0f,  1.0f,  0.0f,    0.0f,  1.0f,  0.0f,    0.0f,  1.0f,  0.0f,
         0.0f, -1.0f,  0.0f,    0.0f, -1.0f,  0.0f,    0.0f, -1.0f,  0.0f,    0.0f, -1.0f,  0.0f,
		 0.0f,  0.0f, -1.0f,    0.0f,  0.0f, -1.0f,    0.0f,  0.0f, -1.0f,    0.0f,  0.0f, -1.0f
    	};        
    
    GLfloat texCoords[] = {
		0.0f,  1.0f,    0.0f,  0.0f,    1.0f,  0.0f,    1.0f,  1.0f,
		0.0f,  1.0f,    0.0f,  0.0f,    1.0f,  0.0f,    1.0f,  1.0f,
		0.0f,  1.0f,    0.0f,  0.0f,    1.0f,  0.0f,    1.0f,  1.0f,
		0.0f,  1.0f,    0.0f,  0.0f,    1.0f,  0.0f,    1.0f,  1.0f,
		0.0f,  1.0f,    0.0f,  0.0f,    1.0f,  0.0f,    1.0f,  1.0f,
		0.0f,  1.0f,    0.0f,  0.0f,    1.0f,  0.0f,    1.0f,  1.0f
    	};

	m_points = 24;
	m_indices = 36;

	for(int i=0; i<m_points; i++)
	{
        m_vertices.push_back(glm::vec4( vertices[i*3] * size, vertices[i*3+1] * size, vertices[i*3+2] * size, 1.0f));
		m_normals.push_back(glm::vec3( normals[i*3], normals[i*3+1], normals[i*3+2]));
		m_uvs.push_back(glm::vec2( texCoords[i*2], texCoords[i*2+1]));
	}

	for(int i=0; i<6; i++)
	{
		m_index.push_back( i*4+0);
		m_index.push_back( i*4+1);
		m_index.push_back( i*4+2);
		m_index.push_back( i*4+2);
		m_index.push_back( i*4+3);
		m_index.push_back( i*4+0);
	}

	createBuffers();
}

void Cube::draw() 
{
    glBindVertexArray(m_vao);
    glLineWidth(2.0f);
	glDrawArrays(GL_LINE_STRIP, 0, m_vertices.size());

}