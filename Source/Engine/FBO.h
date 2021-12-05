//
// Created by maxbr on 24.05.2020.
//

#pragma once

#include <cstdlib>
#include "Defs.h"

GLuint createFBO();

void bindFBO(GLuint fboID, size_t width, size_t height);

void unbindFBO(size_t width, size_t height);

GLuint *createColorAttachments(size_t width, size_t height, GLuint numberOfColorAttachments);

GLuint createDepthTextureAttachment(size_t width, size_t height);

GLuint createTextureAttachment(size_t width, size_t height);

GLuint createDepthBufferAttachment(size_t width, size_t height);

class FBO
{
public:
    FBO(size_t width, size_t height);

    FBO(size_t width, size_t height, size_t numberOfColorAttachments);

    ~FBO();

    void bind();

    GLuint getID();

    GLuint getColorAttachment(size_t i);

private:
    size_t m_width, m_height;
    GLuint m_fboID;

    size_t m_numberOfColorAttachments;
    GLuint *m_colorAttachments;

};