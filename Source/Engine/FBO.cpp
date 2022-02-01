//
// Created by maxbr on 24.05.2020.
//

#include "FBO.h"

void bindFBO(GLuint fboID, size_t width, size_t height)
{
    // unbind textures that may already be bound
    glBindTexture(GL_TEXTURE_2D, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, fboID);
    glViewport(0, 0, width, height);
}

void unbindFBO(size_t width, size_t height)
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, width, height);
}

GLuint createFBO()
{
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    return fbo;
}

GLuint *createColorAttachments(size_t width, size_t height, GLuint numberOfColorAttachments)
{
    GLuint *colorAttachments = new GLuint[numberOfColorAttachments];
    glGenTextures(numberOfColorAttachments, colorAttachments);

    for(GLuint i = 0; i < numberOfColorAttachments; i++)
    {
        glBindTexture(GL_TEXTURE_2D, colorAttachments[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, 0);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, colorAttachments[i], 0);
    }
    return colorAttachments;
}

GLuint createTextureAttachment(size_t width, size_t height)
{
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, width, height);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureID, 0);

    return textureID;
}

GLuint createDepthBufferAttachment(size_t width, size_t height)
{
    GLuint depthAttachement;
    glGenTextures(1, &depthAttachement);

    glBindTexture(GL_TEXTURE_2D, depthAttachement);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, 0);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, width, height, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, nullptr);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, depthAttachement, 0);

    return depthAttachement;
}

FBO::FBO(size_t width, size_t height)
{
    m_width = width;
    m_height = height;

    m_fboID = createFBO();
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_fboID);  
    m_numberOfColorAttachments = 0;
    m_colorAttachments = NULL;
}

FBO::FBO(size_t width, size_t height, size_t numberOfColorAttachments)
{
    m_width = width;
    m_height = height;

    m_fboID = createFBO();
    glBindFramebuffer(GL_FRAMEBUFFER, m_fboID);
    m_numberOfColorAttachments = numberOfColorAttachments;
    m_colorAttachments = createColorAttachments(m_width, m_height, m_numberOfColorAttachments);
    m_depthAttachment = createDepthBufferAttachment(m_width, m_height);

}

FBO::~FBO()
{

}

void FBO::bind()
{
    bindFBO(m_fboID, m_width, m_height);
}

GLuint FBO::getID()
{
    return m_fboID;
}

GLuint FBO::getDepthAttachment()
{
    return m_depthAttachment;
}

GLuint FBO::getColorAttachment(size_t i)
{
    if(i > m_numberOfColorAttachments)
    {
        LOG_ERROR("Color attachment index out of range!");
        return 0;
    }
    return m_colorAttachments[i];
}