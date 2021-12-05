//
// Created by maxbr on 23.05.2020.
//

#include "Texture.h"

GLuint createTexture2D(std::size_t width, std::size_t height)
{
    GLuint texHandle;
    glGenTextures(1, &texHandle);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texHandle);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

    glBindImageTexture(0, texHandle, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA);
    return texHandle;
}

GLuint createGroundTexture2D(std::size_t width, std::size_t height)
{
    GLuint texHandle;
    glGenTextures(1, &texHandle);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texHandle);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

    glBindImageTexture(0, texHandle, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    return texHandle;
}

GLuint createTexture3D(std::size_t width, std::size_t height, std::size_t depth)
{
    GLuint textureId;
    glGenTextures(1, &textureId);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, textureId);

    // set some usable default tex parameters
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    // create texture
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, width, height, depth, 0, GL_RGBA, GL_FLOAT, nullptr);

    // generate mipmap
    glGenerateTextureMipmap(GL_TEXTURE_3D);

    // bind level of texture to texture unit
    glBindImageTexture(0, textureId, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    return textureId;
}

GLuint createWeatherTexture3D(std::size_t width, std::size_t height, std::size_t depth)
{
    GLuint textureId;
    glGenTextures(1, &textureId);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, textureId);

    // set some usable default tex parameters
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);


    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    // create texture
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, width, height, depth, 0, GL_RGBA, GL_FLOAT, nullptr);

    // generate mipmap
    glGenerateTextureMipmap(GL_TEXTURE_3D);

    // bind level of texture to texture unit
    glBindImageTexture(0, textureId, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    return textureId;
}

GLuint createTexture3D(std::size_t width, std::size_t height, std::size_t depth, float *data)
{
    GLuint textureId;
    glGenTextures(1, &textureId);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, textureId);

    // set some usable default tex parameters
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    // create texture
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, width, height, depth, 0, GL_RGBA, GL_FLOAT, (GLvoid *) data);

    // generate mipmap
    glGenerateTextureMipmap(GL_TEXTURE_3D);

    // bind level of texture to texture unit
    glBindImageTexture(0, textureId, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    return textureId;
}

GLuint createTextureFromFile(std::string path)
{
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int width, height, nrChannels;
    unsigned char *data = stbi_load(path.c_str(), &width, &height, &nrChannels, 0);
    if(data)
    {
        // account for possible alpha channel in the data
        assert((1 <= nrChannels) && (4 >= nrChannels));
        GLenum glformat;
        switch(nrChannels)
        {
            case 1:
                glformat = GL_RED;
                break;
            case 2:
                glformat = GL_RG8;
                break;
            case 3:
                glformat = GL_RGB8;
                break;
            case 4:
                glformat = GL_RGBA8;
                break;
        }

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

    } else
    {
        LOG_ERROR("failed to load texture " + path);
    }
    stbi_image_free(data);

    return texture;
}

GLuint createGroundTextureFromFile(std::string path)
{
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_3D, texture);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int width, height, nrChannels;
    unsigned char *data = stbi_load(path.c_str(), &width, &height, &nrChannels, 0);

    if(data)
    {
        // account for possible alpha channel in the data
        assert((1 <= nrChannels) && (4 >= nrChannels));
        GLenum glformat;
        switch(nrChannels)
        {
            case 1:
                glformat = GL_RED;
                break;
            case 2:
                glformat = GL_RG8;
                break;
            case 3:
                glformat = GL_RGB8;
                break;
            case 4:
                glformat = GL_RGBA8;
                break;
        }

        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, width, height, 3, 0, GL_RED, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_3D);

    } else
    {
        LOG_ERROR("failed to load texture" + path);
    }
    stbi_image_free(data);

    return texture;
}

void bindTexture2D(GLuint textureID, unsigned int textureUnit)
{
    glBindImageTexture(textureUnit, textureID, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8);
}

void bindTexture3D(GLuint textureID, unsigned int textureUnit)
{
    glBindImageTexture(textureUnit, textureID, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
}