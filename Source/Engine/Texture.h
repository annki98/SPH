//
// Created by maxbr on 23.05.2020.
//

#pragma once

#include "Defs.h"
#include <string>
#include <stb_image.h>

GLuint createTexture2D(std::size_t width, std::size_t height);
GLuint createGroundTexture2D(std::size_t width, std::size_t height);

GLuint createTexture3D(std::size_t width, std::size_t height, std::size_t depth);
GLuint createWeatherTexture3D(std::size_t width, std::size_t height, std::size_t depth);

GLuint createTexture3D(std::size_t width, std::size_t height, std::size_t depth, float *data);

GLuint createTextureFromFile(std::string path);
GLuint createGroundTextureFromFile(std::string path);

void bindTexture2D(GLuint textureID, unsigned int textureUnit);
void bindGroundTexture2D(GLuint textureID, unsigned int textureUnit);

void bindTexture3D(GLuint textureID, unsigned int textureUnit);