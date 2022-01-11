#include <iostream>
#include <filesystem>
#include <random>
#define _USE_MATH_DEFINES
#include <math.h>

#include "Engine/Defs.h"

class particle 
{
    public:
        float density;
        float pressure;
        glm::vec3 pressureGradient;
        float mass; //130.9 * (radius^3 /  96)
        float viscocity;

        glm::vec3 position;
        glm::vec3 velocity;
        glm::vec3 convecAccel;
};

//placeholders!
std::vector<particle> neighbours = {};
particle thisParticle;

float particleRadius = 0.3f;
float smoothingRadius = particleRadius / 4;

float constantK = 1481.f;
float restingDensity = 1000.f;

glm::vec3 gravity = glm::vec3(0.f,-9.8f,0.f);

void calcMass()
{
    //mass

    thisParticle.mass = ( (4 * M_PI * (pow(smoothingRadius, 3)) / (3 * neighbours.size()) ) * restingDensity);
}

glm::vec3 viscousTerm;

void sph_Density()
{
    //equation (2): denstiy

    float resultDensity = 0.f;

    for (int j = 0; j < neighbours.size(); j++)
    {
        float temp = static_cast<float>(64 * M_PI * pow(smoothingRadius, 9));

        glm::vec3 tempVector = thisParticle.position - neighbours[j].position;

        resultDensity += neighbours[j].mass * ( 315 / temp ) * glm::pow( (glm::pow(smoothingRadius, 2) - glm::pow(glm::length(tempVector), 2)), 3);
    }
    thisParticle.density = resultDensity;
}

void sph_Pressure()
{
    //pressure

    float pressure = constantK * (thisParticle.density * restingDensity);
}

void sph_PressureGradient()
{
    //equation(3): pressure gradient

    glm::vec3 resultPG;

    for (int j = 0; j < neighbours.size(); j++)
    {
        float thisTemp = (thisParticle.pressure / pow(thisParticle.density, 2));

        float neighbourTemp = (neighbours[j].pressure / pow(neighbours[j].density, 2));

        glm::vec3 tempVector = thisParticle.position - neighbours[j].position;

        resultPG += (tempVector / glm::length(tempVector)) * static_cast<float>(neighbours[j].mass * ( thisTemp + neighbourTemp ) * ( -45 / (M_PI * pow(smoothingRadius, 6))) * pow(smoothingRadius - glm::length(tempVector), 2));
    }
    thisParticle.pressureGradient = resultPG;
}

void sph_Viscosity()
{
    //equation(4): viscocity

    glm::vec3 resultV;

    for (int j = 0; j < neighbours.size(); j++)
    {
        glm:: vec3 tempVector = thisParticle.position - neighbours[j].position;

        resultV += ( ( neighbours[j].velocity - thisParticle.velocity) / neighbours[j].density)* static_cast<float>(neighbours[j].mass * ( 45 / (M_PI * pow(smoothingRadius, 6))) * (smoothingRadius - glm::length(tempVector)));   
    }
    viscousTerm = (thisParticle.viscocity / thisParticle.density) * resultV;
}

void sph_Acceleration()
{
    //equation (1) : acceleration

    glm::vec3 acceleration = gravity - thisParticle.pressureGradient + viscousTerm;
}

    

    

    

    
