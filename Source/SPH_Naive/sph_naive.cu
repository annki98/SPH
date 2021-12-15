#include <iostream>
#include <filesystem>
#include <random>

class particle 
{
    public:
        float density;
        float pressure;
        float pressureGradient;
        float mass;
        float viscocity;

        glm::vec3 position;
        glm::vec3 velocity;
        glm::vec3 convecAccel;
}

#define _USE_MATH_DEFINES
#include<cmath>

float radius;
glm::vec3 gravity; //no bleeding idea how to represent gravity as a vector

//placeholders!
std::vector<particle> neighbours = {};
particle thisParticle;


void sphNaive()
{
    //equation (2): denstiy

    float resultDensity;

    for (int j = 0; j < neighbours.size(); j++)
    {
        float temp = 64 * M_PI * pow(radius, 9);

        glm::vec3 tempVector = thisParticle.position - neighbours[j].position;

        resultDensity += neighbours[j].mass * ( 315 / temp ) * pow( (pow(radius, 2) - pow(glm::length(tempVector), 2)), 3);
    }
    thisParticle.density = resultDensity;

    //pressure

    float constantK; //need to find out what the value of this goddamn constant is
    float restingDensity; //same for this thing
    float pressure = constantK * (thisParticle.density * restingDensity);

    //equation(3): pressure gradient

    float resultPG;

    for (int j = 0; j < neighbours; j++)
    {
        float thisTemp = (thisParticle.pressure / pow(thisParticle.density, 2));

        float neighbourTemp = (neighbours[j].pressure / pow(neighbours[j].density, 2));

        glm::vec3 tempVector = thisParticle.position - neighbours[j].position;

        resultPG += neighbours[j].mass * ( thisTemp + neighbourTemp ) * ( -45 / (M_PI * pow(radius, 6))) * pow(radius - glm::length(tempVector), 2) *  (tempVector / glm::length(tempVector));
    }
    thisParticle.pressureGradient = resultPG;

    //equation(4): viscocity

    float resultV;

    for (int j = 0; j < neighbours.size(); j++)
    {
        glm:: vec3 tempVector = thisParticle.position - neighbours[j].position;

        resultV += neighbours[j].mass * ( ( neighbours[j].velocity - thisParticle.velocity) / neighbours[j].density) * ( 45 / (M_PI * pow(radius, 6))) * (radius - glm::length(tempVector));   
    }
    glm::vec3 viscousTerm = (thisParticle.viscocity / thisParticle.density) * resultV;

    //equation (1) : acceleration

    glm::vec3 acceleration = gravity - thisParticle.pressureGradient + viscousTerm;
}