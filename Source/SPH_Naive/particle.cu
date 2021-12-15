#include<iostream>
using namespace std;
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