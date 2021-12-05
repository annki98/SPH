//
// Created by maxbr on 10.05.2020.
//

#pragma once

#include "../Engine/Drawable.h"

class Triangle : public Drawable
{
public:
    Triangle();

private:
    void create(glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 na, glm::vec3 nb, glm::vec3 nc, glm::vec2 tca,
                glm::vec2 tcb, glm::vec2 tcc);

};

