//
// Created by maxbr on 10.05.2020.
//

#pragma once

#include "../Engine/Drawable.h"

class Cube : public Drawable
{
public:
    Cube(float size);
    void draw();

private:
    void create(float size);

    int m_points; 
	int m_indices;
};

