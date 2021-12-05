#version 450 core

layout (location = 0) in vec2 pos;
layout (location = 1) in vec2 tc;


out vec2 uvCoords;

void main(){
    gl_Position = vec4(pos.xy, 0.0, 1.0);
    uvCoords = tc;
}