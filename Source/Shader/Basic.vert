#version 450 core

layout (location = 0) in vec3 aPos;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

//out float vertexID;
out vec3 passPos;

void main()
{
    gl_Position = projectionMatrix * viewMatrix * vec4(aPos, 1.0);
    // vertexID = gl_VertexID;
    passPos = aPos;
}