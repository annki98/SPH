#version 450 core
out vec4 FragColor;

uniform float fullGridSize;
// in float vertexID;
uniform vec3 color;

in vec3 passPos;

void main()
{
    FragColor = (2*(passPos.y/fullGridSize) + 0.5f) * vec4(color, 0);
    // if (fract(vertexID) < 0.1){
    //     FragColor = vec4(1,0,0,0);
    // }
}