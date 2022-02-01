#version 450 core
out vec4 FragColor;

in float vertexID;

void main()
{
    FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
    if (fract(vertexID) < 0.1){
    FragColor = vec4(1,0,0,0);
}
}