#version 450 core

layout (location = 0) in vec3 pos;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

uniform float sphereRadius;
uniform vec2 resolution;

out vec2 passPointPos;
out vec4 passPos;
out float passPointSize;

const float PI = 3.1415926535897932384626433832795;


void main()
{
    gl_Position = projectionMatrix * viewMatrix * vec4(pos, 1.0);
    passPos = gl_Position;

    vec3 posEye = vec3(viewMatrix * vec4(pos, 1.0));
    float dist = length(posEye);
    float scaleNormalization = resolution.y / tan(90.0 * PI / 180.0);
    passPointSize = sphereRadius * (scaleNormalization / dist);

    vec2 posDepthless = gl_Position.xy / gl_Position.w;
    passPointPos = resolution * (posDepthless * 0.5 + 0.5);

}
