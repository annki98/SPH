#version 450 core

uniform vec2 resolution;
in vec2 uvCoords;
uniform sampler2D fbo;

out vec4 fragmentColor;

void main()
{
    fragmentColor = texture(fbo, vec2(gl_FragCoord.x / resolution.x, gl_FragCoord.y / resolution.y));
}