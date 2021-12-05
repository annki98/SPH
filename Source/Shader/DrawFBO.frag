#version 450 core

in vec2 uvCoords;
uniform sampler2D fbo;

out vec4 fragmentColor;

void main()
{
    fragmentColor = texture(fbo, gl_FragCoord.xy);
}