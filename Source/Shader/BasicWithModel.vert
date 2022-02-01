#version 450 core

layout (location = 0) in vec4 aPos;
layout (location = 1) in vec3 normalAttribute;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 modelMatrix;




out vec4 passPosition;
out vec3 passNormal;

void main(){
	// the vertex has to be rasterized so we can do phong stuff
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * aPos;

    //pass normal and position in viewspace
	passPosition = viewMatrix * modelMatrix * aPos;
	vec4 n = transpose(inverse(viewMatrix * modelMatrix)) * vec4(normalAttribute, 0);
	passNormal = normalize(n.xyz);
}