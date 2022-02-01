#version 450 core

in vec2 passPointPos;
in vec4 passPos;

uniform float sphereRadius;
uniform vec2 resolution;

out vec4 FragColor;

const vec3 lightDir = vec3(0.6, 0.6, 0.7);
const vec4 baseColor = vec4(0.4, 0.7, 0.82, 0.2);

float map(float value, float min1, float max1, float min2, float max2) {
    return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

void main()
{
    vec2 sphereConstraint = (passPointPos - gl_FragCoord.xy) * passPos.w;
    float rSquare = dot(sphereConstraint, sphereConstraint) / (sphereRadius * 300.0); // offset for sphere radius
    
    if (rSquare > 1.0) 
        discard; // draw as circles

    vec3 normal;
    normal.xy = gl_PointCoord * vec2(2.0, -2.0) + vec2(-1.0, 1.0);

    // soft normals over sphere radius
    normal.z = sqrt(1.0 - rSquare);

    // lighting
    float diffuse = max(0.0, dot(normal, lightDir));

    FragColor = diffuse * baseColor;
}