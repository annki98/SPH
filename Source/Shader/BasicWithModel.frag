#version 450 core
out vec4 fragColor;

in vec4 passPosition;
in vec3 passNormal;

const vec3 lightPosition = vec3(500.0f, 500.0f, 500.0f);

void main(){
    //compute the light vector as the normalized vector between 
    //the vertex position and the light position:
    vec3 lightVector = normalize(lightPosition - passPosition.xyz);

    //compute the eye vector as the normalized negative vertex position in camera coordinates:
    vec3 eye = normalize(-passPosition.xyz);
    
    //compute the normalized reflection vector using glsl's reflect function:
    vec3 reflection = normalize(reflect(-lightVector, normalize(passNormal)));

    //variables used in the phong lighting model:
    float phi = max(dot(passNormal, lightVector), 0);
    float psi = pow(max(dot(reflection, eye), 0), 15);

	// colors for the different phong effects
    vec3 ambientColor = vec3(0.3, 0.2, 0.2);
    vec3 diffuseColor = vec3(1.0, 0.0, 0.0);
    vec3 specularColor = vec3(1.0, 1.0, 1.0);

    fragColor = vec4(
       ambientColor +
       phi * diffuseColor + 
       psi * specularColor,
        1);
       
}