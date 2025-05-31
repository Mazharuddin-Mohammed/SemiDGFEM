#version 450
layout(location = 0) in vec2 inVector;
layout(location = 0) out vec4 outColor;

void main() {
    float mag = length(inVector);
    outColor = vec4(mag, 0.0, 1.0 - mag, 1.0);
}