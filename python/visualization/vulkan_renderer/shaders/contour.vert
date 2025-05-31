#version 450
layout(location = 0) in vec2 inPosition;
layout(location = 1) in float inValue;
layout(location = 0) out float outValue;

void main() {
    gl_Position = vec4(inPosition, 0.0, 1.0);
    outValue = inValue;
}