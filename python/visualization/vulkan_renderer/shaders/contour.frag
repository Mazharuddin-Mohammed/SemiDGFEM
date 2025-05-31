#version 450
layout(location = 0) in float inValue;
layout(location = 0) out vec4 outColor;

void main() {
    // Simple viridis colormap
    vec3 color = mix(vec3(0.0, 0.0, 0.5), vec3(1.0, 1.0, 0.0), inValue);
    outColor = vec4(color, 1.0);
}