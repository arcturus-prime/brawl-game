#version 450

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

void main() {
    vec3 light_dir = normalize(vec3(1.0, 2.0, 3.0));
    float diffuse = max(dot(normalize(in_normal), light_dir), 0.0);
    out_color = vec4(vec3(0.6 + 0.4 * diffuse), 1.0);
}
