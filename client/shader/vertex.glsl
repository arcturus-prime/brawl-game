#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

layout(set = 0, binding = 0) uniform Camera {
    mat4 view;
    mat4 proj;
};

layout(push_constant) uniform Push {
    mat4 model;
};

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec2 out_uv;

void main() {
    out_normal = mat3(transpose(inverse(model))) * normal;
    out_uv = uv;
    gl_Position = proj * view * model * vec4(position, 1.0);
}
