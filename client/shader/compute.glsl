#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform PushData {
    vec3 position;
    vec4 rotation;
} push_data;
layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;
layout(set = 0, binding = 1, r32f) uniform image2D depth_img;
layout(set = 0, binding = 2) uniform readonly ViewData {
    uint resolution_x;
    uint resolution_y;
    vec3 camera_position;
    float fov;
    vec4 camera_rotation;
} view_data;

#define SOLID 0xFFFFFFFE
#define EMPTY 0xFFFFFFFF
#define MAX_DEPTH 64

struct BSPNode {
    vec4 plane;
    uint positive;
    uint negative;
    uint metadata1;
    uint metadata2;
};

layout(set = 0, binding = 3) buffer readonly Geometry {
    BSPNode nodes[];
} geometry;

struct StackElement {
    float t_min;
    float t_max;
    uint node;
    uint first_entered;
};

vec3 rotate_vector(vec3 v, vec4 q) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

vec4 quat_inv(vec4 q) {
    return vec4(-q.xyz, q.w);
}

void transform_ray_to_local(inout vec3 origin, inout vec3 direction) {
    origin -= push_data.position;
    vec4 inv_rot = quat_inv(push_data.rotation);

    origin = rotate_vector(origin, inv_rot);
    direction = rotate_vector(direction, inv_rot);
}

float intersect_plane(vec3 origin, vec3 direction, vec3 normal, float distance) {
    float denom = dot(direction, normal);

    if (abs(denom) < 1e-6) return -1.0;

    float t = (distance - dot(origin, normal)) / denom;

    return t >= 0.0 ? t : -1.0;
}

float plane_distance(vec3 point, vec3 normal, float d) {
    return dot(point, normal) - d;
}

vec4 getColor(uint metadata1, uint metadata2, vec3 position, vec3 normal) {
    return vec4(normal * 0.5 + vec3(0.5, 0.5, 0.5), 1.0);
}

#define GUARD_PUSH(T_MIN, T_MAX, NODE, FIRST) \
    if ((NODE) != EMPTY && stack_pointer < MAX_DEPTH) { \
        stack[stack_pointer] = StackElement((T_MIN), (T_MAX), (NODE), (FIRST)); \
        stack_pointer++; \
    }

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= view_data.resolution_x || pixel.y >= view_data.resolution_y) return;

    float aspect = float(view_data.resolution_x) / float(view_data.resolution_y);
    vec2 uv = (vec2(pixel) + 0.5) / vec2(view_data.resolution_x, view_data.resolution_y);
    uv = (uv * 2.0 - 1.0);
    uv.x *= aspect;

    float focal_length = 1.0 / tan(radians(view_data.fov) * 0.5);

    vec3 ray_direction = normalize(vec3(focal_length, uv));
    ray_direction = rotate_vector(ray_direction, view_data.camera_rotation);

    vec3 ray_origin = view_data.camera_position;
    transform_ray_to_local(ray_origin, ray_direction);

    StackElement stack[MAX_DEPTH];
    stack[0] =
        StackElement(
            0.0,
            1e9,
            0,
            0
        );

    uint stack_pointer = 1;
    uint hit = 0;

    while (stack_pointer != 0 && stack_pointer < MAX_DEPTH) {
        stack_pointer--;
        StackElement context = stack[stack_pointer];

        if (context.node == SOLID) {
            hit = 1;
            break;
        }

        BSPNode node = geometry.nodes[context.node];

        vec3 start = ray_origin + ray_direction * context.t_min;
        vec3 stop = ray_origin + ray_direction * context.t_max;

        float start_distance = plane_distance(start, node.plane.xyz, node.plane.w);
        float end_distance = plane_distance(stop, node.plane.xyz, node.plane.w);

        if (start_distance >= 0.0 && end_distance >= 0.0) {
            GUARD_PUSH(context.t_min, context.t_max, node.positive, context.first_entered);
        } else if (start_distance < 0.0 && end_distance < 0.0) {
            GUARD_PUSH(context.t_min, context.t_max, node.negative, context.first_entered);
        } else
        {
            float t_split = intersect_plane(ray_origin, ray_direction, node.plane.xyz, node.plane.w);

            if (start_distance >= 0.0) {
                GUARD_PUSH(t_split, context.t_max, node.negative, context.node);
                GUARD_PUSH(context.t_min, t_split, node.positive, context.first_entered);
            } else {
                GUARD_PUSH(t_split, context.t_max, node.positive, context.node);
                GUARD_PUSH(context.t_min, t_split, node.negative, context.first_entered);
            }
        }
    }

    float depth_previous = imageLoad(depth_img, pixel).r;
    float depth_current = stack[stack_pointer].t_min;

    if (hit == 0 || depth_current > depth_previous) {
        return;
    }

    uint first_index = stack[stack_pointer].first_entered;
    float t_impact = stack[stack_pointer].t_min;

    uint metadata1 = geometry.nodes[first_index].metadata1;
    uint metadata2 = geometry.nodes[first_index].metadata2;
    vec3 normal = geometry.nodes[first_index].plane.xyz;

    vec4 color = getColor(metadata1, metadata2, ray_origin + ray_direction * t_impact, normal);

    imageStore(img, pixel, color);
    imageStore(depth_img, pixel, vec4(depth_current, 0.0, 0.0, 0.0));
}
