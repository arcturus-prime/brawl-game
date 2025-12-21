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

#define LEAF 0x80000000
#define EMPTY 0xFFFFFFFF
#define MAX_DEPTH 256
#define RAY_CLIPPING_DISTANCE 1e9

#define STEEL 0x0
#define PLASTIC 0x1
#define GLASS 0x2

struct BSPNode {
    vec4 plane;
    uint positive;
    uint negative;
    uint padding1;
    uint padding2;
};

layout(set = 0, binding = 3) uniform readonly Geometry {
    BSPNode nodes[1024];
} geometry;

struct StackElement {
    float t_min;
    float t_max;
    uint node;
    vec3 normal;
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
            RAY_CLIPPING_DISTANCE,
            0,
            vec3(0.0, 0.0, 0.0)
        );

    uint stack_pointer = 1;
    uint hit = 0;

    while (stack_pointer != 0 && stack_pointer < MAX_DEPTH) {
        stack_pointer--;
        StackElement context = stack[stack_pointer];

        if (context.node == EMPTY) {
            continue;
        } else if (context.node >= LEAF) {
            hit = 1;
            break;
        }

        BSPNode node = geometry.nodes[context.node];

        vec3 start = ray_origin + ray_direction * context.t_min;
        vec3 stop = ray_origin + ray_direction * context.t_max;

        float start_distance = plane_distance(start, node.plane.xyz, node.plane.w);
        float end_distance = plane_distance(stop, node.plane.xyz, node.plane.w);

        if (start_distance >= 0.0 && end_distance >= 0.0) {
            stack[stack_pointer] =
                StackElement(
                    context.t_min, context.t_max, node.positive, context.normal
                );
            stack_pointer++;
        } else if (start_distance < 0.0 && end_distance < 0.0) {
            stack[stack_pointer] =
                StackElement(
                    context.t_min, context.t_max, node.negative, context.normal
                );
            stack_pointer++;
        } else
        {
            float t_split = intersect_plane(ray_origin, ray_direction, node.plane.xyz, node.plane.w);

            if (start_distance >= 0.0) {
                stack[stack_pointer] =
                    StackElement(
                        t_split, context.t_max, node.negative, node.plane.xyz
                    );
                stack[stack_pointer + 1] =
                    StackElement(
                        context.t_min, t_split, node.positive, context.normal
                    );

                stack_pointer += 2;
            } else {
                stack[stack_pointer] =
                    StackElement(
                        t_split, context.t_max, node.positive, node.plane.xyz
                    );
                stack[stack_pointer + 1] =
                    StackElement(
                        context.t_min, t_split, node.negative, context.normal
                    );

                stack_pointer += 2;
            }
        }
    }

    float depth_previous = imageLoad(depth_img, pixel).r;
    float depth_current = stack[stack_pointer].t_min;

    if (depth_current > depth_previous || hit == 0) {
        imageStore(img, pixel, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }

    imageStore(img, pixel, vec4(1.0, 0.0, 0.0, 1.0));
    imageStore(depth_img, pixel, vec4(depth_current, 0.0, 0.0, 0.0));
}
