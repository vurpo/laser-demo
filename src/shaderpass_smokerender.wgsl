const pi = 3.14159265359;

// Vertex shader

struct Camera {
    view_proj: mat4x4<f32>,
}
@group(1) @binding(0)
var<uniform> camera: Camera;

struct ShaderParams {
    shader_function: i32,
    time: f32,
    x: f32
}

@group(1) @binding(1)
var<uniform> shader_params: ShaderParams;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}
struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9) tex_offset: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

const STEPS_END: i32 = 300;
const STEPS_START: i32 = 90;
const STEP: f32 = 0.5;
const ALPHA: f32 = 0.02;

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    var out: VertexOutput;
    out.tex_coords = model.tex_coords+instance.tex_offset;
    out.clip_position = camera.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
    out.clip_position.z = 0.0;
    return out;
}

// Fragment shader

@group(0) @binding(0)
var smoke: texture_3d<f32>;

fn sample(coords: vec3<f32>) -> vec4<f32> {
    let c0 = vec3<i32>(floor(coords));
    let c1 = vec3<i32>(floor(coords))+vec3<i32>(1,1,1);
    let cd: vec3<f32> = fract(coords);

    let s00 = mix(textureLoad(smoke, vec3<i32>(c0.x, c0.y, c0.z), 0), textureLoad(smoke, vec3<i32>(c1.x, c0.y, c0.z), 0), cd.x);
    let s01 = mix(textureLoad(smoke, vec3<i32>(c0.x, c0.y, c1.z), 0), textureLoad(smoke, vec3<i32>(c1.x, c0.y, c1.z), 0), cd.x);
    let s10 = mix(textureLoad(smoke, vec3<i32>(c0.x, c1.y, c0.z), 0), textureLoad(smoke, vec3<i32>(c1.x, c1.y, c0.z), 0), cd.x);
    let s11 = mix(textureLoad(smoke, vec3<i32>(c0.x, c1.y, c1.z), 0), textureLoad(smoke, vec3<i32>(c1.x, c1.y, c1.z), 0), cd.x);

    let s0 = mix(s00, s10, cd.y);
    let s1 = mix(s01, s11, cd.y);

    return mix(s0, s1, cd.z);
}

fn rotation(theta: f32) -> mat3x3f {
    return mat3x3f(cos(theta), sin(theta), 0., -sin(theta), cos(theta), 0., 0., 0., 1.);
}

fn laser(position: vec3<f32>, offset: vec2<f32>) -> f32 {
    return abs(1.0/max(0.11, (length(position.yz-offset)-(0.1))));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.tex_coords*vec2(16.0/9.0, 1.0);

    let dimensions: vec3<i32> = vec3<i32>(textureDimensions(smoke));
    let steps = STEPS_END-STEPS_START;

    var s_ambient: f32 = 0.0;
    var s_laser: f32 = 0.0;

    let r = rotation(shader_params.time);
    let dir: vec3<f32> = r*(normalize(vec3(1.5, uv.x, uv.y))*STEP);
    var p: vec3<f32> = (r*vec3(-120.0,0.0,f32(dimensions.z)/2.0))+vec3(f32(dimensions.x/2), f32(dimensions.y/2), 0.0)+f32(STEPS_START)*dir;
    for (var i=0; i<steps; i++) {
        p += dir;
        let center = vec2(f32(dimensions.x)/2.0, f32(dimensions.z)/2.0);
        var l0 = 0.0;
        l0 = max(l0, laser(p, center+vec2(0.0, 15.0)));
        let l1 = 3.0*l0;

        let s0 = sample(p).a*STEP*ALPHA;
        s_ambient += 0.2*s0;
        s_laser += l1*s0;
    }
    // let uv = (in.tex_coords+vec2(-1.0,1.0)) * vec2(f32(dimensions.x)/-2.0, f32(dimensions.z)/2.0);
    // let s = sample(vec3<f32>(uv.x,f32(dimensions.y)/2.0,uv.y));
    // //return vec4(s.rgb*s.a, 1.0);
    return vec4(
        s_ambient*vec3(1.0,1.0,1.0)+
        //s_laser*vec3(1.0,0.0,0.0), 1.0);
        max(vec3(0.0,0.0,0.0), vec3(1.0,0.3,0.3)-(1.0-s_laser)), 1.0);
    // //return vec4(uv.x, uv.y, 0.0, 1.0);
}