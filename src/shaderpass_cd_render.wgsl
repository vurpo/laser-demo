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
const STEPS_START: i32 = 100;
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

fn sample(coords: vec3<f32>) -> f32 {
    let c0 = vec3<i32>(floor(coords));
    let packed = textureLoad(smoke, c0, 0);
    let u00 = unpack2x16float(packed.x);
    let u01 = unpack2x16float(packed.y);
    let u10 = unpack2x16float(packed.z);
    let u11 = unpack2x16float(packed.w);
    let c1 = vec3<i32>(floor(coords))+vec3<i32>(1,1,1);
    let cd: vec3<f32> = fract(coords);

    let s00: f32 = mix(u00.x, u00.y, cd.x);
    let s01: f32 = mix(u01.x, u01.y, cd.x);
    let s10: f32 = mix(u10.x, u10.y, cd.x);
    let s11: f32 = mix(u11.x, u11.y, cd.x);

    let s0: f32 = mix(s00, s10, cd.y);
    let s1: f32 = mix(s01, s11, cd.y);

    return mix(s0, s1, cd.z);
    //return u00.x;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    
}
