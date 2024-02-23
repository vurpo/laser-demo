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
    x: f32,
    transition: f32
}

@group(1) @binding(1)
var<uniform> shader_params: ShaderParams;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = (model.tex_coords+vec2(-1.,1.))*vec2(-.5,.5);
    out.clip_position = vec4(model.position*vec3(-1.0,-1.0,1.0), 1.0);
    return out;
}

// Fragment shader

@group(0) @binding(0)
var texture_new: texture_2d<f32>;
@group(0) @binding(1)
var sampler_new: sampler;

@group(2) @binding(0)
var texture_old: texture_2d<f32>;
@group(2) @binding(1)
var sampler_old: sampler;

const distortion: f32 = 0.2;

fn radialDistortion(coord: vec2<f32>) -> vec2<f32> {
  var cc: vec2<f32> = coord;
  var dist: f32 = dot(cc, cc) * distortion;
  return coord + cc * (1.0 - dist) * dist;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    switch shader_params.shader_function {
        case 0: { return textureSample(texture_new, sampler_new, in.tex_coords); }
        case 1: { return fade_transition(in.tex_coords); }
        case 2: { return slide_transition(in.tex_coords); }
        default: { return vec4(0.0); }
    }
}

fn fade_transition(uv: vec2<f32>) -> vec4<f32> {
    let transition = clamp(shader_params.transition, 0., 1.);
    return transition*textureSample(texture_new, sampler_new, uv)
        + (1.-transition)*textureSample(texture_old, sampler_old, uv);
}

fn slide_transition(uv: vec2<f32>) -> vec4<f32> { 
    let transition = 1.-smoothstep(0.,1.,shader_params.transition);
    let uv_1 = uv-vec2(transition,0.);
    let uv_2 = uv_1+vec2(1.0,0.0);
    
    let sample_1 = step(0.0,uv_1.x)*textureSample(texture_new, sampler_new, uv_1);
    let sample_2 = (1.0-step(1.0,uv_2.x))*textureSample(texture_old, sampler_old, uv_2);
    return vec4(0.)
        +sample_1
        +sample_2;
}
