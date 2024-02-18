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

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = (model.tex_coords+vec2(1.0,1.0))*vec2(0.5,0.5);
    out.clip_position = vec4(model.position*vec3(-1.0,-1.0,1.0), 1.0);
    return out;
}

// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

const distortion: f32 = 0.2;

fn radialDistortion(coord: vec2<f32>) -> vec2<f32> {
  var cc: vec2<f32> = coord;
  var dist: f32 = dot(cc, cc) * distortion;
  return coord + cc * (1.0 - dist) * dist;
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var uv: vec2<f32> = in.tex_coords*vec2(2.0,2.0)-vec2(1.0,1.0);
    //uv = radialDistortion(uv);
    uv = uv*vec2(0.5,0.5)+vec2(0.5,0.5);
    return textureSample(t_diffuse, s_diffuse, uv);
}