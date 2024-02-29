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
struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9) tex_offset: vec2<f32>,
    @location(10) dot: f32
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>
}

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
    out.tex_coords = (model.tex_coords+vec2(1.,-1.))*vec2(.5,-.5);
    out.clip_position = camera.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
    return out;
}

@group(0) @binding(0)
var t1: texture_2d<f32>;
@group(0) @binding(1)
var s1: sampler;
@group(2) @binding(0)
var t2: texture_2d<f32>;
@group(2) @binding(1)
var s2: sampler;

const PI: f32 = 3.1415926536;
const HALF_PI: f32 = PI * 0.5; 
const TAU: f32 = PI * 2.0;
const FOV: vec2<f32> = vec2(0.3,0.3);

fn project(coord: vec2<f32>, lookat: vec2<f32>, fov: vec2<f32>) -> vec2<f32> {
    // fragment coordinate mungled to have the FOV and stuff
    let c = coord * fov * vec2(PI, HALF_PI);

    // fragment coordinate in polar coordinates from center of screen
    let frag_radius = length(c);
    let frag_angle = atan(frag_radius);
    
    // direction (lat lon) on equirectangular environment
    return vec2(
        // longitude
        lookat.x+atan2(
            c.x*sin(frag_angle),
            frag_radius*cos(lookat.y)*cos(frag_angle) - c.y*sin(lookat.y)*sin(frag_angle)
        )+PI,
        // latitude
        asin(
            cos(frag_angle)*sin(lookat.y)+
            (c.y*sin(frag_angle)*cos(lookat.y))
            /frag_radius
        )+HALF_PI
    );
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let t = shader_params.time*1.;
    let lookat = vec2(
        t+1.5*sin(t*3.)+10.*sin(t/10.),
        .6*sin(t*5.)
    );
    let dir = project(in.tex_coords, lookat, FOV);
    return vec4(textureSample(t1, s1, dir).rgb, 1.0);
}
