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

fn normpdf(x: f32, sigma: f32) -> f32 {
	return 0.39894*exp(-0.5*x*x/(sigma*sigma))/sigma;
}

const mSize = 17;
const kSize = 8;
const sigma = 7.0;

fn gauss(coord: vec2<f32>) -> vec3<f32> {
    let c = coord*vec2(1920.0,1080.0);
    var kernel: array<f32, mSize> = array<f32, mSize>(0, 0.034567048769393996, 0.039470515171203084, 0.044159091982669076, 0.048406571265966085, 0.051990663058568545, 0.05471208153520407, 0.05641284038284459, 0.056991428571428575, 0.05641284038284459, 0.05471208153520407, 0.051990663058568545, 0.048406571265966085, 0.044159091982669076, 0.039470515171203084, 0.034567048769393996, 0);
    var final_color = vec3(0.0);
    //var Z = 0.0;
    // for (var j=0; j < kSize; j++) {
    //     let f = normpdf(f32(j), sigma);
    //     kernel[kSize+j] = f;
    //     kernel[kSize-j] = f;
    // }
    // for (var j = 0; j < mSize; j++) {
    //     Z += kernel[j];
    // }
    for (var i=-kSize; i <= kSize; i++) {
        for (var j=-kSize; j <= kSize; j++) {
            final_color += kernel[kSize+j]*kernel[kSize+i]*textureSample(texture_new, sampler_new, (c+2.2*vec2(f32(i),f32(j)))/vec2(1920.0,1080.0)).rgb;
        }
    }
    return final_color;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(5.*gauss(in.tex_coords),1.0)+10.*textureSample(texture_new, sampler_new, in.tex_coords);
}
