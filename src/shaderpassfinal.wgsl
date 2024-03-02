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
    transition: f32,
    x2: f32,
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

const STEPS: i32 = 2;

fn texture_radialblur(t: texture_2d<f32>, s: sampler, uv: vec2<f32>) -> vec4<f32> {
    if shader_params.x2 == 0.0 {
       return textureSample(t, s, uv);
    } else {
        let uv2 = uv-0.5;
        let ratio = 1./(9.);
        var out_color: vec3<f32> = vec3(0.);//textureSample(t, s, uv).xyz*ratio;
        for (var i=-4;i<=4;i++) {
            out_color += textureSample(t, s, uv2*(f32(i)*shader_params.x2+1.)+0.5).xyz*ratio;
        }
        return vec4(out_color*(1.+shader_params.x2), 1.0);
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var out_color = vec4(0.0);
    switch shader_params.shader_function {
        case 0: { out_color = texture_radialblur(texture_new, sampler_new, in.tex_coords); }
        case 1: { out_color = fade_transition(in.tex_coords); }
        case 2: { out_color = slide_transition(in.tex_coords); }
        case 3: { out_color = blink_transition(in.tex_coords); }
        default: { out_color = vec4(0.0); }
    }
    return out_color+shader_params.x;
}

fn fade_transition(uv: vec2<f32>) -> vec4<f32> {
    let transition = clamp(shader_params.transition, 0., 1.);
    return transition*texture_radialblur(texture_new, sampler_new, uv)
        + (1.-transition)*texture_radialblur(texture_old, sampler_old, uv);
}

fn slide_transition(uv: vec2<f32>) -> vec4<f32> { 
    let transition = 1.-smoothstep(0.,1.,shader_params.transition);
    let uv_1 = uv-vec2(transition,0.);
    let uv_2 = uv_1+vec2(1.0,0.0);
    
    let sample_1 = step(0.0,uv_1.x)*texture_radialblur(texture_new, sampler_new, uv_1);
    let sample_2 = (1.0-step(1.0,uv_2.x))*texture_radialblur(texture_old, sampler_old, uv_2);
    return sample_1+sample_2;
}

fn blink_transition(uv: vec2<f32>) -> vec4<f32> {
    let y = 1.-abs(uv.y*2.-1.)+.1*cos(uv.x*2.-1.);
    var fade = abs(smoothstep(0.,1.,shader_params.transition)-.5)*2.;
    fade = fade*smoothstep(.75-fade,1.-fade,y);
    if shader_params.transition<0.5 {
        return texture_radialblur(texture_old, sampler_old, uv)*fade;
    } else {
        return texture_radialblur(texture_new, sampler_new, uv)*fade;
    }
}
