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
    if shader_params.shader_function == 2 {
        out.clip_position.z = 0.0;
    }
    return out;
}

// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    switch shader_params.shader_function {
        case 0: { return simple_texture(in); }
        case 1: { return cool_background(in); }
        case 2: { return fade_to_white(in); }
        case 3: { return trans(in); }
        case 4: { return another_cool_background(in); }
        case 5: { return cool_background_3(in); }
        case 6: { return sky_1(in); }
        case 7: { return sky_2(in); }
        case 8: { return sky_3(in); }
        case 9: { return sky_4(in); }
        default: { return vec4(1.0, 0.0, 0.0, 1.0); }
    }
}

fn simple_texture(in: VertexOutput) -> vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex_coords);
}

fn sky_1(in: VertexOutput) -> vec4<f32> {
    let x = cos(in.tex_coords.y*2.5)*0.5+0.5;
    return vec4(-0.3+(x*1.3), x, 1.0, 1.0);
}

fn sky_2(in: VertexOutput) -> vec4<f32> {
    let x = cos(in.tex_coords.y*2.5)*0.5+0.5;
    return vec4(x*0.7, x*0.2, 0.7-x, 1.0);
}

fn sky_3(in: VertexOutput) -> vec4<f32> {
    let x = sin(in.tex_coords.y);
    return vec4(x, x, x*0.5+0.5, 1.0);
}

fn sky_4(in: VertexOutput) -> vec4<f32> {
    let x = sin(in.tex_coords.y);
    return vec4(1.0, 1.0-x, 1.0-x, 1.0);
}

fn cool_background(in: VertexOutput) -> vec4<f32> {
    let normalized_coords = in.tex_coords * vec2(16.0/9.0, 1.0);
    let angle = 5.0*atan2(normalized_coords.y, normalized_coords.x);
    return vec4(
        sin(4.0*shader_params.time+angle+sin(length(normalized_coords)*10.0))/2.0+0.5,
        0.0,
        sin(-4.0*shader_params.time+angle+cos(length(normalized_coords)*10.0))/2.0+0.5,
        1.0);
}

fn another_cool_background(in: VertexOutput) -> vec4<f32> {
    let normalized_coords = in.tex_coords * vec2(16.0/9.0, 1.0);
    let angle = 5.0*atan2(normalized_coords.y, normalized_coords.x);
    let x = sin(length(normalized_coords)*10.0-(10.0*shader_params.time)+(1.5*sin(angle*2.0))+sin(angle*3.0+shader_params.time))*0.25+0.3;
    return vec4(
        x,
        x,
        1.0,
        1.0);
}

fn cool_background_3(in: VertexOutput) -> vec4<f32> {
    let normalized_coords = in.tex_coords * vec2(16.0/9.0, 1.0);
    //let angle = 5.0*atan2(normalized_coords.y, normalized_coords.x);
    //let x = sin(length(normalized_coords)*10.0-(10.0*shader_params.time)+(1.5*sin(angle*2.0))+sin(angle*3.0+shader_params.time))*0.25+0.3;
    return vec4(
        sin(length(normalized_coords-vec2(-16.0/9.0, -1.0))*10.0-shader_params.time*10.0),
        0.0,
        sin(length(normalized_coords-vec2(16.0/9.0, -1.0))*10.0+shader_params.time*10.0),
        1.0);
}

fn trans(in: VertexOutput) -> vec4<f32> {
    let normalized_coords = in.tex_coords * vec2(16.0/9.0, 1.0);
    let angle = 3.0*shader_params.time+8.0*atan2(normalized_coords.y, normalized_coords.x)/pi;
    
    var c = array<vec3<f32>, 3>(vec3(1.0, 1.0, 1.0),vec3(1.0,.57,.62),vec3(.357,.808,.98));
    let col = c[u32(2.0*triangle(angle)+0.5)];
    return vec4(
        col,
        1.0);
}

fn fade_to_white(in: VertexOutput) -> vec4<f32> {
    return vec4(1.0, 1.0, 1.0, shader_params.x);
}

fn step2(edge1: f32, edge2: f32, x: f32) -> f32 {
    return step(edge1, x)*(1.0-step(edge2, x));
}

fn triangle(x: f32) -> f32 {
    return abs(-1.0+(sign(x)*x%2.0));
}