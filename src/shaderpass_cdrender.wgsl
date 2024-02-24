const pi = 3.14159265359;

// Vertex shader

struct Camera {
    view_proj: mat4x4<f32>,
}
@group(0) @binding(0)
var<uniform> camera: Camera;

struct ShaderParams {
    shader_function: i32,
    time: f32,
    x: f32
}

@group(0) @binding(1)
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
    @location(0) tex_coords: vec2<f32>,
    @location(1) dot: f32
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
    out.dot = instance.dot;
    return out;
}

fn rainbow(x: f32) -> vec3<f32> {
    let c: vec3<f32> = vec3(x, .5, 1.);
    let K: vec4<f32> = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    let p: vec3<f32> = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3(0.0), vec3(1.0)), c.y);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let outer_edge: f32 = 1.0-step(1.0, length(in.tex_coords));
    let edge_1: f32 = step(0.25, length(in.tex_coords));
    let inner_edge: f32 = step(0.1, length(in.tex_coords));
    if inner_edge*outer_edge < 0.5 { discard; }
    
    let angle: f32 = atan2(in.tex_coords.y, in.tex_coords.x)+shader_params.time;
    
    let c: f32 = pow((sin(4.*angle)+1.)/2.,(sin(in.dot*2.+angle*2.)+1.)*3.);
    let color: vec3<f32> = rainbow((0.6-c)+sin(angle)+in.dot*2.);
    
    let out: vec3<f32> = clamp(in.clip_position.w*10.,0.,1.)*outer_edge*edge_1*(c*color+(1.-c)*vec3(0.12,0.11,0.1))
        +(1.-edge_1)*inner_edge*(vec3(0.4+0.3*c));
    return vec4(out, 1.0);
}
