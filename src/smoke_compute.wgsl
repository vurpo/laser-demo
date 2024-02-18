@group(0) @binding(0) var input_texture : texture_3d<f32>;
@group(0) @binding(1) var input_poisson : texture_3d<f32>;
@group(0) @binding(2) var output_texture : texture_storage_3d<rgba32float, write>;
@group(0) @binding(3) var output_poisson : texture_storage_3d<r32float, write>;

struct ShaderParams {
    step: i32,
    delta_time: f32
}

@group(1) @binding(0)
var<uniform> shader_params: ShaderParams;

const DIFFUSION: f32 = 6.0;
const SCALE: f32 = 1.0;

fn trilinear_sample(coords: vec3<f32>) -> vec4<f32> {
    let c0 = vec3<i32>(floor(coords));
    let c1 = vec3<i32>(floor(coords))+vec3<i32>(1,1,1);
    let cd: vec3<f32> = fract(coords);

    let s00 = mix(load(vec3<i32>(c0.x, c0.y, c0.z)), load(vec3<i32>(c1.x, c0.y, c0.z)), cd.x);
    let s01 = mix(load(vec3<i32>(c0.x, c0.y, c1.z)), load(vec3<i32>(c1.x, c0.y, c1.z)), cd.x);
    let s10 = mix(load(vec3<i32>(c0.x, c1.y, c0.z)), load(vec3<i32>(c1.x, c1.y, c0.z)), cd.x);
    let s11 = mix(load(vec3<i32>(c0.x, c1.y, c1.z)), load(vec3<i32>(c1.x, c1.y, c1.z)), cd.x);

    let s0 = mix(s00, s10, cd.y);
    let s1 = mix(s01, s11, cd.y);

    return mix(s0, s1, cd.z);
}

fn divergence(coords: vec3<i32>) -> f32 {
    return (
             load(coords+vec3( 1, 0, 0)).x
            -load(coords+vec3(-1, 0, 0)).x
            +load(coords+vec3( 0, 1, 0)).y
            -load(coords+vec3( 0,-1, 0)).y
            +load(coords+vec3( 0, 0, 1)).z
            -load(coords+vec3( 0, 0,-1)).z
        ) / 3.0;
}

fn poisson(coords: vec3<i32>) -> f32 {
    return (
        load_poisson(coords+vec3( 1, 0, 0)) +
        load_poisson(coords+vec3(-1, 0, 0)) +
        load_poisson(coords+vec3( 0, 1, 0)) +
        load_poisson(coords+vec3( 0,-1, 0)) +
        load_poisson(coords+vec3( 0, 0, 1)) +
        load_poisson(coords+vec3( 0, 0,-1)) -
        divergence(coords)) / 6.0;
}

fn diffuse_velocity(coords: vec3<i32>) -> vec3<f32> {
    let current: vec4<f32> = load(coords);
    let k: f32 = shader_params.delta_time*DIFFUSION;
    let k0: f32 = 1.0-(shader_params.delta_time*0.1);
    let out: vec4<f32> = (current*k0
        +(k*(load(coords+vec3( 1, 0, 0))
            +load(coords+vec3(-1, 0, 0))
            +load(coords+vec3( 0, 1, 0))
            +load(coords+vec3( 0,-1, 0))
            +load(coords+vec3( 0, 0, 1))
            +load(coords+vec3( 0, 0,-1))))
        /6.0)
    /(1.0+k);
    return out.rgb;
}

fn diffuse_density(coords: vec3<i32>) -> f32 {
    let current = load(coords);
    let k: f32 = shader_params.delta_time*DIFFUSION;
    let k0: f32 = 1.0-(shader_params.delta_time*0.1);
    return (current.a*k0
        +(k*(load(coords+vec3( 1, 0, 0)).w
            +load(coords+vec3(-1, 0, 0)).w
            +load(coords+vec3( 0, 1, 0)).w
            +load(coords+vec3( 0,-1, 0)).w
            +load(coords+vec3( 0, 0, 1)).w
            +load(coords+vec3( 0, 0,-1)).w))
        /6.0)
    /(1.0+k);
}

fn project_velocity(coords: vec3<i32>) -> vec3<f32> {
    let current: vec4<f32> = load(coords);
    let out: vec4<f32> = current - vec4(
        (load_poisson(coords+vec3( 1, 0, 0))-load_poisson(coords+vec3(-1, 0, 0)))*0.5,
        (load_poisson(coords+vec3( 0, 1, 0))-load_poisson(coords+vec3( 0,-1, 0)))*0.5,
        (load_poisson(coords+vec3( 0, 0, 1))-load_poisson(coords+vec3( 0, 0,-1)))*0.5,
        0.0
    );
    return out.rgb;
}

fn advect_velocity(coords: vec3<i32>) -> vec3<f32> {
    let current: vec4<f32> = load(coords);
    let old_pos = vec3<f32>(coords) - current.xyz*shader_params.delta_time;
    let out: vec4<f32> = trilinear_sample(old_pos);
    return out.rgb;
}

fn advect_density(coords: vec3<i32>) -> f32 {
    let current: vec4<f32> = load(coords);
    let old_pos = vec3<f32>(coords) - current.xyz*shader_params.delta_time;
    let out = trilinear_sample(old_pos);
    return out.a;
}

fn step2(edge1: f32, edge2: f32, x: f32) -> f32 {
    return step(edge1, x)*(1.0-step(edge2, x));
}

fn load(coords: vec3<i32>) -> vec4<f32> {
    let dimensions: vec3<i32> = vec3<i32>(textureDimensions(input_texture));
    let border: f32 = 
         step2(1.0, f32(dimensions.x)-(1.0), f32(coords.x))
        *step2(1.0, f32(dimensions.y)-(1.0), f32(coords.y))
        *step2(1.0, f32(dimensions.z)-(1.0), f32(coords.z));
    return border*textureLoad(input_texture, coords.xyz, 0);
}

fn load_poisson(coords: vec3<i32>) -> f32 {
    let dimensions: vec3<i32> = vec3<i32>(textureDimensions(input_texture));
    let border: f32 = 
         step2(1.0, f32(dimensions.x)-(1.0), f32(coords.x))
        *step2(1.0, f32(dimensions.y)-(1.0), f32(coords.y))
        *step2(1.0, f32(dimensions.z)-(1.0), f32(coords.z));
    return border*textureLoad(input_poisson, coords.xyz, 0).r;
}

@compute @workgroup_size(8, 8, 4)
fn fluid_main(
  @builtin(global_invocation_id) global_id : vec3<u32>,
) {
    let dimensions: vec3<i32> = vec3<i32>(textureDimensions(input_texture));
    let coords: vec3<i32> = vec3<i32>(global_id.xyz);

    if(coords.x >= dimensions.x || coords.y >= dimensions.y || coords.z >= dimensions.z) {
        return;
    }

    let current: vec4<f32> = load(coords);
    let current_poisson: f32 = load_poisson(coords);
    switch shader_params.step {
        // add smoke and velocity
        case 0: {
            let point_1: vec3<f32> = vec3<f32>(f32(dimensions.x)/3.0, f32(dimensions.y)/2.0-(1.0), 1.0);
            let point_2: vec3<f32> = vec3<f32>(f32(dimensions.x)-f32(dimensions.x)/3.0, f32(dimensions.y)/2.0+(1.0), 1.0);
            let d: f32 = step(3.0, min(distance(vec3<f32>(coords), point_1), distance(vec3<f32>(coords), point_2)));
            let replace: vec4<f32> = vec4<f32>(-500.0*sign(f32(coords.x)-f32(dimensions.x)/2.0),0.0,700.0,1.0);
            textureStore(output_texture, coords, current*d+replace*(1.0-d));
            textureStore(output_poisson, coords, vec4(poisson(coords)));
        }
        // advect velocity and density
        case 1: {
            textureStore(output_texture, coords, vec4(diffuse_velocity(coords), diffuse_density(coords)));
            textureStore(output_poisson, coords, vec4(poisson(coords)));
        }
        case 2: {
            textureStore(output_texture, coords, vec4(diffuse_velocity(coords), diffuse_density(coords)));
            textureStore(output_poisson, coords, vec4(poisson(coords)));
        }
        case 3: {
            textureStore(output_texture, coords, vec4(project_velocity(coords), current.w));
            textureStore(output_poisson, coords, vec4(poisson(coords)));
        }
        case 4: {
            textureStore(output_texture, coords, vec4(advect_velocity(coords), advect_density(coords)));
            textureStore(output_poisson, coords, vec4(poisson(coords)));
        }
        case 5: {
            textureStore(output_texture, coords, vec4(project_velocity(coords), current.w));
            textureStore(output_poisson, coords, vec4(poisson(coords)));
        }
        // case 6: {
        //     textureStore(output_texture, coords, current);
        //     textureStore(output_poisson, coords, vec4(poisson(coords)));
        // }
        default: {}
    }
}