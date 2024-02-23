const pi = 3.14159265359;

@group(0) @binding(0) var input_texture : texture_3d<f32>;
@group(0) @binding(1) var input_poisson : texture_3d<f32>;
@group(0) @binding(2) var output_texture : texture_storage_3d<rgba32float, write>;
@group(0) @binding(3) var output_poisson : texture_storage_3d<r32float, write>;
@group(0) @binding(4) var output_packed : texture_storage_3d<rgba32uint, write>;

struct ShaderParams {
    step: i32,
    delta_time: f32,
    time: f32
}

@group(1) @binding(0)
var<uniform> shader_params: ShaderParams;

const DIFFUSION: f32 = 5.0;
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

fn divergence(loaded: array<vec4<f32>,7>) -> f32 {
    return (
             loaded[1].x
            -loaded[2].x
            +loaded[3].y
            -loaded[4].y
            +loaded[5].z
            -loaded[6].z
        ) / 3.0;
}

fn poisson(loaded: array<vec4<f32>,7>, loaded_poisson: array<f32,7>) -> f32 {
    return (
        loaded_poisson[1] +
        loaded_poisson[2] +
        loaded_poisson[3] +
        loaded_poisson[4] +
        loaded_poisson[5] +
        loaded_poisson[6] -
        divergence(loaded)) / 6.0;
}

fn diffuse(loaded: array<vec4<f32>,7>) -> vec4<f32> {
    let k: f32 = shader_params.delta_time*DIFFUSION;
    let k0: f32 = 1.0-(shader_params.delta_time*0.1);
    return (loaded[0]*k0
        +(k*(loaded[1]
            +loaded[2]
            +loaded[3]
            +loaded[4]
            +loaded[5]
            +loaded[6]))
        /6.0*SCALE)
    /(1.0+k);
}

fn project_velocity(loaded: array<vec4<f32>,7>, loaded_poisson: array<f32,7>) -> vec3<f32> {
    return loaded[0].xyz - vec3(
        (loaded_poisson[1]-loaded_poisson[2])*0.5,
        (loaded_poisson[3]-loaded_poisson[4])*0.5,
        (loaded_poisson[5]-loaded_poisson[6])*0.5
    );
}

fn advect(coords: vec3<i32>) -> vec4<f32> {
    let current: vec4<f32> = load(coords);
    let old_pos = vec3<f32>(coords) - current.xyz*shader_params.delta_time*SCALE;
    let out = trilinear_sample(old_pos);
    return out;
}

fn step2(edge1: f32, edge2: f32, x: f32) -> f32 {
    return step(edge1, x)*(1.0-step(edge2, x));
}

fn load(coords: vec3<i32>) -> vec4<f32> {
    let dimensions: vec3<i32> = vec3<i32>(textureDimensions(input_texture));
    // let border2: f32 = step(f32(dimensions.z)-(1.0), f32(coords.z));

    let border: f32 = 
         step2(1.0, f32(dimensions.x)-(1.0), f32(coords.x))
        *step2(1.0, f32(dimensions.y)-(1.0), f32(coords.y))
        *step2(1.0, f32(dimensions.z)-(1.0), f32(coords.z));
    let current = border*textureLoad(input_texture, coords.xyz, 0);
    return vec4(current.xy, max(current.z, 5.0), current.w);//(1.0-border2)*current+border2*vec4(current.xy, -abs(current.z), current.w);
}

fn load_poisson(coords: vec3<i32>) -> f32 {
    let dimensions: vec3<i32> = vec3<i32>(textureDimensions(input_texture));
    let border: f32 = 
         step2(1.0, f32(dimensions.x)-(1.0), f32(coords.x))
        *step2(1.0, f32(dimensions.y)-(1.0), f32(coords.y))
        *step2(1.0, f32(dimensions.z)-(1.0), f32(coords.z));
    return border*textureLoad(input_poisson, coords.xyz, 0).r;
}

fn rotation(theta: f32) -> mat3x3f {
    return mat3x3f(cos(theta), sin(theta), 0., -sin(theta), cos(theta), 0., 0., 0., 1.);
}

@compute @workgroup_size(8,8,4)
fn fluid_main(
  @builtin(global_invocation_id) global_id : vec3<u32>,
) {
    let dimensions: vec3<i32> = vec3<i32>(textureDimensions(input_texture));
    let coords: vec3<i32> = vec3<i32>(global_id.xyz);

    if(coords.x >= dimensions.x || coords.y >= dimensions.y || coords.z >= dimensions.z) {
        return;
    }

    let loaded: array<vec4<f32>,7> = array(
        load(coords),
        load(coords+vec3( 1, 0, 0)),
        load(coords+vec3(-1, 0, 0)),
        load(coords+vec3( 0, 1, 0)),
        load(coords+vec3( 0,-1, 0)),
        load(coords+vec3( 0, 0, 1)),
        load(coords+vec3( 0, 0,-1))
    );
    let loaded_poisson: array<f32,7> = array(
        load_poisson(coords),
        load_poisson(coords+vec3( 1, 0, 0)),
        load_poisson(coords+vec3(-1, 0, 0)),
        load_poisson(coords+vec3( 0, 1, 0)),
        load_poisson(coords+vec3( 0,-1, 0)),
        load_poisson(coords+vec3( 0, 0, 1)),
        load_poisson(coords+vec3( 0, 0,-1))
    );
    switch shader_params.step {
        // add smoke and velocity
        case 0: {
            let center: vec3<f32> = vec3(f32(dimensions.x)/2.0, f32(dimensions.y)/2.0, 1.0);
            let point_1: vec3<f32> = center+(rotation(           2.)*vec3(30.0, 0.0, 0.0));
            let point_2: vec3<f32> = center+(rotation(2.0/3.0*pi+2.)*vec3(30.0, 0.0, 0.0));
            let point_3: vec3<f32> = center+(rotation(4.0/3.0*pi+2.)*vec3(30.0, 0.0, 0.0));
            let d: f32 = step(3.0, min(min(distance(vec3<f32>(coords), point_1), distance(vec3<f32>(coords), point_2)), distance(vec3<f32>(coords), point_3)));
            let dir: vec3<f32> = 
                ((center-vec3<f32>(coords))*10.0+vec3(0.0,0.0,150.0))
                +10.0*sin(10.*shader_params.time);
            let replace: vec4<f32> = vec4(dir,1.5);
            textureStore(output_texture, coords, loaded[0]*d+replace*(1.0-d));
            textureStore(output_poisson, coords, vec4(poisson(loaded, loaded_poisson)));
        }
        case 1,2: {
            textureStore(output_texture, coords, diffuse(loaded));
            textureStore(output_poisson, coords, vec4(poisson(loaded, loaded_poisson)));
        }
        case 3: {
            textureStore(output_texture, coords, vec4(project_velocity(loaded, loaded_poisson), loaded[0].w));
            textureStore(output_poisson, coords, vec4(poisson(loaded, loaded_poisson)));
        }
        case 4: {
            textureStore(output_texture, coords, advect(coords));
            textureStore(output_poisson, coords, vec4(poisson(loaded, loaded_poisson)));
        }
        case 5: {
            textureStore(output_texture, coords, vec4(project_velocity(loaded, loaded_poisson), loaded[0].w));
            textureStore(output_packed, coords, vec4(
                pack2x16float(vec2(load(coords+vec3(0,0,0)).w, load(coords+vec3(1,0,0)).w)),
                pack2x16float(vec2(load(coords+vec3(0,0,1)).w, load(coords+vec3(1,0,1)).w)),
                pack2x16float(vec2(load(coords+vec3(0,1,0)).w, load(coords+vec3(1,1,0)).w)),
                pack2x16float(vec2(load(coords+vec3(0,1,1)).w, load(coords+vec3(1,1,1)).w)),
            ));
            textureStore(output_poisson, coords, vec4(poisson(loaded, loaded_poisson)));
        }

        case 6: {
            // var letter: array<f32, 15> = array(
            //     1., 1., 1.,
            //     1., 0., 0.,
            //     1., 1., 0.,
            //     1., 0., 0.,
            //     1., 1., 1.
            // );
            // let c = vec3<i32>((vec3<f32>(coords)-vec3(75.,75.,20.))*0.1);
            // let p: f32 = step2(0.0,2.5,f32(c.x))*step2(0.0,0.5,f32(c.y))*step2(0.0,4.5,f32(c.z))*0.1*letter[clamp(3*c.z+c.x, 0, 5*3)];
            // textureStore(output_texture, coords, vec4(loaded[0].xyz, loaded[0].w+p));
            // textureStore(output_poisson, coords, vec4(poisson(loaded, loaded_poisson)));
        }
        case 7: {
            // textureStore(output_texture, coords, loaded[0]);
            // textureStore(output_poisson, coords, vec4(poisson(loaded, loaded_poisson)));
        }
        default: {}
    }
}
