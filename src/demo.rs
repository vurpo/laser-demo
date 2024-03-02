use std::{f32::consts::PI, sync::{Arc, Mutex}};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use web_sys::HtmlInputElement;
use web_time::{Instant, Duration};

use cgmath::{Deg, Matrix4, Quaternion, Rotation3, SquareMatrix, Vector3, Zero};
use cpal::{traits::{DeviceTrait, HostTrait, StreamTrait}, BufferSize};
use rand::{Rng, SeedableRng};
use wgpu::{
    util::DeviceExt, BindGroup, Buffer, CommandEncoder, ComputePipeline, Extent3d, RenderPipeline, TextureFormat, TextureView
};
use xmrs::xm::xmmodule::XmModule;
use xmrsplayer::xmrsplayer::XmrsPlayer;

use crate::{
    model::{Model,Vertex,DrawModel},
    resources::{self, ASSETS, QUAD_INDICES, QUAD_VERTICES},
    texture::{self, Texture}, Instance, FLUID_SIZE, OPENGL_TO_WGPU_MATRIX
};

const COMPUTE_PASSES: i32 = 6;
const COMPUTE_EXTRAS: i32 = 2;

const NUM_CDS: usize = 300;
const NUM_STARWARS: usize = 100;

// This file is where the fun happens! It's also the worst spaghetti ever devised.
// 
// 2024 update: it's even worse now

fn compute_work_group_count(
    (width, height, depth): (u32, u32, u32),
    (workgroup_width, workgroup_height, workgroup_depth): (u32, u32, u32),
) -> (u32, u32, u32) {
    let x = (width + workgroup_width - 1) / workgroup_width;
    let y = (height + workgroup_height - 1) / workgroup_height;
    let z = (depth + workgroup_depth - 1) / workgroup_depth;

    (x, y, z)
}

const STEPS: [((usize,usize), Scene, Transition); 33] = [
    ((0x00,0x00), Scene::Black,        Transition::None),
    ((0x00,0x18), Scene::Slide(0),     Transition::None),
    ((0x01,0x16), Scene::Black,        Transition::None),
    ((0x01,0x1b), Scene::Slide(1),     Transition::None),
    ((0x01,0x3b), Scene::Slide(2),     Transition::Slide),
    ((0x02,0x0f), Scene::Slide(3),     Transition::Slide),
    ((0x02,0x1b), Scene::Slide(4),     Transition::Fade(1.)),
    ((0x02,0x34), Scene::Slide(5),     Transition::Fade(1.)),
    ((0x03,0x0f), Scene::Slide(6),     Transition::Fade(1.)),
    ((0x03,0x23), Scene::Slide(7),     Transition::Fade(1.)),
    ((0x03,0x2d), Scene::CDs(8),       Transition::Blink),
    ((0x04,0x30), Scene::CDs(9),       Transition::None),
    ((0x05,0x10), Scene::CDs(10),       Transition::None),
    ((0x07,0x00), Scene::StarWars(11), Transition::Slide),
    ((0x07,0x30), Scene::StarWars(12), Transition::None),
    ((0x08,0x10), Scene::StarWars(13), Transition::None),
    ((0x0a,0x00), Scene::Ocean(14),    Transition::Slide),
    ((0x0b,0x00), Scene::Ocean(15),    Transition::None),
    ((0x0c,0x00), Scene::Ocean(16),    Transition::None),
    ((0x0e,0x00), Scene::Slide(17),    Transition::Slide),
    ((0x0e,0x20), Scene::Black,        Transition::Fade(0.2)),
    ((0x0f,0x00), Scene::Slide(18),    Transition::Fade(0.3)),
    ((0x0f,0x10), Scene::Slide(19),    Transition::Fade(1.)),
    ((0x0f,0x20), Scene::Slide(20),    Transition::Fade(1.)),
    ((0x10,0x00), Scene::Black,        Transition::Fade(0.5)),
    ((0x10,0x08), Scene::Smoke(1),     Transition::None),
    ((0x11,0x00), Scene::Smoke(2),     Transition::None),
    ((0x12,0x00), Scene::Smoke(3),     Transition::None),
    ((0x13,0x00), Scene::Smoke(4),     Transition::None),
    ((0x18,0x00), Scene::Slide(20),    Transition::Fade(0.5)),
    ((0x18,0x08), Scene::Slide(21),    Transition::Slide),
    ((0x19,0x08), Scene::Slide(22),    Transition::Fade(1.)),
    ((0x19,0x3d), Scene::Slide(23),    Transition::Blink2),
];
const START_FROM: usize = 0;

pub struct Demo {
    current_step: i32,
    scene: Scene,
    transition: Transition,
    transitioned_at: Instant,
    full_quad_vertex_buffer: Buffer,
    full_quad_index_buffer: Buffer,
    pub instances: Vec<Instance>,
    pub cd_instances: Vec<Instance>,
    pub starwars_instances: Vec<Instance>,
    pub instance_buffer: Buffer,
    pub cd_instance_buffer: Buffer,
    pub starwars_instance_buffer: Buffer,
    camera_buffer: wgpu::Buffer,
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
    pub uniform_bind_group_layout: wgpu::BindGroupLayout,
    bg_function_buffer: wgpu::Buffer,
    bg_uniform_bind_group: wgpu::BindGroup,
    object_uniform_bind_group: wgpu::BindGroup,
    final_function_buffer: wgpu::Buffer,
    final_function_bindgroup: BindGroup,
    start_time: Instant,
    last_time: Instant,
    last_row: usize,
    beat: Instant,
    // last_pattern: usize,
    // pattern: Instant,
    next_step: (usize, usize),
    rng: rand::rngs::SmallRng,
    player: Arc<Mutex<XmrsPlayer>>,
    bg_shader_params: ShaderParamsUniform,
    final_shader_params: ShaderParamsUniform,
    pub camera: Camera,
    camera_uniform: CameraUniform,
    
    lasers_uniform: LasersUniform,
    lasers_uniform_buffer: Buffer,
    lasers_uniform_bindgroup: BindGroup,
    
    
    slide_textures: Vec<Texture>,
    slide_texture_bindgroups: Vec<BindGroup>,
    ocean_texture_bindgroup: BindGroup,
    
    pewpew_model: Model,
    
    render_pipeline_smokerender: RenderPipeline,
    render_pipeline_cdrender: RenderPipeline,
    render_pipeline_starwars1: RenderPipeline,
    render_pipeline_starwars2: RenderPipeline,
    render_pipeline_ocean: RenderPipeline,
    render_pipeline_simple: RenderPipeline,
    
    texture_pass1: texture::Texture,
    texture_pass1_bindgroup: wgpu::BindGroup,
    texture_pass2: texture::Texture,
    texture_pass2_bindgroup: BindGroup,
    texture_pass_window: texture::Texture,
    texture_pass_window_bindgroup: BindGroup,
    previous_pass_texture: texture::Texture,
    previous_pass_texture_bind_group: BindGroup,

    pub smoke_render_bind_group_layout: wgpu::BindGroupLayout,
    smoke_render_bind_group: wgpu::BindGroup,

    compute_pipeline: ComputePipeline,
    pub smoke_texture_bind_group_layout: wgpu::BindGroupLayout,
    smoke_compute_bindgroup1: BindGroup,
    smoke_compute_bindgroup2: BindGroup,
    smoke_shader_params: Vec<ComputeParamsUniform>,
    smoke_shader_params_buffer: Vec<wgpu::Buffer>,
    smoke_shader_params_bindgroup: Vec<BindGroup>,
    current_size: usize,

    frame_log: (Instant, i32),
}

impl Demo {
    pub async fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: TextureFormat
    ) -> Self {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0x4375746552616363);
                
        let instances = vec![
            Instance {
                position: cgmath::Vector3::new(0.0, 0.0, 0.0),
                rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Rad(0.0)),
                scale: cgmath::Vector3::new(1.0,1.0,1.0),
                tex_offset: cgmath::Vector2::new(0.0, 0.0)
            },
            Instance {
                position: cgmath::Vector3::new(-0.37, 0.0, 0.0),
                rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Rad(0.0)),
                scale: cgmath::Vector3::new(0.55,0.55,1.0),
                tex_offset: cgmath::Vector2::new(0.0, 0.0)
            },
            Instance {
                position: cgmath::Vector3::new(0.37, 0.0, 0.0),
                rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Rad(0.0)),
                scale: cgmath::Vector3::new(0.55,0.55,1.0),
                tex_offset: cgmath::Vector2::new(0.0, 0.0)
            },
        ];
        let mut cd_instances = vec![];
        for i in 0..NUM_CDS+1 {
            cd_instances.push(Instance {
                position: cgmath::Vector3::new(rng.gen_range(-30.0..30.0), rng.gen_range(-16.0..16.0), rng.gen_range(-25.0..0.0)),
                rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(i as f32*3.0)),
                scale: cgmath::Vector3::new(1.0,1.0,1.0),
                tex_offset: cgmath::Vector2::new(0.0, 0.0)
            });
        }
        let mut starwars_instances = vec![];
        for i in 0..NUM_STARWARS*2 {
            let side = i as i32%2*2-1;
            starwars_instances.push(Instance {
                position: cgmath::Vector3::new(
                    side as f32*40.0+rng.gen_range(-20.0..20.0),
                    rng.gen_range(-10.0..10.0),
                    rng.gen_range(-10.0..0.0)
                ),
                rotation: cgmath::Quaternion::zero(),
                scale: cgmath::Vector3::new(1.0,1.0,1.0),
                tex_offset: cgmath::Vector2::new(0.0,0.0)
            });
        }

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        
        let instance_data = cd_instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let cd_instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CD instances Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        
        let instance_data = starwars_instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let starwars_instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Starwars instances Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let camera = Camera {
            eye: (0.0, 0.0, 10.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: 16.0 / 9.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let full_quad_vertex_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fullscreen quad vertex buffer"),
                contents: bytemuck::cast_slice(&QUAD_VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let full_quad_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fullscreen quad index buffer"),
            contents: bytemuck::cast_slice(&QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let bg_shader_params = ShaderParamsUniform {
            shader_function: 0,
            t: 0.0,
            x: 0.0,
            transition: 0.0,
            x2: 0.0
        };
        
        let final_shader_params = ShaderParamsUniform {
            shader_function: 0,
            t: 0.0,
            x: 0.0,
            transition: 0.0,
            x2: 0.005,
        };

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let none_camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("No Camera Buffer"),
            contents: bytemuck::cast_slice(&[CameraUniform {
                view_proj: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let bg_function_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Background params buffer"),
            contents: bytemuck::cast_slice(&[bg_shader_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let final_function_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Final pass params buffer"),
            contents: bytemuck::cast_slice(&[final_shader_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("uniform_bind_group_layout"),
            });


        let bg_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: none_camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bg_function_buffer.as_entire_binding(),
                },
            ],
            label: Some("uniform_bind_group_1"),
        });

        let object_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &uniform_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: camera_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: bg_function_buffer.as_entire_binding(),
                        },
                    ],
                    label: Some("uniform_bind_group_2"),
                });
        let final_function_bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &uniform_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: none_camera_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: final_function_buffer.as_entire_binding(),
                        },
                    ],
                    label: Some("uniform_bind_group_2"),
                });
        
        let previous_pass_texture = Texture::from_texture(&device, device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: 1920,
                height: 1080,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            label: None,
            view_formats: &vec![]
        }), wgpu::FilterMode::Linear);
        let previous_pass_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&previous_pass_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&previous_pass_texture.sampler),
                },
            ],
            label: None,
        });

        // Smoke simulation compute stuff:

        let smoke_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D3,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D3,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D3,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D3,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba32Uint,
                            view_dimension: wgpu::TextureViewDimension::D3,
                        },
                        count: None,
                    },
                ],
                label: Some("smoke_texture_bind_group_layout"),
            });
        let smoke_shader_params_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("uniform_bind_group_layout"),
            });

        let shader_smoke_compute = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Smoke compute shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("smoke_compute.wgsl").into()),
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute pipeline layout"),
                bind_group_layouts: &[
                    &smoke_texture_bind_group_layout,
                    &smoke_shader_params_layout,
                ],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Smoke compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader_smoke_compute,
            entry_point: "fluid_main",
        });

        let smoke_texture1 = texture::Texture::from_texture(
            device,
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("smoke texture 1"),
                size: wgpu::Extent3d {
                    width: FLUID_SIZE.0 as u32,
                    height: FLUID_SIZE.0 as u32,
                    depth_or_array_layers: FLUID_SIZE.0 as u32,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &vec![],
            }),
            wgpu::FilterMode::Linear,
        );

        let smoke_texture2 = texture::Texture::from_texture(
            device,
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("smoke texture 2"),
                size: wgpu::Extent3d {
                    width: FLUID_SIZE.0 as u32,
                    height: FLUID_SIZE.0 as u32,
                    depth_or_array_layers: FLUID_SIZE.0 as u32,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &vec![],
            }),
            wgpu::FilterMode::Linear,
        );

        let packed_smoke_texture = texture::Texture::from_texture(
            device,
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("packed smoke texture"),
                size: wgpu::Extent3d {
                    width: FLUID_SIZE.0 as u32,
                    height: FLUID_SIZE.0 as u32,
                    depth_or_array_layers: FLUID_SIZE.0 as u32,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba32Uint,
                usage: wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &vec![],
            }),
            wgpu::FilterMode::Linear,
        );

        let poisson_texture1 = texture::Texture::from_texture(
            device,
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("poisson texture 1"),
                size: wgpu::Extent3d {
                    width: FLUID_SIZE.0 as u32,
                    height: FLUID_SIZE.0 as u32,
                    depth_or_array_layers: FLUID_SIZE.0 as u32,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &vec![],
            }),
            wgpu::FilterMode::Linear,
        );

        let poisson_texture2 = texture::Texture::from_texture(
            device,
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("poisson texture 2"),
                size: wgpu::Extent3d {
                    width: FLUID_SIZE.0 as u32,
                    height: FLUID_SIZE.0 as u32,
                    depth_or_array_layers: FLUID_SIZE.0 as u32,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &vec![],
            }),
            wgpu::FilterMode::Linear,
        );

        let smoke_compute_bindgroup1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Smoke compute bind group 1"),
            layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&smoke_texture1.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&poisson_texture1.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&smoke_texture2.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&poisson_texture2.view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&packed_smoke_texture.view),
                },
            ],
        });
        let smoke_compute_bindgroup2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Smoke compute bind group 2"),
            layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&smoke_texture2.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&poisson_texture2.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&smoke_texture1.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&poisson_texture1.view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&packed_smoke_texture.view),
                },
            ],
        });

        let smoke_shader_params: Vec<ComputeParamsUniform> = (0..COMPUTE_PASSES + COMPUTE_EXTRAS)
            .map(|i| ComputeParamsUniform {
                step: i,
                delta_time: 0.0,
                time: 0.0,
                x: 0.0,
            })
            .collect();
        let smoke_shader_params_buffer: Vec<wgpu::Buffer> = smoke_shader_params
            .iter()
            .map(|p| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&"Smoke shader params buffer"),
                    contents: bytemuck::cast_slice(&[*p]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
            })
            .collect();
        let smoke_shader_params_bindgroup: Vec<BindGroup> = smoke_shader_params_buffer
            .iter()
            .map(|b| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Smoke shader params bind group"),
                    layout: &compute_pipeline.get_bind_group_layout(1),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: b.as_entire_binding(),
                    }],
                })
            })
            .collect();
        
        let slide_textures: Vec<Texture> = ASSETS.get_dir("slides").unwrap().entries().iter().filter_map(
            |entry| {
                let path = entry.path().to_str().unwrap();
                match entry {
                    include_dir::DirEntry::File(f) => {
                        log::info!("loading {}", path);
                        Some(Texture::from_bytes(device, queue, f.contents(), path, surface_format, (wgpu::AddressMode::ClampToEdge, wgpu::AddressMode::ClampToEdge)).unwrap())
                    }
                    include_dir::DirEntry::Dir(_) => None
                }
            }
        ).collect();
        let slide_texture_bindgroups = slide_textures.iter().map(|t| device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&t.view)
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&t.sampler)
                },
            ]
        })).collect();
        
        let shader_smokerender = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaderpass_smokerender.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaderpass_smokerender.wgsl").into()),
        });
        let shader_cdrender = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaderpass_cdrender.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaderpass_cdrender.wgsl").into()),
        });
        let shader_starwars1 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaderpass_starwars1.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaderpass_starwars1.wgsl").into()),
        });
        let shader_starwars2 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaderpass_starwars2.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaderpass_starwars2.wgsl").into()),
        });
        let shader_ocean = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaderpass_ocean.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaderpass_ocean.wgsl").into()),
        });
        let shader_simple = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaderpass_simple.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaderpass_simple.wgsl").into()),
        });
        
        let smoke_render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D3,
                    sample_type: wgpu::TextureSampleType::Uint,
                },
                count: None,
            }],
            label: Some("Smoke render bind group layout"),
        });
        
        let lasers_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
           entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
           }],
           label: Some("Lasers bind group layout")
        });
        
        let smoke_render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Smoke render bind group"),
            layout: &smoke_render_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&packed_smoke_texture.view),
            }],
        });
        
        let render_pipeline_layout_smokerender =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render layout smoke render"),
                bind_group_layouts: &[&smoke_render_bind_group_layout, &uniform_bind_group_layout, &lasers_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline_smokerender = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render pipeline smoke render"),
            layout: Some(&render_pipeline_layout_smokerender),
            vertex: wgpu::VertexState {
                module: &shader_smokerender,
                entry_point: "vs_main",
                buffers: &[crate::model::ModelVertex::desc(), crate::InstanceRaw::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_smokerender,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format.add_srgb_suffix(),
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent{
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,},
                        alpha: wgpu::BlendComponent::REPLACE
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        
        let render_pipeline_layout_cdrender =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render layout CD render"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline_cdrender = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render pipeline CD render"),
            layout: Some(&render_pipeline_layout_cdrender),
            vertex: wgpu::VertexState {
                module: &shader_cdrender,
                entry_point: "vs_main",
                buffers: &[crate::model::ModelVertex::desc(), crate::InstanceRaw::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_cdrender,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format.add_srgb_suffix(),
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent{
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,},
                        alpha: wgpu::BlendComponent::OVER
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        
        let render_pipeline_starwars1 = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render pipeline star wars pass 1"),
            layout: Some(&render_pipeline_layout_cdrender),
            vertex: wgpu::VertexState {
                module: &shader_starwars1,
                entry_point: "vs_main",
                buffers: &[crate::model::ModelVertex::desc(), crate::InstanceRaw::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_starwars1,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format.add_srgb_suffix(),
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent{
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,},
                        alpha: wgpu::BlendComponent::OVER
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        
        let render_pipeline_layout_simple =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render layout simple"),
                bind_group_layouts: &[&texture_bind_group_layout, &uniform_bind_group_layout, &texture_bind_group_layout],
                push_constant_ranges: &[],
            });
        
        let render_pipeline_starwars2 = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render pipeline star wars pass 2"),
            layout: Some(&render_pipeline_layout_simple),
            vertex: wgpu::VertexState {
                module: &shader_starwars2,
                entry_point: "vs_main",
                buffers: &[crate::model::ModelVertex::desc(), crate::InstanceRaw::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_starwars2,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format.add_srgb_suffix(),
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent{
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,},
                        alpha: wgpu::BlendComponent::OVER
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let render_pipeline_ocean = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render pipeline ocean"),
            layout: Some(&render_pipeline_layout_simple),
            vertex: wgpu::VertexState {
                module: &shader_ocean,
                entry_point: "vs_main",
                buffers: &[crate::model::ModelVertex::desc(), crate::InstanceRaw::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_ocean,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format.add_srgb_suffix(),
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent{
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::Zero,
                            operation: wgpu::BlendOperation::Add,},
                        alpha: wgpu::BlendComponent::REPLACE
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        
        let render_pipeline_simple = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render pipeline simple"),
            layout: Some(&render_pipeline_layout_simple),
            vertex: wgpu::VertexState {
                module: &shader_simple,
                entry_point: "vs_main",
                buffers: &[crate::model::ModelVertex::desc(), crate::InstanceRaw::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_simple,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format.add_srgb_suffix(),
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent{
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,},
                        alpha: wgpu::BlendComponent::OVER
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        
        let texture_pass1 = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: 1920,
                height: 1080,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format.add_srgb_suffix(),
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            label: None,
            view_formats: &vec![]
        });
        let texture_pass1 = Texture::from_texture(&device, texture_pass1, wgpu::FilterMode::Linear);
        
        let texture_pass2 = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: 1920,
                height: 1080,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format.add_srgb_suffix(),
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            label: None,
            view_formats: &vec![]
        });
        let texture_pass2 = Texture::from_texture(&device, texture_pass2, wgpu::FilterMode::Linear);
        
        let texture_pass_window = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: 1920,
                height: 1080,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format.add_srgb_suffix(),
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            label: None,
            view_formats: &vec![]
        });
        let texture_pass_window = Texture::from_texture(&device, texture_pass_window, wgpu::FilterMode::Linear);
        let texture_pass_window_bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_pass_window.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_pass_window.sampler),
                },
            ],
            label: None,
        });

        let texture_pass1_bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_pass1.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_pass1.sampler),
                },
            ],
            label: None,
        });
        
        let texture_pass2_bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_pass2.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_pass2.sampler),
                },
            ],
            label: None,
        });
        
        let pewpew_model = resources::load_model("pewpew.obj", device, 0.15).await.unwrap();
        
        let ocean_texture = Texture::from_bytes(
            device,
            queue,
            ASSETS.get_file("environment.jpg").unwrap().contents(),
            "Ocean texture",
            surface_format.add_srgb_suffix(),
            (wgpu::AddressMode::Repeat, wgpu::AddressMode::MirrorRepeat)).unwrap();
        let ocean_texture_bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&ocean_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&ocean_texture.sampler),
                },
            ],
            label: None,
        });
        
        let lasers_uniform = LasersUniform::new();
        let lasers_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&"Smoke lasers buffer"),
            contents: bytemuck::cast_slice(&[lasers_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let lasers_uniform_bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &lasers_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lasers_uniform_buffer.as_entire_binding(),
            }],
            label: Some("Lasers uniform bind group")
        });
        
        // The music:

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .expect("no output device available");
    
        let config = device.default_output_config().unwrap();
        
        let xm = XmModule::load(ASSETS.get_file("music.xm").unwrap().contents()).unwrap();
        let player = Arc::new(Mutex::new(XmrsPlayer::new(
            xm.to_module().into(),
            config.sample_rate().0 as f32,
        )));
        {
            let mut player_lock = player.lock().unwrap();
            player_lock.goto(STEPS[START_FROM].0.0, STEPS[START_FROM].0.1);
        }
        start_audio_player(player.clone()).expect("failed to start player");

        Demo {
            current_step: START_FROM as i32,
            next_step: STEPS[START_FROM+1].0,
            scene: STEPS[START_FROM].1,
            transition: STEPS[START_FROM].2,
            transitioned_at: Instant::now(),
            full_quad_vertex_buffer,
            full_quad_index_buffer,
            instances,
            cd_instances,
            starwars_instances,
            instance_buffer,
            cd_instance_buffer,
            starwars_instance_buffer,
            start_time: Instant::now(),
            last_time: Instant::now(),
            last_row: 0,
            beat: Instant::now(),
            // last_pattern: 0,
            // pattern: Instant::now(),
            rng,
            player,
            bg_shader_params,
            final_shader_params,
            texture_bind_group_layout,
            uniform_bind_group_layout,
            bg_uniform_bind_group,
            object_uniform_bind_group,
            camera,
            camera_uniform,
            camera_buffer,
            bg_function_buffer,
            final_function_buffer,
            final_function_bindgroup,
            texture_pass1,
            texture_pass1_bindgroup,
            texture_pass2,
            texture_pass2_bindgroup,
            texture_pass_window,
            texture_pass_window_bindgroup,
            previous_pass_texture,
            previous_pass_texture_bind_group,
            slide_textures,
            slide_texture_bindgroups,
            ocean_texture_bindgroup,
            
            lasers_uniform,
            lasers_uniform_buffer,
            lasers_uniform_bindgroup,

            smoke_render_bind_group_layout,
            smoke_render_bind_group,
            
            pewpew_model,
            
            render_pipeline_smokerender,
            render_pipeline_cdrender,
            render_pipeline_simple,
            render_pipeline_starwars1,
            render_pipeline_starwars2,
            render_pipeline_ocean,

            compute_pipeline,
            smoke_texture_bind_group_layout,
            smoke_compute_bindgroup1,
            smoke_compute_bindgroup2,
            smoke_shader_params,
            smoke_shader_params_buffer,
            smoke_shader_params_bindgroup,
            current_size: FLUID_SIZE.0,

            frame_log: (Instant::now(), 0),
        }
    }

    pub fn update(&mut self, queue: &wgpu::Queue, encoder: &mut CommandEncoder) {
        let now = Instant::now();
        let time = now.duration_since(self.start_time).as_secs_f64();
        let delta_time = now.duration_since(self.last_time).as_secs_f64();
        self.last_time = now;
        
        let (pattern, row) = {
            let player = self.player.lock().unwrap();
            let row = player.get_current_row();
            let pattern = player.get_current_table_index();
            (pattern, row)
        };
        
        self.step(encoder, pattern, row);
        let row_beats = match pattern {
            0x0a..=0x0d|0x18..=0x19 => 8,
            _ => 4,
        };
        if row % row_beats == 0 && row != self.last_row {
            self.beat = Instant::now();
        }
        self.last_row = row;
        // if pattern != self.last_pattern && pattern <= 9 {
        //     self.pattern = Instant::now();
        // }
        // self.last_pattern = pattern;

        match &self.scene {
            Scene::Slide(_) => {}
            Scene::Black => {
                queue.write_texture(
                    self.texture_pass1.texture.as_image_copy(),
                    vec![0; 4*1920*1080].as_slice(),
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(4*1920),
                        rows_per_image: Some(1080)
                    },
                    self.texture_pass1.texture.size());
            }
            Scene::CDs(_) => {
                for i in 0..NUM_CDS {
                    self.cd_instances[i].position.z += 10.*delta_time as f32;
                    if self.cd_instances[i].position.z >= 0. {
                        self.cd_instances[i].position.z = self.rng.gen_range(-40.0..-20.0);
                    }
                    self.cd_instances[i].rotation = 
                        Quaternion::from_angle_x(cgmath::Rad(i as f32+time as f32))
                        *Quaternion::from_angle_y(cgmath::Rad(i as f32-time as f32));
                }
                self.cd_instances[NUM_CDS].position = Vector3::new(0.0,0.0,7.5);
                self.cd_instances[NUM_CDS].rotation = 
                    Quaternion::from_angle_x(cgmath::Rad(1.*(time as f32).cos()))
                    *Quaternion::from_angle_y(cgmath::Rad(1.*(time as f32).sin()));
                    
                let instance_data = self.cd_instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
                queue.write_buffer(&self.cd_instance_buffer, 0, bytemuck::cast_slice(&instance_data));
            }
            Scene::StarWars(_) => {
                for i in 0..NUM_STARWARS*2 {
                    let side = i as i32%2*2-1;
                    self.starwars_instances[i].position.x += side as f32*-10.0*delta_time as f32;
                    if side as f32*self.starwars_instances[i].position.x < -20.0 {
                        self.starwars_instances[i].position = cgmath::Vector3::new(
                            side as f32*30.0+self.rng.gen_range(-10.0..10.0),
                            self.rng.gen_range(-10.0..10.0),
                            self.rng.gen_range(-10.0..0.0)
                        );
                    }
                }
                let instance_data = self.starwars_instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
                queue.write_buffer(&self.starwars_instance_buffer, 0, bytemuck::cast_slice(&instance_data));
                
                self.camera.eye = (-6.0, -6.0, 10.0).into();
                self.camera.target = (-5.0, -1.0, 0.0).into();
                
            }
            Scene::Ocean(_) => {}
            Scene::Smoke(number) => {
                let t = time as f32*7.;
                for (i, params) in self.smoke_shader_params.iter_mut().enumerate() {
                    params.delta_time = delta_time as f32;
                    params.time = time as f32;
                    params.x = match (row/16)%2 {
                        0 => 0.0,
                        1 => 1.0,
                        _ => panic!()
                    };
                    queue.write_buffer(
                        &self.smoke_shader_params_buffer[i],
                        0,
                        bytemuck::cast_slice(&[*params]),
                    );
                }
                
                let angle = PI*0.75+0.2*t.sin();
                let axis = Vector3::new(angle.cos(),angle.sin(),0.0);
                self.lasers_uniform.laser1_transform = 
                    (//Matrix4::from_translation(Vector3::new(0.0,0.0,0.0))
                    Matrix4::from(Quaternion::from_axis_angle(axis, Deg(50.+10.*t.cos()))))
                    .invert().unwrap().into();
                self.lasers_uniform.laser1_color = [1.0,0.3,0.3,0.0];
                
                if *number > 1 {
                    let angle = PI*0.25+0.2*t.sin();
                    let axis = Vector3::new(angle.cos(),angle.sin(),0.0);
                    self.lasers_uniform.laser2_transform = 
                        (Matrix4::from_translation(Vector3::new(0.0,100.0,0.0))
                        *Matrix4::from(Quaternion::from_axis_angle(axis, Deg(50.+10.*t.sin()))))
                        .invert().unwrap().into();
                    self.lasers_uniform.laser2_color = [0.3,1.0,0.3,0.0];
                }
                
                if *number > 2 {
                    let angle = PI*1.25+0.2*t.sin();
                    let axis = Vector3::new(angle.cos(),angle.sin(),0.0);
                    self.lasers_uniform.laser3_transform = 
                        (Matrix4::from_translation(Vector3::new(100.0,0.0,0.0))
                        *Matrix4::from(Quaternion::from_axis_angle(axis, Deg(50.-10.*t.sin()))))
                        .invert().unwrap().into();
                    self.lasers_uniform.laser3_color = [0.3,0.3,1.0,0.0];
                }
                
                if *number > 3 {
                    let angle = PI*1.75+0.2*t.sin();
                    let axis = Vector3::new(angle.cos(),angle.sin(),0.0);
                    self.lasers_uniform.laser4_transform = 
                        (Matrix4::from_translation(Vector3::new(100.0,100.0,0.0))
                        *Matrix4::from(Quaternion::from_axis_angle(axis, Deg(50.-10.*t.cos()))))
                        .invert().unwrap().into();
                    self.lasers_uniform.laser4_color = [1.0,1.0,0.3,0.0];
                }
                
                queue.write_buffer(&self.lasers_uniform_buffer, 0, bytemuck::cast_slice(&[self.lasers_uniform]));
                {
                    let (dispatch_width, dispatch_height, dispatch_depth) = compute_work_group_count(
                        (
                            self.current_size as u32,
                            self.current_size as u32,
                            self.current_size as u32,
                        ),
                        (8, 8, 4),
                    );
                    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Smoke pass"),
                        ..Default::default()
                    });
                    compute_pass.set_pipeline(&self.compute_pipeline);
        
                    let cube_time = (0x14..=0x17).contains(&pattern) && (row%16 <= 2);
                    if cube_time {
                        compute_pass.set_bind_group(0, &self.smoke_compute_bindgroup1, &[]);
                        compute_pass.set_bind_group(1, &self.smoke_shader_params_bindgroup[6], &[]);
                        compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, dispatch_depth);
                        compute_pass.set_bind_group(0, &self.smoke_compute_bindgroup2, &[]);
                        compute_pass.set_bind_group(1, &self.smoke_shader_params_bindgroup[7], &[]);
                        compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, dispatch_depth);
                    }
                    for i in 0..COMPUTE_PASSES {
                        let texture_bindgroup = match i % 2 {
                            1 => &self.smoke_compute_bindgroup2,
                            _ => &self.smoke_compute_bindgroup1,
                        };
                        compute_pass.set_bind_group(0, &texture_bindgroup, &[]);
                        compute_pass.set_bind_group(
                            1,
                            &self.smoke_shader_params_bindgroup[i as usize],
                            &[],
                        );
                        compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, dispatch_depth);
                    }
                }
            }
        }

        let beat_time = now.duration_since(self.beat).as_secs_f32();
        // let pattern_time = now.duration_since(self.pattern).as_secs_f64();
        let transition = now.duration_since(self.transitioned_at).as_secs_f32();
        
        self.bg_shader_params.t = time as f32;
        
        self.final_shader_params.shader_function = match self.transition {
            Transition::None => 0,
            Transition::Fade(_) => 1,
            Transition::Slide => 2,
            Transition::Blink => 3,
            Transition::Blink2 => 3,
        };
        self.final_shader_params.t = time as f32;
        self.final_shader_params.x = match self.scene {
            Scene::Slide(0) => -(1.0/transition),
            Scene::Slide(1) => (1.0/(transition+1.)-0.2).max(0.0),
            _ => 0.0
        };
        self.final_shader_params.x2 = match pattern {
            0x04..=0x0f | 0x18..=0x19 => (1./(beat_time+0.01))/5000.0,
            _ => 0.0
        };
        self.final_shader_params.transition = match self.transition {
            Transition::Blink => {
                let t = transition*0.2;
                (0.526*t).max(10.*t-9.)
            },
            Transition::Blink2 => transition*2.,
            Transition::Fade(duration) => transition/duration,
            _ => transition
        };

        if (now.duration_since(self.frame_log.0)).as_secs_f64() > 0.5 {
            self.frame_log.0 += Duration::from_millis(500);
            let fps = self.frame_log.1*2;
            #[cfg(target_arch = "wasm32")]
            {
                web_sys::window()
                    .and_then(|w| w.document())
                    .and_then(|d| {
                        let fps_label = d.get_element_by_id("fps")?;
                        fps_label.set_inner_html(&format!("{} FPS", fps));
                        Some(())
                    })
                    .unwrap();
            }
            log::info!(
                "{} frames per second",
                fps
            );
            self.frame_log.1 = 0;
        }
        self.frame_log.1 += 1;
        
        self.camera_uniform.update_view_proj(&self.camera);

        queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));
        queue.write_buffer(&self.final_function_buffer, 0, bytemuck::cast_slice(&[self.final_shader_params]));
        queue.write_buffer(&self.bg_function_buffer, 0, bytemuck::cast_slice(&[self.bg_shader_params]));
        let raw_instance = self.instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&raw_instance));
    }
    
    fn step(&mut self, encoder: &mut CommandEncoder, pattern: usize, row: usize) {
        if (self.current_step as usize) < STEPS.len() && pattern > self.next_step.0 || (pattern == self.next_step.0 && row >= self.next_step.1) {
            self.current_step += 1;
            self.copy_to_previous(encoder);
            self.transitioned_at = Instant::now();
            (_, self.scene, self.transition) = STEPS[self.current_step as usize];
            self.next_step = STEPS.get(self.current_step as usize+1)
                .map(|step| step.0)
                .unwrap_or((0xff,0xff));
        }
    }
    
    fn copy_to_previous(&mut self, encoder: &mut CommandEncoder) {
        let from_texture = match self.scene {
            Scene::Slide(number) => &self.slide_textures[number as usize],
            Scene::Black => &self.texture_pass1,
            Scene::CDs(_) => &self.texture_pass1,
            Scene::StarWars(_) => &self.texture_pass1,
            Scene::Ocean(_) => &self.texture_pass1,
            Scene::Smoke(_) => &self.texture_pass1
        }.texture.as_image_copy();
        encoder.copy_texture_to_texture(from_texture,
            self.previous_pass_texture.texture.as_image_copy(),
            Extent3d{
                width: 1920,
                height: 1080,
                depth_or_array_layers: 1
            }
        );
    }

    pub fn render(
        &mut self,
        view_final: &TextureView,
        depth_view: &TextureView,
        pipeline_final: &RenderPipeline,
        encoder: &mut CommandEncoder,
    ) {
        match self.scene {
            Scene::Slide(number) => {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Render Pass final"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view_final,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                render_pass.set_pipeline(pipeline_final);
    
                render_pass.set_vertex_buffer(0, self.full_quad_vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    self.full_quad_index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                render_pass.set_bind_group(1, &self.final_function_bindgroup, &[]);
                render_pass.set_bind_group(0, &self.slide_texture_bindgroups[number as usize], &[]);
                render_pass.set_bind_group(2, &self.previous_pass_texture_bind_group, &[]);
                render_pass.draw_indexed(0..6, 0, 0..1);
            }
            Scene::Black => {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass final"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view_final,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        ..Default::default()
                    });
                    render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                    render_pass.set_pipeline(pipeline_final);
        
                    render_pass.set_vertex_buffer(0, self.full_quad_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        self.full_quad_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.set_bind_group(1, &self.final_function_bindgroup, &[]);
                    render_pass.set_bind_group(0, &self.texture_pass1_bindgroup, &[]);
                    render_pass.set_bind_group(2, &self.previous_pass_texture_bind_group, &[]);
                    render_pass.draw_indexed(0..6, 0, 0..1);
            }
            Scene::CDs(number) => {
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass 1"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.texture_pass_window.view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });
                    render_pass.set_pipeline(&self.render_pipeline_cdrender);
        
                    render_pass.set_vertex_buffer(1, self.cd_instance_buffer.slice(..));
                    render_pass.set_vertex_buffer(0, self.full_quad_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        self.full_quad_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.set_bind_group(0, &self.object_uniform_bind_group, &[]);
                    render_pass.draw_indexed(0..6, 0, 0..NUM_CDS as u32+1);
                }
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass 2"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.texture_pass1.view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });
                    render_pass.set_pipeline(&self.render_pipeline_simple);
                    render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                    render_pass.set_vertex_buffer(0, self.full_quad_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        self.full_quad_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    // render background (slide)
                    render_pass.set_bind_group(1, &self.final_function_bindgroup, &[]);
                    render_pass.set_bind_group(0, &self.slide_texture_bindgroups[number as usize], &[]);
                    render_pass.set_bind_group(2, &self.slide_texture_bindgroups[(number as usize-1).max(7)], &[]);
                    render_pass.draw_indexed(0..6, 0, 0..1);
                    // render window (CDs)
                    render_pass.set_bind_group(1, &self.final_function_bindgroup, &[]);
                    render_pass.set_bind_group(0, &self.texture_pass_window_bindgroup, &[]);
                    render_pass.set_bind_group(2, &self.texture_pass_window_bindgroup, &[]);
                    render_pass.draw_indexed(0..6, 0, 1..2);
                }
                // Final pass (to screen)
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass final"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view_final,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        ..Default::default()
                    });
                    render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                    render_pass.set_pipeline(pipeline_final);
        
                    render_pass.set_vertex_buffer(0, self.full_quad_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        self.full_quad_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.set_bind_group(1, &self.final_function_bindgroup, &[]);
                    render_pass.set_bind_group(0, &self.texture_pass1_bindgroup, &[]);
                    render_pass.set_bind_group(2, &self.previous_pass_texture_bind_group, &[]);
                    render_pass.draw_indexed(0..6, 0, 0..1);
                }
            }
            Scene::StarWars(number) => {
                { // Pass 1: 
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass 1"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.texture_pass2.view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });
                    render_pass.set_pipeline(&self.render_pipeline_starwars1);
        
                    render_pass.set_vertex_buffer(1, self.starwars_instance_buffer.slice(..));
                    render_pass.set_bind_group(0, &self.object_uniform_bind_group, &[]);
                    render_pass.draw_model_instanced(&self.pewpew_model, 0..NUM_STARWARS as u32*2);
                }
                { // Pass 2: blur it
                        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Render Pass 2"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &self.texture_pass_window.view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: depth_view,
                                depth_ops: Some(wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(1.0),
                                    store: wgpu::StoreOp::Store,
                                }),
                                stencil_ops: None,
                            }),
                            ..Default::default()
                        });
                        render_pass.set_pipeline(&self.render_pipeline_starwars2);
                        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                        render_pass.set_vertex_buffer(0, self.full_quad_vertex_buffer.slice(..));
                        render_pass.set_index_buffer(
                            self.full_quad_index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        render_pass.set_bind_group(0, &self.texture_pass2_bindgroup, &[]);
                        render_pass.set_bind_group(1, &self.final_function_bindgroup, &[]);
                        render_pass.set_bind_group(2, &self.texture_pass2_bindgroup, &[]);
                        render_pass.draw_indexed(0..6, 0, 0..1);
                    }
                { // Pass 3: render the window
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass 3"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.texture_pass1.view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });
                    render_pass.set_pipeline(&self.render_pipeline_simple);
                    render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                    render_pass.set_vertex_buffer(0, self.full_quad_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        self.full_quad_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    // render background (slide)
                    render_pass.set_bind_group(1, &self.final_function_bindgroup, &[]);
                    render_pass.set_bind_group(0, &self.slide_texture_bindgroups[number as usize], &[]);
                    render_pass.set_bind_group(2, &self.slide_texture_bindgroups[(number as usize-1).max(7)], &[]);
                    render_pass.draw_indexed(0..6, 0, 0..1);
                    // render window (lasers)
                    render_pass.set_bind_group(1, &self.final_function_bindgroup, &[]);
                    render_pass.set_bind_group(0, &self.texture_pass_window_bindgroup, &[]);
                    render_pass.set_bind_group(2, &self.texture_pass_window_bindgroup, &[]);
                    render_pass.draw_indexed(0..6, 0, 2..3);
                }
                { // Final pass (to screen)
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass final"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view_final,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        ..Default::default()
                    });
                    render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                    render_pass.set_pipeline(pipeline_final);
        
                    render_pass.set_vertex_buffer(0, self.full_quad_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        self.full_quad_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.set_bind_group(1, &self.final_function_bindgroup, &[]);
                    render_pass.set_bind_group(0, &self.texture_pass1_bindgroup, &[]);
                    render_pass.set_bind_group(2, &self.previous_pass_texture_bind_group, &[]);
                    render_pass.draw_indexed(0..6, 0, 0..1);
                }
            }
            Scene::Ocean(number) => {
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass 1"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.texture_pass_window.view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });
                    render_pass.set_pipeline(&self.render_pipeline_ocean);
        
                    render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                    render_pass.set_vertex_buffer(0, self.full_quad_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        self.full_quad_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.set_bind_group(1, &self.final_function_bindgroup, &[]);
                    render_pass.set_bind_group(0, &self.ocean_texture_bindgroup, &[]);
                    render_pass.set_bind_group(2, &self.ocean_texture_bindgroup, &[]);
                    render_pass.draw_indexed(0..6, 0, 0..1);
                }
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass 2"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.texture_pass1.view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });
                    render_pass.set_pipeline(&self.render_pipeline_simple);
                    render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                    render_pass.set_vertex_buffer(0, self.full_quad_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        self.full_quad_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    // render background (slide)
                    render_pass.set_bind_group(1, &self.final_function_bindgroup, &[]);
                    render_pass.set_bind_group(0, &self.slide_texture_bindgroups[number as usize], &[]);
                    render_pass.set_bind_group(2, &self.slide_texture_bindgroups[(number as usize-1).max(7)], &[]);
                    render_pass.draw_indexed(0..6, 0, 0..1);
                    // render window (ocean)
                    render_pass.set_bind_group(1, &self.final_function_bindgroup, &[]);
                    render_pass.set_bind_group(0, &self.texture_pass_window_bindgroup, &[]);
                    render_pass.set_bind_group(2, &self.texture_pass_window_bindgroup, &[]);
                    render_pass.draw_indexed(0..6, 0, 1..2);
                }
                // Final pass (to screen)
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass final"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view_final,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        ..Default::default()
                    });
                    render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                    render_pass.set_pipeline(pipeline_final);
        
                    render_pass.set_vertex_buffer(0, self.full_quad_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        self.full_quad_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.set_bind_group(1, &self.final_function_bindgroup, &[]);
                    render_pass.set_bind_group(0, &self.texture_pass1_bindgroup, &[]);
                    render_pass.set_bind_group(2, &self.previous_pass_texture_bind_group, &[]);
                    render_pass.draw_indexed(0..6, 0, 0..1);
                }
            }
            Scene::Smoke(_) => {
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass 1"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.texture_pass1.view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });
                    render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                    render_pass.set_pipeline(&self.render_pipeline_smokerender);
        
                    render_pass.set_vertex_buffer(0, self.full_quad_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        self.full_quad_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.set_bind_group(1, &self.bg_uniform_bind_group, &[]);
                    render_pass.set_bind_group(0, &self.smoke_render_bind_group, &[]);
                    render_pass.set_bind_group(2, &self.lasers_uniform_bindgroup, &[]);
                    render_pass.draw_indexed(0..6, 0, 0..1);
                }
                // Final pass (to screen)
                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass final"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view_final,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        ..Default::default()
                    });
                    render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                    render_pass.set_pipeline(pipeline_final);
        
                    render_pass.set_vertex_buffer(0, self.full_quad_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        self.full_quad_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.set_bind_group(1, &self.final_function_bindgroup, &[]);
                    render_pass.set_bind_group(0, &self.texture_pass1_bindgroup, &[]);
                    render_pass.set_bind_group(2, &self.previous_pass_texture_bind_group, &[]);
                    render_pass.draw_indexed(0..6, 0, 0..1);
                }
            }
        }
    }
    
    #[cfg(target_arch = "wasm32")]
    pub fn resize_cube(&mut self, size: usize, device: &wgpu::Device) {
        if self.current_size != size && size > 0 {
            self.current_size = size;
            self.smoke_texture1 = texture::Texture::from_texture(
                device,
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("smoke texture 1"),
                    size: wgpu::Extent3d {
                        width: self.current_size as u32,
                        height: self.current_size as u32,
                        depth_or_array_layers: self.current_size as u32,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: wgpu::TextureFormat::Rgba32Float,
                    usage: wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &vec![],
                }),
                wgpu::FilterMode::Linear,
            );
    
            self.smoke_texture2 = texture::Texture::from_texture(
                device,
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("smoke texture 2"),
                    size: wgpu::Extent3d {
                        width: self.current_size as u32,
                        height: self.current_size as u32,
                        depth_or_array_layers: self.current_size as u32,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: wgpu::TextureFormat::Rgba32Float,
                    usage: wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &vec![],
                }),
                wgpu::FilterMode::Linear,
            );
    
            self.packed_smoke_texture = texture::Texture::from_texture(
                device,
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("packed smoke texture"),
                    size: wgpu::Extent3d {
                        width: self.current_size as u32,
                        height: self.current_size as u32,
                        depth_or_array_layers: self.current_size as u32,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: wgpu::TextureFormat::Rgba32Uint,
                    usage: wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &vec![],
                }),
                wgpu::FilterMode::Linear,
            );
    
            self.poisson_texture1 = texture::Texture::from_texture(
                device,
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("poisson texture 1"),
                    size: wgpu::Extent3d {
                        width: self.current_size as u32,
                        height: self.current_size as u32,
                        depth_or_array_layers: self.current_size as u32,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &vec![],
                }),
                wgpu::FilterMode::Linear,
            );
    
            self.poisson_texture2 = texture::Texture::from_texture(
                device,
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("poisson texture 2"),
                    size: wgpu::Extent3d {
                        width: self.current_size as u32,
                        height: self.current_size as u32,
                        depth_or_array_layers: self.current_size as u32,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &vec![],
                }),
                wgpu::FilterMode::Linear,
            );
    
            self.smoke_compute_bindgroup1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Smoke compute bind group 1"),
                layout: &self.compute_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.smoke_texture1.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.poisson_texture1.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.smoke_texture2.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.poisson_texture2.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&self.packed_smoke_texture.view),
                    },
                ],
            });
            self.smoke_compute_bindgroup2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Smoke compute bind group 2"),
                layout: &self.compute_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.smoke_texture2.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.poisson_texture2.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.smoke_texture1.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.poisson_texture1.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&self.packed_smoke_texture.view),
                    },
                ],
            });
            self.smoke_render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Smoke render bind group"),
                layout: &self.smoke_render_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.packed_smoke_texture.view),
                }],
            });
        }
    }
}

#[derive(Copy,Clone,Debug)]
enum Transition {
    None,
    Fade(f32),
    Slide,
    Blink,
    Blink2
}

#[derive(Copy,Clone,Debug)]
enum Scene {
    Slide(i32),
    Black,
    CDs(i32),
    StarWars(i32),
    Ocean(i32),
    Smoke(i32),
}

pub struct Camera {
    pub eye: cgmath::Point3<f32>,
    pub target: cgmath::Point3<f32>,
    pub up: cgmath::Vector3<f32>,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        proj * view
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = (OPENGL_TO_WGPU_MATRIX * camera.build_view_projection_matrix()).into();
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LasersUniform {
    pub laser1_transform: [[f32;4];4],
    pub laser2_transform: [[f32;4];4],
    pub laser3_transform: [[f32;4];4],
    pub laser4_transform: [[f32;4];4],
    pub laser1_color: [f32;4],
    pub laser2_color: [f32;4],
    pub laser3_color: [f32;4],
    pub laser4_color: [f32;4]
}

impl LasersUniform {
    fn new() -> Self {
        LasersUniform {
            laser1_transform: Matrix4::identity().into(),
            laser2_transform: Matrix4::identity().into(),
            laser3_transform: Matrix4::identity().into(),
            laser4_transform: Matrix4::identity().into(),
            laser1_color: [0.;4],
            laser2_color: [0.;4],
            laser3_color: [0.;4],
            laser4_color: [0.;4],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ComputeParamsUniform {
    pub step: i32,
    pub delta_time: f32,
    pub time: f32,
    pub x: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderParamsUniform {
    pub shader_function: i32,
    pub t: f32,
    pub x: f32,
    pub transition: f32,
    pub x2: f32,
}

fn start_audio_player(player: Arc<Mutex<XmrsPlayer>>) -> Result<(), cpal::StreamError> {
    let mut host = cpal::default_host();
    let mut device = host.default_output_device().unwrap();
    #[cfg(target_os = "windows")]
    {
        let wasapi_host = cpal::platform::WasapiHost::new();
        if let Ok(wasapi_host) = wasapi_host {
            let wasapi_device = wasapi_host.default_output_device().unwrap();
            host = wasapi_host.into();
            device = wasapi_device.into();
        }
    }

    let config = device
        .default_output_config()
        .expect("failed to get default output config");
    let mut config = config.config();
    config.buffer_size = BufferSize::Fixed(256);
    
    std::thread::spawn(move || {
        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let mut player_lock = player.lock().unwrap();
                    for sample in data.iter_mut() {
                        *sample = player_lock.next().unwrap_or(0.0);
                    }
                },
                |_: cpal::StreamError| {},
                None,
            )
            .expect("failed to build output stream");

        stream.play().expect("failed to play stream");
        std::thread::sleep(std::time::Duration::from_secs_f32(300.0));
    });

    Ok(())
}
