#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use web_sys::HtmlInputElement;
use web_time::{Instant, Duration};

use cgmath::{Quaternion, Rotation3, SquareMatrix, Vector3};
//use cpal::traits::{HostTrait, StreamTrait};
use rand::{Rng, SeedableRng};
//use rodio::DeviceTrait;
use wgpu::{
    core::validation::BindingError, util::DeviceExt, BindGroup, Buffer, CommandEncoder, ComputePipeline, RenderPipeline, TextureFormat, TextureView
};
//use xmrs::xm::xmmodule::XmModule;
//use xmrsplayer::xmrsplayer::XmrsPlayer;

use crate::{
    resources::{ASSETS, QUAD_INDICES, QUAD_VERTICES},
    texture::{self, Texture}, Instance, FLUID_SIZE, OPENGL_TO_WGPU_MATRIX,
    model::{Model,Vertex}
};

const COMPUTE_PASSES: i32 = 5;
const COMPUTE_EXTRAS: i32 = 2;

// This file is where the fun happens! It's also the worst spaghetti ever devised.

fn compute_work_group_count(
    (width, height, depth): (u32, u32, u32),
    (workgroup_width, workgroup_height, workgroup_depth): (u32, u32, u32),
) -> (u32, u32, u32) {
    let x = (width + workgroup_width - 1) / workgroup_width;
    let y = (height + workgroup_height - 1) / workgroup_height;
    let z = (depth + workgroup_depth - 1) / workgroup_depth;

    (x, y, z)
}

// fn beat(x: f32) -> f32 {
//     (((1.0 / ((x % 1.0) + 0.8)).powf(3.0) - 1.0) * 0.3) + 1.1
// }

// const SAMPLE_RATE: u32 = 44100;

pub struct Demo {
    scene: Scene,
    transition: Transition,
    transitioned_at: Instant,
    full_quad_vertex_buffer: Buffer,
    full_quad_index_buffer: Buffer,
    pub instances: Vec<Instance>,
    pub cd_instances: Vec<Instance>,
    pub instance_buffer: Buffer,
    pub cd_instance_buffer: Buffer,
    camera_buffer: wgpu::Buffer,
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
    pub uniform_bind_group_layout: wgpu::BindGroupLayout,
    bg_function_buffer: wgpu::Buffer,
    bg_uniform_bind_group: wgpu::BindGroup,
    fg_function_buffer: wgpu::Buffer,
    fg_uniform_bind_group: wgpu::BindGroup,
    object_uniform_bind_group: wgpu::BindGroup,
    final_function_buffer: wgpu::Buffer,
    final_function_bindgroup: BindGroup,
    final_pass_texture_bind_group: wgpu::BindGroup,
    start_time: Instant,
    last_time: Instant,
    last_row: usize,
    beat: Instant,
    last_pattern: usize,
    pattern: Instant,
    rng: rand::rngs::SmallRng,
    //player: Arc<Mutex<XmrsPlayer>>,
    pub bg_shader_params: ShaderParamsUniform,
    pub fg_shader_params: ShaderParamsUniform,
    pub final_shader_params: ShaderParamsUniform,
    pub camera: Camera,
    pub camera_uniform: CameraUniform,
    slide_texture_bindgroups: Vec<BindGroup>,
    
    render_pipeline_smokerender: RenderPipeline,
    render_pipeline_cdrender: RenderPipeline,
    render_pipeline_simple: RenderPipeline,
    
    texture_pass1: texture::Texture,
    texture_pass_window: texture::Texture,
    texture_pass_window_bindgroup: BindGroup,
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
        for i in 0..100 {
            cd_instances.push(Instance {
                position: cgmath::Vector3::new(rng.gen_range(-30.0..30.0), rng.gen_range(-16.0..16.0), rng.gen_range(-25.0..0.0)),
                rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(i as f32*3.0)),
                scale: cgmath::Vector3::new(1.0,1.0,1.0),
                tex_offset: cgmath::Vector2::new(0.0, 0.0)
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
            transition: 0.0
        };
        let fg_shader_params = ShaderParamsUniform {
            shader_function: 0,
            t: 0.0,
            x: 0.0,
            transition: 0.0
        };
        let final_shader_params = ShaderParamsUniform {
            shader_function: 0,
            t: 0.0,
            x: 0.0,
            transition: 0.0
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

        let textured_function_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Textured params buffer"),
                contents: bytemuck::cast_slice(&[ShaderParamsUniform::new()]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bg_function_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Background params buffer"),
            contents: bytemuck::cast_slice(&[bg_shader_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let fg_function_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Foreground params buffer"),
            contents: bytemuck::cast_slice(&[fg_shader_params]),
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

        let fg_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: none_camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: fg_function_buffer.as_entire_binding(),
                },
            ],
            label: Some("uniform_bind_group_2"),
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
                            resource: fg_function_buffer.as_entire_binding(),
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

        let smoke_shader_params: Vec<ComputeParamsUniform> = (0..=COMPUTE_PASSES + COMPUTE_EXTRAS)
            .map(|i| ComputeParamsUniform {
                step: i,
                delta_time: 0.0,
                time: 0.0,
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
                        Some(Texture::from_bytes(device, queue, f.contents(), path).unwrap())
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
        let shader_simple = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaderpass_simple.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaderpass_simple.wgsl").into()),
        });
        
        let smoke_render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                bind_group_layouts: &[&smoke_render_bind_group_layout, &uniform_bind_group_layout],
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
        
        let render_pipeline_layout_simple =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render layout simple"),
                bind_group_layouts: &[&texture_bind_group_layout, &uniform_bind_group_layout, &texture_bind_group_layout],
                push_constant_ranges: &[],
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
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            label: None,
            view_formats: &vec![]
        });
        let texture_pass1 = Texture::from_texture(&device, texture_pass1, wgpu::FilterMode::Linear);
        
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

        let final_pass_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        
        
        // The music:

        //let xm = XmModule::load(MUSIC).unwrap();
        // let player = Arc::new(Mutex::new(XmrsPlayer::new(
        //     xm.to_module().into(),
        //     SAMPLE_RATE as f32,
        // )));
        // {
        //     let mut player_lock = player.lock().unwrap();
        //     player_lock.goto(0, 0);
        // }
        //start_audio_player(player.clone()).expect("failed to start player");

        Demo {
            scene: Scene::CDs(8),
            transition: Transition::None,
            transitioned_at: Instant::now(),
            full_quad_vertex_buffer,
            full_quad_index_buffer,
            instances,
            cd_instances,
            instance_buffer,
            cd_instance_buffer,
            start_time: Instant::now(),
            last_time: Instant::now(),
            last_row: 0,
            beat: Instant::now(),
            last_pattern: 0,
            pattern: Instant::now(),
            rng,
            //player,
            bg_shader_params,
            fg_shader_params,
            final_shader_params,
            texture_bind_group_layout,
            uniform_bind_group_layout,
            fg_uniform_bind_group,
            bg_uniform_bind_group,
            object_uniform_bind_group,
            final_pass_texture_bind_group,
            camera,
            camera_uniform,
            camera_buffer,
            bg_function_buffer,
            fg_function_buffer,
            final_function_buffer,
            final_function_bindgroup,
            texture_pass1,
            texture_pass_window,
            texture_pass_window_bindgroup,
            previous_pass_texture_bind_group,
            slide_texture_bindgroups,

            smoke_render_bind_group_layout,
            smoke_render_bind_group,
            
            render_pipeline_smokerender,
            render_pipeline_cdrender,
            render_pipeline_simple,

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
        
        self.final_shader_params.shader_function = match self.transition {
            Transition::None => 0,
            Transition::Fade => 1,
            Transition::Slide => 2
        };
        self.final_shader_params.transition = time as f32;

        match &self.scene {
            Scene::Slide(_) => {}
            Scene::Black => {
                queue.write_texture(
                    self.texture_pass1.texture.as_image_copy(),
                    vec![0; 4*1920*1080].as_slice(),
                    wgpu::ImageDataLayout::default(),
                    self.texture_pass1.texture.size());
            }
            Scene::CDs(_) => {
                for i in 0..99 {
                    self.cd_instances[i].position.z += 10.*delta_time as f32;
                    if self.cd_instances[i].position.z >= 0. {
                        self.cd_instances[i].position.z = self.rng.gen_range(-40.0..-20.0);
                    }
                    self.cd_instances[i].rotation = 
                        Quaternion::from_angle_x(cgmath::Rad(i as f32+time as f32))
                        *Quaternion::from_angle_y(cgmath::Rad(i as f32-time as f32));
                }
                self.cd_instances[99].position = Vector3::new(0.0,0.0,7.5);
                self.cd_instances[99].rotation = 
                    Quaternion::from_angle_x(cgmath::Rad(1.*(time as f32).cos()))
                    *Quaternion::from_angle_y(cgmath::Rad(1.*(time as f32).sin()));
                    
                let instance_data = self.cd_instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
                queue.write_buffer(&self.cd_instance_buffer, 0, bytemuck::cast_slice(&instance_data));
            }
            Scene::StarWars => {}
            Scene::Smoke => {
                for (i, params) in self.smoke_shader_params.iter_mut().enumerate() {
                    params.delta_time = delta_time as f32;
                    params.time = time as f32;
                    queue.write_buffer(
                        &self.smoke_shader_params_buffer[i],
                        0,
                        bytemuck::cast_slice(&[*params]),
                    );
                }
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
        
                    // if time as i32 % 3 == 2 {
                    //     compute_pass.set_bind_group(0, &self.smoke_compute_bindgroup1, &[]);
                    //     compute_pass.set_bind_group(1, &self.smoke_shader_params_bindgroup[5], &[]);
                    //     compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, dispatch_depth);
                    //     compute_pass.set_bind_group(0, &self.smoke_compute_bindgroup2, &[]);
                    //     compute_pass.set_bind_group(1, &self.smoke_shader_params_bindgroup[6], &[]);
                    //     compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, dispatch_depth);
                    // }
                    for i in 0..=COMPUTE_PASSES {
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

        {
            //let player = self.player.lock().unwrap();
            // let row = player.get_current_row();
            // if row % 4 == 0 && row != self.last_row {
            //     self.beat = Instant::now();
            // }
            // self.last_row = row;
            // let pattern = player.get_current_pattern();
            // if pattern != self.last_pattern && pattern <= 9 {
            //     self.pattern = Instant::now();
            // }
            // self.last_pattern = pattern;
        }
        //let beat_time = now.duration_since(self.beat).as_secs_f64();
        let pattern_time = now.duration_since(self.pattern).as_secs_f64();
        self.bg_shader_params.t = time as f32;
        self.fg_shader_params.t = time as f32;
        self.fg_shader_params.x = -0.1 + 1.0 / (pattern_time * 4.0 + 0.2) as f32;
        //self.instances[5].tex_offset.y = -0.5 * pattern_time as f32;

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
        queue.write_buffer(&self.fg_function_buffer, 0, bytemuck::cast_slice(&[self.fg_shader_params]));
        let raw_instance = self.instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&raw_instance));
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
                if number > 0 {
                    render_pass.set_bind_group(2, &self.slide_texture_bindgroups[number as usize-1], &[]);
                } else {
                    render_pass.set_bind_group(2, &self.slide_texture_bindgroups[number as usize], &[]);
                }
                render_pass.draw_indexed(0..6, 0, 0..1);
            }
            Scene::Black => {}
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
                        render_pass.draw_indexed(0..6, 0, 0..100);
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
                        render_pass.set_bind_group(2, &self.slide_texture_bindgroups[number as usize-1], &[]);
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
                        render_pass.set_bind_group(0, &self.final_pass_texture_bind_group, &[]);
                        render_pass.set_bind_group(2, &self.previous_pass_texture_bind_group, &[]);
                        render_pass.draw_indexed(0..6, 0, 0..1);
                    }}
            Scene::StarWars => {}
            Scene::Smoke => {
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
                    render_pass.set_bind_group(0, &self.final_pass_texture_bind_group, &[]);
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

enum Transition {
    None,
    Fade,
    Slide,
}

enum Scene {
    Slide(i32),
    Black,
    CDs(i32),
    StarWars,
    Smoke,
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
pub struct ComputeParamsUniform {
    pub step: i32,
    pub delta_time: f32,
    pub time: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderParamsUniform {
    pub shader_function: i32,
    pub t: f32,
    pub x: f32,
    pub transition: f32
}

impl ShaderParamsUniform {
    pub fn new() -> Self {
        Self {
            shader_function: 0,
            t: 0.0,
            x: 0.0,
            transition: 0.0
        }
    }
}

// fn start_audio_player(player: Arc<Mutex<XmrsPlayer>>) -> Result<(), cpal::StreamError> {
//     let host = cpal::default_host();
//     let device = host
//         .default_output_device()
//         .expect("no output device available");

//     let config = device
//         .default_output_config()
//         .expect("failed to get default output config");

//     std::thread::spawn(move || {
//         let stream = device
//             .build_output_stream(
//                 &config.config(),
//                 move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
//                     let mut player_lock = player.lock().unwrap();
//                     for sample in data.iter_mut() {
//                         *sample = player_lock.next().unwrap_or(0.0);
//                     }
//                 },
//                 |_: cpal::StreamError| {},
//                 None,
//             )
//             .expect("failed to build output stream");

//         stream.play().expect("failed to play stream");
//         std::thread::sleep(std::time::Duration::from_secs_f32(300.0));
//     });

//     Ok(())
// }
