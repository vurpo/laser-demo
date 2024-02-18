use std::{time::{Instant, Duration}, sync::{Arc, Mutex}, ops::Range};

use cgmath::{Rotation3, SquareMatrix, Point3};
use cpal::traits::{HostTrait, StreamTrait};
use rand::{SeedableRng, Rng, distributions::WeightedIndex, prelude::Distribution};
use rodio::DeviceTrait;
use wgpu::{util::DeviceExt, BindGroup, Buffer, CommandEncoder, ComputePipeline, RenderPipeline, TextureView};
use xmrs::xm::xmmodule::XmModule;
use xmrsplayer::xmrsplayer::XmrsPlayer;

use crate::{
    model::{DrawModel, Model}, resources::{self, FULL_SCREEN_INDICES, FULL_SCREEN_VERTICES, MUSIC}, texture, Instance, FLUID_SCALE, FLUID_SIZE, OPENGL_TO_WGPU_MATRIX
};

const COMPUTE_PASSES: i32 = 5;

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

fn beat(x: f32) -> f32 {
    (((1.0/((x%1.0)+0.8)).powf(3.0)-1.0)*0.3)+1.1
}

fn row_to_range(row: usize) -> Range<u32> {
    let start = match row {
        0x00..=0x47 => 1,
        0x48..=0x57 => 2,
        0x58..=0x67 => 3,
        0x68..=0x77 => 4,
        _ => 5,
    };
    let end = match row {
        0x00..=0x07 => 1,
        0x08..=0x17 => 2,
        0x18..=0x27 => 3,
        0x28..=0x37 => 4,
        _ => 5,
    };
    start..end
}

const WEIGHTS: &[f32] = &[1.0, 1.0, 0.2, 0.4, 1.0, 1.0];

const SHADER_FUNCTION_COOL: i32 = 1;
const SHADER_FUNCTION_WHITE: i32 = 2;
const SHADER_FUNCTION_TRANS: i32 = 3;
const SHADER_FUNCTION_COOL_2: i32 = 4;
const SHADER_FUNCTION_COOL_3: i32 = 5;

const DECAL_MODEL: usize = 7;
const GROUND_MODEL: usize = 6;

const SAMPLE_RATE: u32 = 44100;

pub struct Demo {
    scene: Scene,
    models: Vec<Model>,
    weights: WeightedIndex<f32>,
    vehicle: Option<(usize, Instant, Duration)>,
    full_quad_vertex_buffer: Buffer,
    full_quad_index_buffer: Buffer,
    pub instances: Vec<Instance>,
    pub instance_buffer: Buffer,
    camera_buffer: wgpu::Buffer,
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
    pub uniform_bind_group_layout: wgpu::BindGroupLayout,
    uniform_bind_group: wgpu::BindGroup,
    bg_function_buffer: wgpu::Buffer,
    bg_uniform_bind_group: wgpu::BindGroup,
    fg_function_buffer: wgpu::Buffer,
    fg_uniform_bind_group: wgpu::BindGroup,
    final_pass_uniform_bind_group: wgpu::BindGroup,
    decal_uniform_bind_group: wgpu::BindGroup,
    decal_bindgroups: Vec<BindGroup>,
    ground_bindgroups: Vec<BindGroup>,
    start_time: Instant,
    last_time: Instant,
    last_row: usize,
    beat: Instant,
    last_pattern: usize,
    pattern: Instant,
    rng: rand::rngs::SmallRng,
    player: Arc<Mutex<XmrsPlayer>>,
    pub bg_shader_params: ShaderParamsUniform,
    pub fg_shader_params: ShaderParamsUniform,
    pub camera: Camera,
    pub camera_uniform: CameraUniform,
    texture_pass1: texture::Texture,

    pub smoke_render_bind_group_layout: wgpu::BindGroupLayout,
    smoke_render_bind_group: wgpu::BindGroup,

    compute_pipeline: ComputePipeline,
    pub smoke_texture_bind_group_layout: wgpu::BindGroupLayout,
    // smoke_texture1: texture::Texture,
    // smoke_texture2: texture::Texture,
    smoke_compute_bindgroup1: BindGroup,
    smoke_compute_bindgroup2: BindGroup,
    smoke_shader_params: Vec<ComputeParamsUniform>,
    smoke_shader_params_buffer: Vec<wgpu::Buffer>,
    smoke_shader_params_bindgroup: Vec<BindGroup>
}


impl Demo {
    pub async fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_pass1: texture::Texture,
    ) -> Self {
        let instances = vec![
            Instance {
                position: cgmath::Vector3::new(0.0, 0.0, 0.0),
                rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::new(0.0, 0.0, 1.0), cgmath::Rad(0.0)),
                tex_offset: cgmath::Vector2::new(0.0, 0.0)
            },
            Instance {
                position: cgmath::Vector3::new(-0.9, -0.7, 0.0),
                rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::new(0.0, 0.0, 1.0), cgmath::Rad(-0.2)),
                tex_offset: cgmath::Vector2::new(0.0, 0.25)
            },
            Instance {
                position: cgmath::Vector3::new(0.9, -0.7, 0.0),
                rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::new(0.0, 0.0, 1.0), cgmath::Rad(0.2)),
                tex_offset: cgmath::Vector2::new(0.0, 0.5)
            },
            Instance {
                position: cgmath::Vector3::new(-0.9, 0.7, 0.0),
                rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::new(0.0, 0.0, 1.0), cgmath::Rad(0.2)),
                tex_offset: cgmath::Vector2::new(0.0, 0.75)
            },
            Instance {
                position: cgmath::Vector3::new(0.9, 0.7, 0.0),
                rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::new(0.0, 0.0, 1.0), cgmath::Rad(-0.2)),
                tex_offset: cgmath::Vector2::new(0.0, 0.0)
            },
            Instance {
                position: cgmath::Vector3::new(0.0, 0.0, 0.0),
                rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::new(0.0, 0.0, 1.0), cgmath::Rad(0.0)),
                tex_offset: cgmath::Vector2::new(0.0, 0.0)
            },
        ];

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let camera = Camera {
            eye: (0.0, 5.0, -10.0).into(),
            target: (-2.0, 0.0, -2.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: 16.0 / 9.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let full_quad_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fullscreen quad vertex buffer"),
            contents: bytemuck::cast_slice(&FULL_SCREEN_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let full_quad_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fullscreen quad index buffer"),
            contents: bytemuck::cast_slice(&FULL_SCREEN_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let bg_shader_params = ShaderParamsUniform { shader_function: SHADER_FUNCTION_COOL, t: 0.0, x: 0.0 };
        let fg_shader_params = ShaderParamsUniform { shader_function: SHADER_FUNCTION_WHITE, t: 0.0, x: 0.0 };

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let none_camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("No Camera Buffer"),
            contents: bytemuck::cast_slice(&[CameraUniform { view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ] }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let none_camera_buffer_stretched = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("No Camera Buffer"),
            contents: bytemuck::cast_slice(&[CameraUniform { view_proj: [
                [1.0/camera.aspect, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ] }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let textured_function_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Textured Function Buffer"),
            contents: bytemuck::cast_slice(&[ShaderParamsUniform::new()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bg_function_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Background Function Buffer"),
            contents: bytemuck::cast_slice(&[bg_shader_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let fg_function_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Foreground Function Buffer"),
            contents: bytemuck::cast_slice(&[fg_shader_params]),
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
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("uniform_bind_group_layout"),
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }, wgpu::BindGroupEntry {
                binding: 1,
                resource: textured_function_buffer.as_entire_binding(),
            }],
            label: Some("uniform_bind_group_0"),
        });

        let bg_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: none_camera_buffer.as_entire_binding(),
            }, wgpu::BindGroupEntry {
                binding: 1,
                resource: bg_function_buffer.as_entire_binding(),
            }],
            label: Some("uniform_bind_group_1"),
        });

        let fg_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: none_camera_buffer.as_entire_binding(),
            }, wgpu::BindGroupEntry {
                binding: 1,
                resource: fg_function_buffer.as_entire_binding(),
            }],
            label: Some("uniform_bind_group_2"),
        });

        let decal_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: none_camera_buffer_stretched.as_entire_binding(),
            }, wgpu::BindGroupEntry {
                binding: 1,
                resource: textured_function_buffer.as_entire_binding(),
            }],
            label: Some("uniform_bind_group_3"),
        });

        let final_pass_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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

        // Smoke simulation compute stuff:

        let smoke_texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                        view_dimension: wgpu::TextureViewDimension::D3
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D3
                    },
                    count: None,
                },
            ],
            label: Some("smoke_texture_bind_group_layout"),
        });
        let smoke_shader_params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            bind_group_layouts: &[&smoke_texture_bind_group_layout, &smoke_shader_params_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Smoke compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader_smoke_compute,
            entry_point: "fluid_main",
        });

        let smoke_texture1 = texture::Texture::from_texture(device, device.create_texture(&wgpu::TextureDescriptor {
            label: Some("smoke texture 1"),
            size: wgpu::Extent3d{
                width: FLUID_SIZE.0 as u32,
                height: FLUID_SIZE.1 as u32,
                depth_or_array_layers: FLUID_SIZE.2 as u32
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba32Float,
            usage:  wgpu::TextureUsages::COPY_DST |
                    wgpu::TextureUsages::COPY_SRC |
                    wgpu::TextureUsages::TEXTURE_BINDING |
                    wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &vec![]
        }), wgpu::FilterMode::Linear);

        let smoke_texture2 = texture::Texture::from_texture(device, device.create_texture(&wgpu::TextureDescriptor {
            label: Some("smoke texture 2"),
            size: wgpu::Extent3d{
                width: FLUID_SIZE.0 as u32,
                height: FLUID_SIZE.1 as u32,
                depth_or_array_layers: FLUID_SIZE.2 as u32
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba32Float,
            usage:  wgpu::TextureUsages::COPY_DST |
                    wgpu::TextureUsages::COPY_SRC |
                    wgpu::TextureUsages::TEXTURE_BINDING |
                    wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &vec![]
        }), wgpu::FilterMode::Linear);

        let poisson_texture1 = texture::Texture::from_texture(device, device.create_texture(&wgpu::TextureDescriptor {
            label: Some("poisson texture 1"),
            size: wgpu::Extent3d{
                width: FLUID_SIZE.0 as u32,
                height: FLUID_SIZE.1 as u32,
                depth_or_array_layers: FLUID_SIZE.2 as u32
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage:  wgpu::TextureUsages::COPY_DST |
                    wgpu::TextureUsages::COPY_SRC |
                    wgpu::TextureUsages::TEXTURE_BINDING |
                    wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &vec![]
        }), wgpu::FilterMode::Linear);

        let poisson_texture2 = texture::Texture::from_texture(device, device.create_texture(&wgpu::TextureDescriptor {
            label: Some("poisson texture 2"),
            size: wgpu::Extent3d{
                width: FLUID_SIZE.0 as u32,
                height: FLUID_SIZE.1 as u32,
                depth_or_array_layers: FLUID_SIZE.2 as u32
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage:  wgpu::TextureUsages::COPY_DST |
                    wgpu::TextureUsages::COPY_SRC |
                    wgpu::TextureUsages::TEXTURE_BINDING |
                    wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &vec![]
        }), wgpu::FilterMode::Linear);

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
                ],
            });
        
        let smoke_shader_params: Vec<ComputeParamsUniform> = (0..=COMPUTE_PASSES).map(|i| ComputeParamsUniform {
            step: i,
            delta_time: 0.0
        }).collect();
        let smoke_shader_params_buffer: Vec<wgpu::Buffer> = smoke_shader_params.iter().map(|p| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&"Smoke shader params buffer"),
            contents: bytemuck::cast_slice(&[*p]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })).collect();
        let smoke_shader_params_bindgroup: Vec<BindGroup> = smoke_shader_params_buffer.iter().map(|b| device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Smoke shader params bind group"),
            layout: &compute_pipeline.get_bind_group_layout(1),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: b.as_entire_binding(),
            }],
        })).collect();


        let smoke_render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D3,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                ],
                label: Some("smoke_render_bind_group_layout"),
            });
        let smoke_render_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor{
                label: Some("Smoke render bind group"),
                layout: &smoke_render_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&smoke_texture1.view),
                    },
                ],
            });

        // The music:

        let xm = XmModule::load(MUSIC).unwrap();
        let player = Arc::new(Mutex::new(XmrsPlayer::new(
            xm.to_module().into(),
            SAMPLE_RATE as f32,
        )));
        // {
        //     let mut player_lock = player.lock().unwrap();
        //     player_lock.goto(0, 0);
        // }
        //start_audio_player(player.clone()).expect("failed to start player");

        Demo {
            scene: Scene::Intro,
            models: vec![
                resources::load_model("m100.obj", device, queue, &texture_bind_group_layout, 1.0)
                    .await
                    .unwrap(),
                resources::load_model("edo.obj", device, queue, &texture_bind_group_layout, 0.8)
                    .await
                    .unwrap(),
                resources::load_model("boing.obj", device, queue, &texture_bind_group_layout, 5.0)
                    .await
                    .unwrap(),
                resources::load_model("boat.obj", device, queue, &texture_bind_group_layout, 2.0)
                    .await
                    .unwrap(),
                resources::load_model("tgv.obj", device, queue, &texture_bind_group_layout, 1.2)
                    .await
                    .unwrap(),
                resources::load_model("manse.obj", device, queue, &texture_bind_group_layout, 1.3)
                    .await
                    .unwrap(),
                resources::load_model("flat.obj", device, queue, &texture_bind_group_layout, 100.0)
                    .await
                    .unwrap(),
                resources::load_model("decal.obj", device, queue, &texture_bind_group_layout, 0.7)
                    .await
                    .unwrap(),
            ],
            weights: WeightedIndex::new(WEIGHTS).unwrap(),
            vehicle: None,
            full_quad_vertex_buffer,
            full_quad_index_buffer,
            instances,
            instance_buffer,
            decal_bindgroups: vec![
                resources::load_texture("decal1.png", device, queue).await.unwrap(),
                resources::load_texture("decal2.png", device, queue).await.unwrap(),
                resources::load_texture("decal3.png", device, queue).await.unwrap(),
                resources::load_texture("decal4.png", device, queue).await.unwrap(),
                resources::load_texture("decal5.png", device, queue).await.unwrap(),
            ].iter().map(|t| device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&t.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&t.sampler),
                    },
                ],
                label: None,
            })).collect(),
            ground_bindgroups: vec![
                resources::load_texture("grass1.png", device, queue).await.unwrap(),
                resources::load_texture("grass2.png", device, queue).await.unwrap(),
                resources::load_texture("grass3.png", device, queue).await.unwrap(),
                resources::load_texture("grass4.png", device, queue).await.unwrap(),
            ].iter().map(|t| device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&t.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&t.sampler),
                    },
                ],
                label: None,
            })).collect(),
            start_time: Instant::now(),
            last_time: Instant::now(),
            last_row: 0,
            beat: Instant::now(),
            last_pattern: 0,
            pattern: Instant::now(),
            rng: rand::rngs::SmallRng::seed_from_u64(0x4375746552616363),
            player,
            bg_shader_params,
            fg_shader_params,
            texture_bind_group_layout,
            uniform_bind_group_layout,
            uniform_bind_group,
            fg_uniform_bind_group,
            bg_uniform_bind_group,
            final_pass_uniform_bind_group,
            decal_uniform_bind_group,
            camera,
            camera_uniform,
            camera_buffer,
            bg_function_buffer,
            fg_function_buffer,
            texture_pass1,

            smoke_render_bind_group_layout,
            smoke_render_bind_group,

            compute_pipeline,
            smoke_texture_bind_group_layout,
            // smoke_texture1,
            // smoke_texture2,
            smoke_compute_bindgroup1,
            smoke_compute_bindgroup2,
            smoke_shader_params,
            smoke_shader_params_buffer,
            smoke_shader_params_bindgroup,
        }
    }

    fn random_vehicle(&mut self) -> (usize, Instant, Duration) {
        let idx = self.weights.sample(&mut self.rng);
        let mut weights = Vec::from(WEIGHTS);
        weights[idx] = 0.0;
        let _ = self.weights.update_weights((0..weights.len()).zip(weights.iter()).collect::<Vec<(usize, &f32)>>().as_slice());

        let duration = Duration::from_secs_f32(match idx {
            0 => self.rng.gen_range(0.3..0.8), //m100
            1 => self.rng.gen_range(0.3..0.8), //edo
            2 => self.rng.gen_range(0.15..0.3),//boing
            3 => self.rng.gen_range(0.2..0.4), //färjan
            4 => self.rng.gen_range(0.3..0.8), //tgv
            5 => self.rng.gen_range(0.3..0.8), //manse
            _ => panic!()
        });
        (idx, Instant::now(), duration)
    }

    pub fn update(&mut self, queue: &wgpu::Queue, encoder: &mut CommandEncoder) {
        let now = Instant::now();
        let time = now.duration_since(self.start_time).as_secs_f64();
        let delta_time = now.duration_since(self.last_time).as_secs_f64();
        self.last_time = now;

        for (i, p) in self.smoke_shader_params.iter_mut().enumerate() {
            p.delta_time = delta_time as f32;
            queue.write_buffer(
                &self.smoke_shader_params_buffer[i], 
                0, 
                bytemuck::cast_slice(&[*p]));
        }
        {
            let (dispatch_width, dispatch_height, dispatch_depth) = compute_work_group_count(
                (FLUID_SIZE.0 as u32, FLUID_SIZE.1 as u32, FLUID_SIZE.2 as u32),
                (8, 8, 4)
            );
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Smoke pass"),
            });
            compute_pass.set_pipeline(&self.compute_pipeline);

            for i in 0..=COMPUTE_PASSES {
                let texture_bindgroup = match i%2 {
                    0 => &self.smoke_compute_bindgroup1,
                    1 => &self.smoke_compute_bindgroup2,
                    _ => unreachable!()
                };
                compute_pass.set_bind_group(0, &texture_bindgroup, &[]);
                compute_pass.set_bind_group(1, &self.smoke_shader_params_bindgroup[i as usize], &[]);
                compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, dispatch_depth);
            }
        }

        {
            let player = self.player.lock().unwrap();
            let row = player.get_current_row();
            if row % 4 == 0 && row != self.last_row {
                self.beat = Instant::now();
            }
            self.last_row = row;
            let pattern = player.get_current_pattern();
            if pattern != self.last_pattern && pattern <= 9 {
                self.pattern = Instant::now();
            }
            self.last_pattern = pattern;
        }
        let beat_time = now.duration_since(self.beat).as_secs_f64();
        let pattern_time = now.duration_since(self.pattern).as_secs_f64();
        self.bg_shader_params.t = pattern_time as f32;
        self.fg_shader_params.x = -0.1 + 1.0/(pattern_time*4.0+0.2) as f32;
        self.instances[5].tex_offset.y = -0.5*pattern_time as f32;

        // self.scene = match self.last_pattern {
        //     0 => Scene::Intro,
        //     1 => Scene::Roulette(SHADER_FUNCTION_COOL),
        //     2 => Scene::Drivin(1),
        //     3 => Scene::Roulette(SHADER_FUNCTION_COOL_2),
        //     4 => Scene::Drivin(2),
        //     5 => Scene::Roulette(SHADER_FUNCTION_TRANS),
        //     6 => Scene::Drivin(3),
        //     7 => Scene::Roulette(SHADER_FUNCTION_COOL_3),
        //     8 => Scene::Drivin(4),
        //     _ => Scene::Outro
        // };

        println!("time {} fps {}", delta_time, 1.0/delta_time);

        // match self.scene {
        //     Scene::Intro => {}
        //     Scene::Roulette(x) => {
        //         self.bg_shader_params.shader_function = x;
        //         if self.vehicle.as_ref().map(|(_, start, duration)| now.duration_since(*start) > *duration).unwrap_or(true) {
        //             self.vehicle = Some(self.random_vehicle());
        //         }
        //         self.camera.eye = Point3::new(7.0 + 2.0*(2.0*time).cos() as f32, 2.0 + (6.0*time).sin() as f32, -19.0);
        //         self.camera.target = if let Some((i, _, _)) = self.vehicle { if i == 2 {(0.0, 0.0, 0.0).into()} else {(-2.0, 0.5, -2.0).into()}} else {(-2.0, 0.5, -2.0).into()};
        //         self.camera.fovy = 45.0/(1.1*beat(beat_time as f32));
        //     }
        //     Scene::Drivin(x) => {
        //         self.bg_shader_params.shader_function = match x {
        //             1 => 6,
        //             2 => 7,
        //             3 => 8,
        //             4 => 9,
        //             _ => panic!()
        //         };
        //         self.vehicle = Some((match x {1=>5, 2=>0, 3=>1, 4=>4, _ => panic!()}, Instant::now(), Duration::ZERO));
        //         self.camera.eye = Point3::new((((pattern_time/4.0).exp()).cos()*20.0) as f32, 5.0+pattern_time as f32/4.0+(pattern_time as f32*5.0).sin(), (((pattern_time/4.0).exp()).sin()*20.0) as f32);
        //         self.camera.target = (0.0, 0.0, 0.0).into();
        //         self.camera.fovy = 45.0;
        //     }
        //     Scene::Outro => {}
        // }
        // self.camera_uniform.update_view_proj(&self.camera);

        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
        queue.write_buffer(
            &self.bg_function_buffer,
            0,
            bytemuck::cast_slice(&[self.bg_shader_params]),
        );
        queue.write_buffer(
            &self.fg_function_buffer,
            0,
            bytemuck::cast_slice(&[self.fg_shader_params]),
        );
        let raw_instance = self.instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        queue.write_buffer(
            &self.instance_buffer, 0, bytemuck::cast_slice(&raw_instance));
    }

    pub fn render(
        &mut self,
        view_final: &TextureView,
        depth_view: &TextureView,
        pipeline_1: &RenderPipeline,
        pipeline_final: &RenderPipeline,
        encoder: &mut CommandEncoder) {
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass 1"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.texture_pass1.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_pipeline(pipeline_1);

            render_pass.set_vertex_buffer(0, self.full_quad_vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.full_quad_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.set_bind_group(1, &self.fg_uniform_bind_group, &[]);
            render_pass.set_bind_group(0, &self.smoke_render_bind_group, &[]);
            render_pass.draw_indexed(0..6, 0, 0..1);
/*
            match self.scene {
                Scene::Intro => {
                    let model = &self.models[DECAL_MODEL];
                    render_pass.set_vertex_buffer(0, model.meshes[0].vertex_buffer.slice(..));
                    render_pass.set_index_buffer(model.meshes[0].index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.set_bind_group(0, &self.decal_bindgroups[0], &[]);
                    render_pass.set_bind_group(1, &self.decal_uniform_bind_group, &[]);
                    render_pass.draw_indexed(0..model.meshes[0].num_elements, 0, 0..2);

                }
                Scene::Outro => {
                    let model = &self.models[DECAL_MODEL];
                    render_pass.set_vertex_buffer(0, model.meshes[0].vertex_buffer.slice(..));
                    render_pass.set_index_buffer(model.meshes[0].index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.set_bind_group(0, &self.decal_bindgroups[1], &[]);
                    render_pass.set_bind_group(1, &self.decal_uniform_bind_group, &[]);
                    render_pass.draw_indexed(0..model.meshes[0].num_elements, 0, 0..2);

                }
                Scene::Roulette(_) => {
                    if let Some((vehicle, _, _)) = self.vehicle {
                        let model = &self.models[vehicle];
                        render_pass.draw_model_instanced(
                            &model,
                            0..1,
                            &self.uniform_bind_group,
                        );
                    }
                    
                    render_pass.set_vertex_buffer(0, self.full_quad_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(self.full_quad_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.set_bind_group(0, &self.models[0].materials[0].bind_group, &[]);
                    render_pass.set_bind_group(1, &self.bg_uniform_bind_group, &[]);
                    render_pass.draw_indexed(0..6, 0, 0..1);

                    if self.last_pattern == 7 {
                        let decal = &self.models[DECAL_MODEL];
                        render_pass.set_vertex_buffer(0, decal.meshes[0].vertex_buffer.slice(..));
                        render_pass.set_index_buffer(decal.meshes[0].index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        render_pass.set_bind_group(0, &self.decal_bindgroups[3], &[]);
                        render_pass.set_bind_group(1, &self.decal_uniform_bind_group, &[]);
                        render_pass.draw_indexed(0..decal.meshes[0].num_elements, 0, row_to_range(self.last_row));
                    }
                }
                Scene::Drivin(x) => {
                    if let Some((vehicle, _, _)) = self.vehicle {
                        let model = &self.models[vehicle];
                        render_pass.draw_model_instanced(
                            &model,
                            0..1,
                            &self.uniform_bind_group,
                        );
                    }

                    let ground = &self.models[GROUND_MODEL];
                    render_pass.set_vertex_buffer(0, ground.meshes[0].vertex_buffer.slice(..));
                    render_pass.set_index_buffer(ground.meshes[0].index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.set_bind_group(0, &self.ground_bindgroups[(x-1) as usize], &[]);
                    render_pass.set_bind_group(1, &self.uniform_bind_group, &[]);
                    render_pass.draw_indexed(0..ground.meshes[0].num_elements, 0, 5..6);
                    //render_pass.draw_model_instanced(&self.models[GROUND_MODEL], 5..6, uniform_bind_group);

                    render_pass.set_vertex_buffer(0, self.full_quad_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(self.full_quad_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.set_bind_group(0, &self.models[0].materials[0].bind_group, &[]);
                    render_pass.set_bind_group(1, &self.bg_uniform_bind_group, &[]);
                    render_pass.draw_indexed(0..6, 0, 0..1);

                    if self.last_pattern == 6 || self.last_pattern == 8 {
                        let decal = &self.models[DECAL_MODEL];
                        render_pass.set_vertex_buffer(0, decal.meshes[0].vertex_buffer.slice(..));
                        render_pass.set_index_buffer(decal.meshes[0].index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        render_pass.set_bind_group(0, &self.decal_bindgroups[match self.last_pattern{6=>2,8=>4,_=>panic!()}], &[]);
                        render_pass.set_bind_group(1, &self.decal_uniform_bind_group, &[]);
                        render_pass.draw_indexed(0..decal.meshes[0].num_elements, 0, row_to_range(self.last_row));
                    }
                }
            }

            render_pass.set_vertex_buffer(0, self.full_quad_vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.full_quad_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.set_bind_group(0, &self.models[0].materials[0].bind_group, &[]);

            render_pass.set_bind_group(1, &self.fg_uniform_bind_group, &[]);
            render_pass.draw_indexed(0..6, 0, 0..1);
*/
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass final"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_final,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_pipeline(pipeline_final);

            render_pass.set_vertex_buffer(0, self.full_quad_vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.full_quad_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.set_bind_group(1, &self.fg_uniform_bind_group, &[]);
            render_pass.set_bind_group(0, &self.final_pass_uniform_bind_group, &[]);
            render_pass.draw_indexed(0..6, 0, 0..1);
        }

    }

    pub fn resize(&mut self, device: &wgpu::Device, texture_pass1: texture::Texture) {
        self.texture_pass1 = texture_pass1;
        self.final_pass_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.texture_pass1.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.texture_pass1.sampler),
                    },
                ],
                label: None,
            });
    }
}

enum Scene {
    Intro,
    Roulette(i32),
    Drivin(i32),
    Outro
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
    pub delta_time: f32
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderParamsUniform {
    pub shader_function: i32,
    pub t: f32,
    pub x: f32
}

impl ShaderParamsUniform {
    pub fn new() -> Self {
        Self {
            shader_function: 0,
            t: 0.0,
            x: 0.0
        }
    }
}

fn start_audio_player(player: Arc<Mutex<XmrsPlayer>>) -> Result<(), cpal::StreamError> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("no output device available");

    let config = device
        .default_output_config()
        .expect("failed to get default output config");

    std::thread::spawn(move || {
        let stream = device
            .build_output_stream(
                &config.config(),
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