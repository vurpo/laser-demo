use std::{env, iter, sync::Arc};

use demo::Demo;
#[cfg(target_arch="wasm32")]
use web_sys::HtmlInputElement;
use winit::{
    dpi::PhysicalSize, event::*, event_loop::EventLoop, keyboard::{Key, NamedKey}, window::Window
};

#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;

mod model;
mod resources;
mod texture;
mod demo;
//mod bufferedsource;

use model::Vertex;

pub const FLUID_SIZE: (usize, usize, usize) = (100,200,100);
pub const FLUID_SCALE: f64 = 1.0;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

pub struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
    scale: cgmath::Vector3<f32>,
    tex_offset: cgmath::Vector2<f32>,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: (cgmath::Matrix4::from_translation(self.position)
                * cgmath::Matrix4::from_nonuniform_scale(self.scale.x, self.scale.y, self.scale.z)
                * cgmath::Matrix4::from(self.rotation))
            .into(),
            tex_offset: self.tex_offset.into(),
            dot: cgmath::dot(cgmath::Vector3::unit_z(), self.rotation*cgmath::Vector3::unit_z())
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    #[allow(dead_code)]
    model: [[f32; 4]; 4],
    tex_offset: [f32; 2],
    dot: f32
}

impl InstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<f32>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline_final: wgpu::RenderPipeline,
    demo: Demo,
    depth_texture: texture::Texture,
    window: Arc<Window>,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let mut size = window.inner_size();
        size.width = size.width.max(1);
        size.height = size.height.max(1);
        log::info!("Size {:?}", size);

        log::info!("WGPU setup");
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            flags: wgpu::InstanceFlags::default(),
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
            dx12_shader_compiler: Default::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });
        log::info!("Instance {:?}", instance);

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        log::info!("Adapter {:?}", adapter);
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::default(),
                    required_limits: wgpu::Limits::default(),
                },
                None, // Trace path
            )
            .await
            .unwrap();
        log::info!("{:?}", device.limits());

        let surface_caps = surface.get_capabilities(&adapter);
        log::info!("{:?}", surface_caps.formats);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![surface_format.add_srgb_suffix()],
            desired_maximum_frame_latency: 2
        };

        surface.configure(&device, &config);

        let shader_final = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaderpassfinal.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaderpassfinal.wgsl").into()),
        });

        let demo = Demo::new(&device, &queue, surface_format).await;

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        let render_pipeline_layout_final =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout final"),
                bind_group_layouts: &[&demo.texture_bind_group_layout, &demo.uniform_bind_group_layout, &demo.texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline_final = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline final"),
            layout: Some(&render_pipeline_layout_final),
            vertex: wgpu::VertexState {
                module: &shader_final,
                entry_point: "vs_main",
                buffers: &[model::ModelVertex::desc(), InstanceRaw::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_final,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format.add_srgb_suffix(),
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
                // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
                // or Features::POLYGON_MODE_POINT
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            // If the pipeline will be used with a multiview render pass, this
            // indicates how many array layers the attachments will have.
            multiview: None,
        });
        
        #[cfg(target_arch = "wasm32")]
        {
            web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| {
                    let limits = device.limits();
                    let max_size = (limits.max_buffer_size as f32/16.0).cbrt().floor() as u32;
                    let max_size = max_size.min(limits.max_texture_dimension_3d);
                    let max_label = d.get_element_by_id("max-size")?;
                    max_label.set_inner_html(&format!("(max supported by your hardware: {})", max_size.to_string()));
                    let size_slider = d.get_element_by_id("size")?.dyn_into::<HtmlInputElement>().ok()?;
                    size_slider.set_max(&max_size.to_string());
                    Some(())
                })
                .unwrap();
        }

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline_final,
            demo,
            depth_texture,
            window,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.demo.camera.aspect = self.config.width as f32 / self.config.height as f32;
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        } else {
            panic!("Resized to {:?}", new_size);
        }
    }
    fn input(&mut self, _event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {
        // let mut encoder = self
        //     .device
        //     .create_command_encoder(&wgpu::CommandEncoderDescriptor {
        //         label: Some("Compute Encoder"),
        //     });
        // self.demo.update(&self.queue, &mut encoder);
        // self.queue.submit(iter::once(encoder.finish()));
        #[cfg(target_arch = "wasm32")]
        {
            let _ = web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| {
                    let size_slider = d.get_element_by_id("size")?.dyn_into::<HtmlInputElement>().ok()?;
                    let size = size_slider.value().parse::<usize>().ok()?;
                    self.demo.resize_cube(size, &self.device);
                    Some(())
                });
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture().unwrap();
        let view_final = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor{
                format: Some(self.config.format.add_srgb_suffix()),
                ..Default::default()
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        self.demo.update(&self.queue, &mut encoder);
        self.demo.render(
            &view_final,
            &self.depth_texture.view,
            &mut self.render_pipeline_final,
            &mut encoder);

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

#[cfg_attr(target_arch="wasm32", wasm_bindgen(start))]
pub async fn run() {
    println!("{}", include_str!("../README"));

    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Info).expect("Couldn't initialize logger");
        } else {
            // if env::var("RUST_LOG").is_err() {
            //     env::set_var("RUST_LOG", "info")
            // }
            env_logger::init();
        }
    }
    log::info!("hello world!");

    let event_loop = EventLoop::new().unwrap();
    let title = "laser demo";

    let builder = winit::window::WindowBuilder::new()
        .with_title(title)
        .with_inner_size(PhysicalSize::new(1920, 1080))
        .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
    #[cfg(target_arch = "wasm32")]
    let builder = {
        use winit::platform::web::WindowBuilderExtWebSys;
        builder.with_append(true)
    };

    let window = Arc::new(builder.build(&event_loop).unwrap());
    window.set_cursor_visible(false);

    #[cfg(target_arch = "wasm32")]
    {
        // // Winit prevents sizing with CSS, so we have to set
        // // the size manually when on web.
        // use winit::dpi::PhysicalSize;
        // window.request_inner_size(PhysicalSize::new(450, 400));

        // use winit::platform::web::WindowExtWebSys;
        // web_sys::window()
        //     .and_then(|win| win.document())
        //     .and_then(|doc| {
        //         let dst = doc.get_element_by_id("wasm-example")?;
        //         let canvas = web_sys::Element::from(window.canvas().unwrap());
        //         dst.append_child(&canvas).ok()?;
        //         Some(())
        //     })
        //     .expect("Couldn't append canvas to document body.");
    }

    // State::new uses async code, so we're going to wait for it to finish
    let mut state = State::new(window.clone()).await;

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    event_loop.run(move |event, target| {
        match event {
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            event: KeyEvent {
                                logical_key: Key::Named(NamedKey::Escape),
                                ..
                            },
                            ..
                        } => target.exit(),
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::RedrawRequested => {
                            state.update();
                                match state.render() {
                                    Ok(_) => {}
                                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                        state.resize(state.size)
                                    }
                                    Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                                    Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                                }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }).unwrap();
}
