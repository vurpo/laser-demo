use std::io::{BufReader, Cursor};

use wgpu::util::DeviceExt;

use crate::{model::{self, ModelVertex}, texture};

// OBSERVE!
// The bleeding edge of resources management!

const M100_OBJ: &'static [u8] = include_bytes!("../assets/m100.obj");
const M100_MTL: &'static [u8] = include_bytes!("../assets/m100.mtl");
const M100_PNG: &'static [u8] = include_bytes!("../assets/m100.png");
const EDO_OBJ: &'static [u8] = include_bytes!("../assets/edo.obj");
const EDO_MTL: &'static [u8] = include_bytes!("../assets/edo.mtl");
const EDO_PNG: &'static [u8] = include_bytes!("../assets/edo.png");
const BOING_OBJ: &'static [u8] = include_bytes!("../assets/boing.obj");
const BOING_MTL: &'static [u8] = include_bytes!("../assets/boing.mtl");
const BOING_PNG: &'static [u8] = include_bytes!("../assets/boing.png");
const BOAT_OBJ: &'static [u8] = include_bytes!("../assets/boat.obj");
const BOAT_MTL: &'static [u8] = include_bytes!("../assets/boat.mtl");
const BOAT_PNG: &'static [u8] = include_bytes!("../assets/boat.png");
const TGV_OBJ: &'static [u8] = include_bytes!("../assets/tgv.obj");
const TGV_MTL: &'static [u8] = include_bytes!("../assets/tgv.mtl");
const TGV_PNG: &'static [u8] = include_bytes!("../assets/tgv.png");
const MANSE_OBJ: &'static [u8] = include_bytes!("../assets/manse.obj");
const MANSE_MTL: &'static [u8] = include_bytes!("../assets/manse.mtl");
const MANSE_PNG: &'static [u8] = include_bytes!("../assets/manse.png");
const FLAT_OBJ: &'static [u8] = include_bytes!("../assets/flat.obj");
const FLAT_MTL: &'static [u8] = include_bytes!("../assets/flat.mtl");
const GRASS1_PNG: &'static [u8] = include_bytes!("../assets/grass1.png");
const GRASS2_PNG: &'static [u8] = include_bytes!("../assets/grass2.png");
const GRASS3_PNG: &'static [u8] = include_bytes!("../assets/grass3.png");
const GRASS4_PNG: &'static [u8] = include_bytes!("../assets/grass4.png");
const DECAL_OBJ: &'static [u8] = include_bytes!("../assets/decal.obj");
const DECAL_MTL: &'static [u8] = include_bytes!("../assets/decal.mtl");
const DECAL_PNG: &'static [u8] = include_bytes!("../assets/decaltest.png");

const DECAL1_PNG: &'static [u8] = include_bytes!("../assets/decal1.png");
const DECAL2_PNG: &'static [u8] = include_bytes!("../assets/decal2.png");
const DECAL3_PNG: &'static [u8] = include_bytes!("../assets/decal3.png");
const DECAL4_PNG: &'static [u8] = include_bytes!("../assets/decal4.png");
const DECAL5_PNG: &'static [u8] = include_bytes!("../assets/decal5.png");

pub const MUSIC: &'static [u8] = include_bytes!("../music/take a train.xm");

pub const FULL_SCREEN_VERTICES: &[ModelVertex] = &[
    model::ModelVertex {position: [-1.0,-1.0, 0.999], tex_coords: [-1.0,-1.0], normal: [0.0,0.0,1.0]},
    model::ModelVertex {position: [ 1.0,-1.0, 0.999], tex_coords: [ 1.0,-1.0], normal: [0.0,0.0,1.0]},
    model::ModelVertex {position: [-1.0, 1.0, 0.999], tex_coords: [-1.0, 1.0], normal: [0.0,0.0,1.0]},
    model::ModelVertex {position: [ 1.0, 1.0, 0.999], tex_coords: [ 1.0, 1.0], normal: [0.0,0.0,1.0]}
];
pub const FULL_SCREEN_INDICES: &[u32] = &[0,1,2,1,3,2];

pub async fn load_string(file_name: &str) -> anyhow::Result<String> {
    match file_name {
        "m100.obj" => { return Ok(std::str::from_utf8(M100_OBJ).unwrap().to_owned()) },
        "m100.mtl" => { return Ok(std::str::from_utf8(M100_MTL).unwrap().to_owned()) },
        "edo.obj" => { return Ok(std::str::from_utf8(EDO_OBJ).unwrap().to_owned()) },
        "edo.mtl" => { return Ok(std::str::from_utf8(EDO_MTL).unwrap().to_owned()) },
        "boing.obj" => { return Ok(std::str::from_utf8(BOING_OBJ).unwrap().to_owned()) },
        "boing.mtl" => { return Ok(std::str::from_utf8(BOING_MTL).unwrap().to_owned()) },
        "boat.obj" => { return Ok(std::str::from_utf8(BOAT_OBJ).unwrap().to_owned()) },
        "boat.mtl" => { return Ok(std::str::from_utf8(BOAT_MTL).unwrap().to_owned()) },
        "tgv.obj" => { return Ok(std::str::from_utf8(TGV_OBJ).unwrap().to_owned()) },
        "tgv.mtl" => { return Ok(std::str::from_utf8(TGV_MTL).unwrap().to_owned()) },
        "manse.obj" => { return Ok(std::str::from_utf8(MANSE_OBJ).unwrap().to_owned()) },
        "manse.mtl" => { return Ok(std::str::from_utf8(MANSE_MTL).unwrap().to_owned()) },
        "flat.obj" => { return Ok(std::str::from_utf8(FLAT_OBJ).unwrap().to_owned()) },
        "flat.mtl" => { return Ok(std::str::from_utf8(FLAT_MTL).unwrap().to_owned()) },
        "decal.obj" => { return Ok(std::str::from_utf8(DECAL_OBJ).unwrap().to_owned()) },
        "decal.mtl" => { return Ok(std::str::from_utf8(DECAL_MTL).unwrap().to_owned()) },
        _ => ()
    }
    
    let path = std::path::Path::new("assets").join(file_name);
    let txt = std::fs::read_to_string(path)?;

    Ok(txt)
}

pub async fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
    match file_name {
        "m100.png" => { return Ok(Vec::from(M100_PNG)) },
        "edo.png" => { return Ok(Vec::from(EDO_PNG)) },
        "boing.png" => { return Ok(Vec::from(BOING_PNG)) },
        "boat.png" => { return Ok(Vec::from(BOAT_PNG)) },
        "tgv.png" => { return Ok(Vec::from(TGV_PNG)) },
        "manse.png" => { return Ok(Vec::from(MANSE_PNG)) },
        "grass1.png" => { return Ok(Vec::from(GRASS1_PNG)) },
        "grass2.png" => { return Ok(Vec::from(GRASS2_PNG)) },
        "grass3.png" => { return Ok(Vec::from(GRASS3_PNG)) },
        "grass4.png" => { return Ok(Vec::from(GRASS4_PNG)) },
        "decaltest.png" => { return Ok(Vec::from(DECAL_PNG)) },
        "decal1.png" => { return Ok(Vec::from(DECAL1_PNG)) },
        "decal2.png" => { return Ok(Vec::from(DECAL2_PNG)) },
        "decal3.png" => { return Ok(Vec::from(DECAL3_PNG)) },
        "decal4.png" => { return Ok(Vec::from(DECAL4_PNG)) },
        "decal5.png" => { return Ok(Vec::from(DECAL5_PNG)) },
        _ => ()
    }

    let path = std::path::Path::new("assets").join(file_name);
    println!("{:?}", path);
    let data = std::fs::read(path)?;

    Ok(data)
}

pub async fn load_texture(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<texture::Texture> {
    let data = load_binary(file_name).await?;
    texture::Texture::from_bytes(device, queue, &data, file_name)
}

pub async fn load_model(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    scale: f32
) -> anyhow::Result<model::Model> {
    let obj_text = load_string(file_name).await?;
    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);

    let (models, obj_materials) = tobj::load_obj_buf_async(
        &mut obj_reader,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |p| async move {
            let mat_text = load_string(&p).await.unwrap();
            tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
        },
    )
    .await?;

    let mut materials = Vec::new();
    for m in obj_materials? {
        if m.name == "none" { continue; } // HAX
        let diffuse_texture = load_texture(&m.diffuse_texture, device, queue).await?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: None,
        });

        materials.push(model::Material {
            name: m.name,
            diffuse_texture,
            bind_group,
        })
    }

    let meshes = models
        .into_iter()
        .map(|m| {
            let vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| model::ModelVertex {
                    position: [
                        m.mesh.positions[i * 3]*scale,
                        m.mesh.positions[i * 3 + 1]*scale,
                        m.mesh.positions[i * 3 + 2]*scale,
                    ],
                    tex_coords: [m.mesh.texcoords[i * 2], 1.0-m.mesh.texcoords[i * 2 + 1]],
                    normal: [
                        m.mesh.normals[i * 3],
                        m.mesh.normals[i * 3 + 1],
                        m.mesh.normals[i * 3 + 2],
                    ],
                })
                .collect::<Vec<_>>();

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Vertex Buffer", file_name)),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Index Buffer", file_name)),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            model::Mesh {
                name: file_name.to_string(),
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len() as u32,
                material: m.mesh.material_id.unwrap_or(0),
            }
        })
        .collect::<Vec<_>>();

    Ok(model::Model { meshes, materials })
}