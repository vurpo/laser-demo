use std::io::{BufReader, Cursor};

use wgpu::util::DeviceExt;
use include_dir::{include_dir, Dir};

use crate::model::{self, ModelVertex};

pub static ASSETS: Dir = include_dir!("$CARGO_MANIFEST_DIR/assets");

pub const QUAD_VERTICES: &[ModelVertex] = &[
    model::ModelVertex {position: [-1.0,-1.0, 0.], tex_coords: [-1.0,-1.0], normal: [0.0,0.0,1.0]},
    model::ModelVertex {position: [ 1.0,-1.0, 0.], tex_coords: [ 1.0,-1.0], normal: [0.0,0.0,1.0]},
    model::ModelVertex {position: [-1.0, 1.0, 0.], tex_coords: [-1.0, 1.0], normal: [0.0,0.0,1.0]},
    model::ModelVertex {position: [ 1.0, 1.0, 0.], tex_coords: [ 1.0, 1.0], normal: [0.0,0.0,1.0]}
];
pub const QUAD_INDICES: &[u32] = &[0,1,2,1,3,2];

pub async fn load_string(file_name: &str) -> anyhow::Result<String> {
    // let path = std::path::Path::new("assets").join(file_name);
    // let txt = std::fs::read_to_string(path)?;
    let file = ASSETS.get_file(file_name).ok_or(anyhow::anyhow!("no such file"))?;
    return file.contents_utf8().ok_or(anyhow::anyhow!("file is not utf-8")).map(str::to_owned);
}

// pub async fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
//     match file_name {
//         // "m100.png" => { return Ok(Vec::from(M100_PNG)) },
//         // "edo.png" => { return Ok(Vec::from(EDO_PNG)) },
//         // "boing.png" => { return Ok(Vec::from(BOING_PNG)) },
//         // "boat.png" => { return Ok(Vec::from(BOAT_PNG)) },
//         // "tgv.png" => { return Ok(Vec::from(TGV_PNG)) },
//         // "manse.png" => { return Ok(Vec::from(MANSE_PNG)) },
//         // "grass1.png" => { return Ok(Vec::from(GRASS1_PNG)) },
//         // "grass2.png" => { return Ok(Vec::from(GRASS2_PNG)) },
//         // "grass3.png" => { return Ok(Vec::from(GRASS3_PNG)) },
//         // "grass4.png" => { return Ok(Vec::from(GRASS4_PNG)) },
//         // "decaltest.png" => { return Ok(Vec::from(DECAL_PNG)) },
//         // "decal1.png" => { return Ok(Vec::from(DECAL1_PNG)) },
//         // "decal2.png" => { return Ok(Vec::from(DECAL2_PNG)) },
//         // "decal3.png" => { return Ok(Vec::from(DECAL3_PNG)) },
//         // "decal4.png" => { return Ok(Vec::from(DECAL4_PNG)) },
//         // "decal5.png" => { return Ok(Vec::from(DECAL5_PNG)) },
//         _ => ()
//     }

//     let path = std::path::Path::new("assets").join(file_name);
//     println!("{:?}", path);
//     let data = std::fs::read(path)?;

//     Ok(data)
// }

// pub async fn load_texture(
//     file_name: &str,
//     device: &wgpu::Device,
//     queue: &wgpu::Queue,
// ) -> anyhow::Result<texture::Texture> {
//     let data = load_binary(file_name).await?;
//     texture::Texture::from_bytes(device, queue, &data, file_name)
// }

pub async fn load_model(
    file_name: &str,
    device: &wgpu::Device,
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
    // for m in obj_materials? {
    //     if m.name == "none" { continue; } // HAX
    //     let diffuse_texture = load_texture(&m.diffuse_texture, device, queue).await?;
    //     let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
    //         layout,
    //         entries: &[
    //             wgpu::BindGroupEntry {
    //                 binding: 0,
    //                 resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
    //             },
    //             wgpu::BindGroupEntry {
    //                 binding: 1,
    //                 resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
    //             },
    //         ],
    //         label: None,
    //     });

    //     materials.push(model::Material {
    //         name: m.name,
    //         diffuse_texture,
    //         bind_group,
    //     })
    // }

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
