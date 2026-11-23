use std::path::Path;

use cook::Refusal;
use engine_math::{clamp, max};
use engine_meshlet::SourceMesh;
use glam::{Mat3, Mat4, Quat, Vec2, Vec3};
use gltf::Gltf;
use gltf::buffer::Source as BufferSource;
use gltf::mesh::Mode;

fn node_local(node: &gltf::Node<'_>) -> Mat4 {
    let (translation, rotation, scale) = node.transform().decomposed();
    Mat4::from_scale_rotation_translation(
        Vec3::from_array(scale),
        Quat::from_array(rotation),
        Vec3::from_array(translation),
    )
}

fn resolve_buffers(path: &Path, gltf: &Gltf) -> Result<Vec<Vec<u8>>, Refusal> {
    let file = path.display().to_string();
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut buffers = Vec::with_capacity(gltf.buffers().count());
    for buffer in gltf.buffers() {
        match buffer.source() {
            BufferSource::Bin => {
                let blob = gltf.blob.as_ref().ok_or_else(|| {
                    Refusal::new("NoBinaryChunk", &file, "GLB binary chunk is missing")
                })?;
                buffers.push(blob.clone());
            }
            BufferSource::Uri(uri) => {
                if uri.starts_with("data:") {
                    return Err(Refusal::new(
                        "UnsupportedUri",
                        &file,
                        "data URI buffers are not supported; use .glb or external .bin",
                    ));
                }
                let bin_path = dir.join(uri);
                let data = std::fs::read(&bin_path).map_err(|error| {
                    Refusal::new(
                        "BufferRead",
                        &bin_path.display().to_string(),
                        format!("{error}"),
                    )
                })?;
                buffers.push(data);
            }
        }
    }
    Ok(buffers)
}

fn compute_smooth_normals(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let mut accum = vec![Vec3::ZERO; positions.len()];
    for &[a, b, c] in indices.as_chunks::<3>().0 {
        let (a, b, c) = (a as usize, b as usize, c as usize);
        let cross = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
        if cross.is_finite() {
            accum[a] += cross;
            accum[b] += cross;
            accum[c] += cross;
        }
    }
    accum
        .into_iter()
        .map(|n| {
            if n.length_squared() > 1.0e-18 {
                n.normalize()
            } else {
                Vec3::Y
            }
        })
        .collect()
}

pub fn compute_curvature_ao(positions: &[Vec3], normals: &[Vec3], indices: &[u32]) -> Vec<f32> {
    let n = positions.len();
    let mut sum = vec![0.0_f32; n];
    let mut weight = vec![0.0_f32; n];
    for &[i0, i1, i2] in indices.as_chunks::<3>().0 {
        let idx = [i0 as usize, i1 as usize, i2 as usize];
        for &(a, b) in &[(idx[0], idx[1]), (idx[1], idx[2]), (idx[2], idx[0])] {
            let delta = positions[b] - positions[a];
            let len = delta.length();
            if len > 1.0e-7 {
                let dir = delta / len;
                sum[a] += dir.dot(normals[a]);
                weight[a] += 1.0;
                sum[b] += (-dir).dot(normals[b]);
                weight[b] += 1.0;
            }
        }
    }
    (0..n)
        .map(|i| {
            let mean_concavity = if weight[i] > 0.0 {
                sum[i] / weight[i]
            } else {
                0.0
            };
            clamp(1.0 - max(mean_concavity, 0.0) * 1.4, 0.25, 1.0)
        })
        .collect()
}

fn clean_uv_coord(v: f32) -> f32 {
    if v.is_finite() && v.abs() <= 64.0 {
        v
    } else {
        0.0
    }
}

fn read_colors_and_ao(
    primitive: &gltf::Primitive<'_>,
    buffers: &[Vec<u8>],
    positions: &[Vec3],
    normals: &[Vec3],
    indices: &[u32],
) -> (Vec<[f32; 3]>, Vec<f32>) {
    let v_count = positions.len();
    let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(Vec::as_slice));
    if let Some(iter) = reader.read_colors(0) {
        let raw: Vec<[f32; 4]> = iter.into_rgba_f32().collect();
        if raw.len() == v_count {
            let cols = raw
                .iter()
                .map(|c| [clamp(c[0], 0.0, 1.0), clamp(c[1], 0.0, 1.0), clamp(c[2], 0.0, 1.0)])
                .collect();
            let aos = raw.iter().map(|c| clamp(c[3], 0.0, 1.0)).collect();
            return (cols, aos);
        }
    }
    let bcf = primitive.material().pbr_metallic_roughness().base_color_factor();
    let rgb = [clamp(bcf[0], 0.0, 1.0), clamp(bcf[1], 0.0, 1.0), clamp(bcf[2], 0.0, 1.0)];
    (vec![rgb; v_count], compute_curvature_ao(positions, normals, indices))
}

fn append_primitive(
    file: &str,
    primitive: &gltf::Primitive<'_>,
    buffers: &[Vec<u8>],
    world: Mat4,
    material_override: Option<u8>,
    out: &mut SourceMesh,
) -> Result<(), Refusal> {
    if primitive.material().alpha_mode() == gltf::material::AlphaMode::Blend {
        return Ok(());
    }
    if primitive.mode() != Mode::Triangles {
        return Err(Refusal::new(
            "PrimitiveMode",
            file,
            format!("unsupported mode {:?}", primitive.mode()),
        ));
    }
    let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(Vec::as_slice));
    let positions: Vec<Vec3> = reader
        .read_positions()
        .ok_or_else(|| Refusal::new("MissingAttribute", file, "POSITION"))?
        .map(|p| world.transform_point3(Vec3::from_array(p)))
        .collect();
    if positions.is_empty() {
        return Ok(());
    }
    let mut indices: Vec<u32> = match reader.read_indices() {
        Some(iter) => iter.into_u32().collect(),
        None => (0..positions.len() as u32).collect(),
    };
    if indices.len() < 3 || !indices.len().is_multiple_of(3) {
        return Err(Refusal::new(
            "InvalidIndices",
            file,
            format!("{} indices", indices.len()),
        ));
    }
    if world.determinant() < 0.0 {
        for tri in indices.as_chunks_mut::<3>().0 {
            tri.swap(1, 2);
        }
    }
    let v_count = positions.len();
    let normal_mat = Mat3::from_mat4(world).inverse().transpose();
    let normals: Vec<Vec3> = match reader.read_normals() {
        Some(iter) => {
            let raw: Vec<Vec3> = iter
                .map(|n| {
                    let t = normal_mat * Vec3::from_array(n);
                    if t.is_finite() && t.length_squared() > 1.0e-18 {
                        t.normalize()
                    } else {
                        Vec3::Y
                    }
                })
                .collect();
            if raw.len() == v_count {
                raw
            } else {
                compute_smooth_normals(&positions, &indices)
            }
        }
        None => compute_smooth_normals(&positions, &indices),
    };
    let uvs: Vec<Vec2> = match reader.read_tex_coords(0) {
        Some(iter) => {
            let raw: Vec<Vec2> = iter
                .into_f32()
                .map(|[u, v]| Vec2::new(clean_uv_coord(u), clean_uv_coord(v)))
                .collect();
            if raw.len() == v_count {
                raw
            } else {
                vec![Vec2::ZERO; v_count]
            }
        }
        None => vec![Vec2::ZERO; v_count],
    };
    let (colors, ao) = read_colors_and_ao(primitive, buffers, &positions, &normals, &indices);
    let mat_id = material_override.unwrap_or_else(|| {
        primitive
            .material()
            .index()
            .unwrap_or(0)
            .min(usize::from(u8::MAX)) as u8
    });
    let base = out.positions.len() as u32;
    let tri_count = indices.len() / 3;
    out.positions.extend_from_slice(&positions);
    out.normals.extend_from_slice(&normals);
    out.uvs.extend_from_slice(&uvs);
    out.colors.extend_from_slice(&colors);
    out.ao.extend_from_slice(&ao);
    out.indices.extend(indices.into_iter().map(|i| i + base));
    out.material_ids
        .extend(std::iter::repeat_n(mat_id, tri_count));
    if primitive.material().double_sided() && !out.double_sided_materials.contains(&mat_id) {
        out.double_sided_materials.push(mat_id);
    }
    Ok(())
}

fn walk_node(
    file: &str,
    node: &gltf::Node<'_>,
    parent_world: Mat4,
    buffers: &[Vec<u8>],
    material_override: Option<u8>,
    out: &mut SourceMesh,
) -> Result<(), Refusal> {
    let world = parent_world * node_local(node);
    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            append_primitive(file, &primitive, buffers, world, material_override, out)?;
        }
    }
    for child in node.children() {
        walk_node(file, &child, world, buffers, material_override, out)?;
    }
    Ok(())
}

pub fn load_gltf_mesh(path: &Path, material_override: Option<u8>) -> Result<SourceMesh, Refusal> {
    let file = path.display().to_string();
    let bytes = std::fs::read(path)
        .map_err(|error| Refusal::new("FileUnreadable", &file, format!("{error}")))?;
    let gltf = Gltf::from_slice(&bytes)
        .map_err(|error| Refusal::new("GltfInvalid", &file, format!("{error}")))?;
    let buffers = resolve_buffers(path, &gltf)?;
    let mut source = SourceMesh::default();
    let mut walked_scene = false;
    for scene in gltf.scenes() {
        walked_scene = true;
        for node in scene.nodes() {
            walk_node(
                &file,
                &node,
                Mat4::IDENTITY,
                &buffers,
                material_override,
                &mut source,
            )?;
        }
    }
    if !walked_scene || source.indices.is_empty() {
        source = SourceMesh::default();
        for mesh in gltf.meshes() {
            for primitive in mesh.primitives() {
                append_primitive(
                    &file,
                    &primitive,
                    &buffers,
                    Mat4::IDENTITY,
                    material_override,
                    &mut source,
                )?;
            }
        }
    }
    if source.indices.is_empty() {
        return Err(Refusal::new("EmptyMesh", &file, "no triangle primitives found"));
    }
    Ok(source)
}
