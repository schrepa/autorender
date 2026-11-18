use engine_assets::{
    AttributeFormat, Mesh, MeshStreams, MeshVertex, NamedStream, encode_mesh_buffer_with,
};
use glam::{Mat3, Mat4, Vec3};
use gltf::mesh::Mode;
use serde_json::{Value, json};

use crate::fault::Fault;
use crate::rig::Rig;
use crate::source::{Source, node_label};

pub struct Built {
    pub stem: String,
    pub bytes: Vec<u8>,
    pub material: usize,
    pub vertices: usize,
    pub triangles: usize,
    pub influences: usize,
}

struct Part {
    material: Option<usize>,
    vertices: Vec<MeshVertex>,
    indices: Vec<u32>,
    uv1: Option<Vec<[f32; 2]>>,
    skin: Option<(Vec<u8>, Vec<u8>)>,
    influences: usize,
    normals: &'static str,
    uv: &'static str,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Reading {
    Drawn,
    Shape,
}

struct Skin<'a> {
    rig: &'a Rig,
    joints: Vec<[u16; 4]>,
    weights: Vec<[f32; 4]>,
}

fn skin_bytes(
    source: &Source,
    item: &str,
    skin: &Skin<'_>,
) -> Result<(Vec<u8>, Vec<u8>, usize), Fault> {
    let mut joint_bytes = Vec::with_capacity(skin.joints.len() * 16);
    let mut weight_bytes = Vec::with_capacity(skin.joints.len() * 8);
    let mut influences = 0;
    for (vertex, (joints, weights)) in skin.joints.iter().zip(&skin.weights).enumerate() {
        let total: f32 = weights.iter().sum();
        if !total.is_finite() || total <= 0.0 || weights.iter().any(|weight| *weight < 0.0) {
            return Err(source.fault(
                "Weights",
                format!("{item} vertex {vertex}"),
                format!("{weights:?}"),
            ));
        }
        let mut quantised = weights.map(|weight| (weight / total * 65535.0).round() as u32);
        let sum: u32 = quantised.iter().sum();
        let largest = (0..4)
            .max_by_key(|&lane| quantised[lane])
            .expect("four_lanes");
        quantised[largest] = (quantised[largest] + 65535).saturating_sub(sum);
        influences = influences.max(quantised.iter().filter(|&&weight| weight > 0).count());
        for (joint, weight) in joints.iter().zip(quantised) {
            let mapped = skin.rig.of_skin.get(usize::from(*joint)).ok_or_else(|| {
                source.fault(
                    "JointIndex",
                    format!("{item} vertex {vertex}"),
                    format!("joint {joint} of {}", skin.rig.of_skin.len()),
                )
            })?;
            joint_bytes.extend_from_slice(&mapped.to_le_bytes());
            weight_bytes.extend_from_slice(
                &u16::try_from(weight)
                    .expect("weight_fits_u16")
                    .to_le_bytes(),
            );
        }
    }
    Ok((joint_bytes, weight_bytes, influences))
}

fn computed_normals(positions: &[Vec3], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut sums = vec![Vec3::ZERO; positions.len()];
    for triangle in indices.as_chunks::<3>().0 {
        let [a, b, c] = triangle.map(|index| index as usize);
        if a.max(b).max(c) >= positions.len() {
            continue;
        }
        let face = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
        for corner in [a, b, c] {
            sums[corner] += face;
        }
    }
    sums.into_iter()
        .map(|sum| sum.normalize_or_zero().to_array())
        .collect()
}

fn read_skin<'a, 's, F>(
    source: &Source,
    item: &str,
    reader: &gltf::mesh::Reader<'a, 's, F>,
    rig: &Rig,
    count: usize,
) -> Result<(Vec<u8>, Vec<u8>, usize), Fault>
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    let missing = |name: &str| source.fault("MissingAttribute", item, name);
    if reader.read_joints(1).is_some() || reader.read_weights(1).is_some() {
        return Err(source.fault(
            "InfluenceSets",
            item,
            "JOINTS_1 or WEIGHTS_1: more than 4 influences",
        ));
    }
    let skin = Skin {
        rig,
        joints: reader
            .read_joints(0)
            .ok_or_else(|| missing("JOINTS_0"))?
            .into_u16()
            .collect(),
        weights: reader
            .read_weights(0)
            .ok_or_else(|| missing("WEIGHTS_0"))?
            .into_f32()
            .collect(),
    };
    if skin.joints.len() != count || skin.weights.len() != count {
        return Err(source.fault(
            "AttributeCount",
            item,
            "skin attributes differ in vertex count",
        ));
    }
    skin_bytes(source, item, &skin)
}

fn read_part(
    source: &Source,
    item: &str,
    held: &gltf::Primitive<'_>,
    place: Mat4,
    rig: Option<&Rig>,
    reading: Reading,
) -> Result<Part, Fault> {
    if held.mode() != Mode::Triangles {
        return Err(source.fault("PrimitiveMode", item, format!("{:?}", held.mode())));
    }
    if held.morph_targets().next().is_some() {
        return Err(source.fault("MorphTargets", item, "morph targets are refused"));
    }
    let material = held.material().index();
    if material.is_none() && reading == Reading::Drawn {
        return Err(source.fault("NoMaterial", item, "a primitive needs a material with maps"));
    }
    let reader = held.reader(|buffer| source.data(&buffer));
    let missing = |name: &str| source.fault("MissingAttribute", item, name);
    let positions: Vec<Vec3> = reader
        .read_positions()
        .ok_or_else(|| missing("POSITION"))?
        .map(|position| place.transform_point3(Vec3::from_array(position)))
        .collect();
    let count = positions.len();
    let indices: Vec<u32> = match reader.read_indices() {
        Some(indices) => indices.into_u32().collect(),
        None => (0..u32::try_from(count).expect("vertex_count_fits_u32")).collect(),
    };
    if indices.is_empty() || !indices.len().is_multiple_of(3) {
        return Err(source.fault("Indices", item, format!("{} indices", indices.len())));
    }
    let turn = Mat3::from_mat4(place).inverse().transpose();
    let (normals, normals_from): (Vec<[f32; 3]>, &'static str) =
        match (reader.read_normals(), reading) {
            (Some(normals), _) => (
                normals
                    .map(|normal| {
                        (turn * Vec3::from_array(normal))
                            .normalize_or_zero()
                            .to_array()
                    })
                    .collect(),
                "source",
            ),
            (None, Reading::Shape) => (computed_normals(&positions, &indices), "computed"),
            (None, Reading::Drawn) => return Err(missing("NORMAL")),
        };
    let (uvs, uv_from): (Vec<[f32; 2]>, &'static str) = match (reader.read_tex_coords(0), reading) {
        (Some(uvs), _) => (uvs.into_f32().collect(), "source"),
        (None, Reading::Shape) => (vec![[0.0; 2]; count], "zero"),
        (None, Reading::Drawn) => return Err(missing("TEXCOORD_0")),
    };
    let uv1: Option<Vec<[f32; 2]>> = reader
        .read_tex_coords(1)
        .map(|uvs| uvs.into_f32().collect());
    if normals.len() != count
        || uvs.len() != count
        || uv1.as_ref().is_some_and(|uv1| uv1.len() != count)
    {
        return Err(source.fault("AttributeCount", item, "attributes differ in vertex count"));
    }
    let (skin, influences) = match rig {
        Some(rig) => {
            let (joints, weights, influences) = read_skin(source, item, &reader, rig, count)?;
            (Some((joints, weights)), influences)
        }
        None => (None, 0),
    };
    let vertices = positions
        .iter()
        .zip(&normals)
        .zip(&uvs)
        .map(|((position, normal), uv)| MeshVertex {
            position: position.to_array(),
            normal: *normal,
            uv: *uv,
        })
        .collect();
    Ok(Part {
        material,
        vertices,
        indices,
        uv1: if reading == Reading::Drawn { uv1 } else { None },
        skin,
        influences,
        normals: normals_from,
        uv: uv_from,
    })
}

fn read_back(
    source: &Source,
    item: &str,
    bytes: &[u8],
    vertices: usize,
    indices: usize,
    streams: usize,
    uv1: bool,
) -> Result<(), Fault> {
    let mesh =
        Mesh::from_record(bytes).map_err(|error| source.fault("MeshReadBack", item, error))?;
    if mesh.vertices().len() != vertices
        || mesh.indices().len() != indices
        || mesh.streams().len() != streams
        || mesh.uv1().is_some() != uv1
    {
        return Err(source.fault(
            "MeshReadBack",
            item,
            format!(
                "read {} vertices, {} indices, {} streams, uv1 {}",
                mesh.vertices().len(),
                mesh.indices().len(),
                mesh.streams().len(),
                mesh.uv1().is_some()
            ),
        ));
    }
    Ok(())
}

fn encode(source: &Source, item: &str, parts: &[Part]) -> Result<Built, Fault> {
    let with_uv1 = parts.iter().filter(|part| part.uv1.is_some()).count();
    let with_skin = parts.iter().filter(|part| part.skin.is_some()).count();
    if (with_uv1 != 0 && with_uv1 != parts.len()) || (with_skin != 0 && with_skin != parts.len()) {
        return Err(source.fault(
            "MixedAttributes",
            item,
            "merged primitives differ in TEXCOORD_1 or skin",
        ));
    }
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut uv1: Vec<[f32; 2]> = Vec::new();
    let mut joints = Vec::new();
    let mut weights = Vec::new();
    for part in parts {
        let base = u32::try_from(vertices.len()).expect("vertex_count_fits_u32");
        indices.extend(part.indices.iter().map(|index| index + base));
        vertices.extend_from_slice(&part.vertices);
        uv1.extend(part.uv1.iter().flatten().copied());
        if let Some((joint_bytes, weight_bytes)) = &part.skin {
            joints.extend_from_slice(joint_bytes);
            weights.extend_from_slice(weight_bytes);
        }
    }
    let mut named = Vec::new();
    if with_skin != 0 {
        named.push(NamedStream::new(
            "joints",
            AttributeFormat::Uint32x4,
            joints,
        ));
        named.push(NamedStream::new(
            "weights",
            AttributeFormat::Unorm16x4,
            weights,
        ));
    }
    let bytes = encode_mesh_buffer_with(
        &vertices,
        &indices,
        MeshStreams {
            colors: None,
            uv1: (with_uv1 != 0).then_some(uv1.as_slice()),
            named: &named,
            lods: None,
        },
    );
    read_back(
        source,
        item,
        &bytes,
        vertices.len(),
        indices.len(),
        named.len(),
        with_uv1 != 0,
    )?;
    Ok(Built {
        stem: String::new(),
        bytes,
        material: parts[0].material.unwrap_or_default(),
        vertices: vertices.len(),
        triangles: indices.len() / 3,
        influences: parts
            .iter()
            .map(|part| part.influences)
            .max()
            .unwrap_or_default(),
    })
}

fn named(name: &str, mut built: Vec<Built>) -> Vec<Built> {
    let single = built.len() == 1;
    for (index, mesh) in built.iter_mut().enumerate() {
        mesh.stem = if single {
            name.to_owned()
        } else {
            format!("{name}_{index}")
        };
    }
    built
}

fn primitive_item(node: &gltf::Node<'_>, held: &gltf::Primitive<'_>) -> String {
    format!("{} primitive {}", node_label(node), held.index())
}

pub fn character(source: &Source, rig: &Rig, name: &str) -> Result<Vec<Built>, Fault> {
    let place = Mat4::from_scale(Vec3::splat(rig.scale)) * rig.space.inverse();
    let mut built = Vec::new();
    for node in source.gltf.nodes() {
        let Some(mesh) = node.mesh() else { continue };
        if node.skin().is_none() {
            return Err(source.fault(
                "UnskinnedMesh",
                node_label(&node),
                "a character file holds only skinned meshes",
            ));
        }
        for held in mesh.primitives() {
            let item = primitive_item(&node, &held);
            let part = read_part(source, &item, &held, place, Some(rig), Reading::Drawn)?;
            built.push(encode(source, &item, &[part])?);
        }
    }
    if built.is_empty() {
        return Err(source.fault("NoMesh", "nodes", "no mesh to cook"));
    }
    Ok(named(name, built))
}

fn rigid(place: Mat4, tolerance: f32) -> bool {
    let (scale, _, _) = place.to_scale_rotation_translation();
    place.determinant() > 0.0
        && (scale - Vec3::splat(scale.x)).abs().max_element() <= tolerance * scale.x
}

fn placed(
    source: &Source,
    node: &gltf::Node<'_>,
    scale: f32,
    tolerance: f32,
) -> Result<Mat4, Fault> {
    if node.skin().is_some() {
        return Err(source.fault(
            "SkinInStatic",
            node_label(node),
            "a static or map entry takes no skin",
        ));
    }
    let world = source.world(node.index());
    if !rigid(world, tolerance) {
        return Err(source.fault(
            "NodeScale",
            node_label(node),
            "negative or non-uniform scale",
        ));
    }
    Ok(Mat4::from_scale(Vec3::splat(scale)) * world)
}

pub fn fixed(source: &Source, name: &str, scale: f32, tolerance: f32) -> Result<Vec<Built>, Fault> {
    let mut built = Vec::new();
    for node in source.gltf.nodes() {
        let Some(mesh) = node.mesh() else { continue };
        let place = placed(source, &node, scale, tolerance)?;
        for held in mesh.primitives() {
            let item = primitive_item(&node, &held);
            let part = read_part(source, &item, &held, place, None, Reading::Drawn)?;
            built.push(encode(source, &item, &[part])?);
        }
    }
    if built.is_empty() {
        return Err(source.fault("NoMesh", "nodes", "no mesh to cook"));
    }
    Ok(named(name, built))
}

pub struct Merged {
    pub output: String,
    pub built: Built,
    pub meshes: Vec<Value>,
}

pub struct Drawn {
    pub built: Vec<Built>,
    pub widest_object: (String, usize),
}

pub fn map_drawn(
    source: &Source,
    name: &str,
    scale: f32,
    tolerance: f32,
    merged: &[&str],
) -> Result<Drawn, Fault> {
    let mut parts: Vec<Part> = Vec::new();
    let mut widest = (String::new(), 0);
    for node in source.gltf.nodes() {
        let Some(mesh) = node.mesh() else { continue };
        if merged
            .iter()
            .any(|prefix| node.name().is_some_and(|held| held.starts_with(prefix)))
        {
            continue;
        }
        let place = placed(source, &node, scale, tolerance)?;
        let mut object = 0;
        for held in mesh.primitives() {
            let part = read_part(
                source,
                &primitive_item(&node, &held),
                &held,
                place,
                None,
                Reading::Drawn,
            )?;
            object += part.vertices.len();
            parts.push(part);
        }
        if object > widest.1 {
            widest = (node_label(&node), object);
        }
    }
    let mut materials: Vec<usize> = parts.iter().filter_map(|part| part.material).collect();
    materials.sort_unstable();
    materials.dedup();
    let mut built = Vec::new();
    for &material in &materials {
        let group: Vec<Part> = parts
            .extract_if(.., |part| part.material == Some(material))
            .collect();
        let mut mesh = encode(source, &format!("materials[{material}] group"), &group)?;
        mesh.stem = if materials.len() == 1 {
            name.to_owned()
        } else {
            format!("{name}_{material}")
        };
        built.push(mesh);
    }
    if built.is_empty() {
        return Err(source.fault("NoMesh", "nodes", "no drawn mesh to cook"));
    }
    Ok(Drawn {
        built,
        widest_object: widest,
    })
}

pub fn map_merged(
    source: &Source,
    prefix: &str,
    output: &str,
    scale: f32,
    tolerance: f32,
) -> Result<Merged, Fault> {
    let mut parts = Vec::new();
    let mut meshes = Vec::new();
    let (mut indices, mut vertices) = (0, 0);
    for node in source.gltf.nodes() {
        let Some(mesh) = node.mesh() else { continue };
        let Some(node_name) = node.name().filter(|held| held.starts_with(prefix)) else {
            continue;
        };
        let place = placed(source, &node, scale, tolerance)?;
        let (first_index, first_vertex) = (indices, vertices);
        let mut sources = (Vec::new(), Vec::new());
        for held in mesh.primitives() {
            let part = read_part(
                source,
                &primitive_item(&node, &held),
                &held,
                place,
                None,
                Reading::Shape,
            )?;
            indices += part.indices.len();
            vertices += part.vertices.len();
            sources.0.push(part.normals);
            sources.1.push(part.uv);
            parts.push(part);
        }
        let said = |held: &[&'static str]| {
            if held.iter().all(|from| *from == held[0]) {
                held[0]
            } else {
                "mixed"
            }
        };
        meshes.push(json!({
            "node": node_name,
            "mesh": mesh.name(),
            "indices": [first_index, indices],
            "triangles": [first_index / 3, indices / 3],
            "vertices": [first_vertex, vertices],
            "normals": said(&sources.0),
            "uv": said(&sources.1),
            "extras": source.extras("nodes", node.index()),
            "mesh_extras": source.extras("meshes", mesh.index()),
        }));
    }
    if parts.is_empty() {
        return Err(source.fault(
            "MergeEmpty",
            format!("merge prefix `{prefix}`"),
            "no mesh node name starts with it",
        ));
    }
    let mut built = encode(source, &format!("merge `{output}`"), &parts)?;
    built.stem = output.to_owned();
    Ok(Merged {
        output: output.to_owned(),
        built,
        meshes,
    })
}
