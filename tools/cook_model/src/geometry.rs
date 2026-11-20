use std::collections::BTreeMap;
use std::sync::Arc;

use cook::{Refusal, Step};
use engine_assets::MeshFile;
use engine_meshpack::{MeshVertex, Unpacked, pack, shape};
use glam::{Mat3, Mat4, Vec3};
use gltf::mesh::Mode;
use serde_json::json;

use crate::order;
use crate::rig::Rig;
use crate::source::{Source, node_label};

pub struct Built {
    pub stem: String,
    pub bytes: Vec<u8>,
    pub material: usize,
    pub vertices: usize,
    pub triangles: usize,
    pub influences: usize,
    pub colored: bool,
    pub uv1: bool,
}

pub struct Part {
    pub material: usize,
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub uv1: Option<Vec<[f32; 2]>>,
    pub colors: Option<Vec<[f32; 4]>>,
    pub skin: Option<Vec<([u32; 4], [u16; 4])>>,
    pub influences: usize,
}

pub struct Shape {
    pub positions: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

struct Skin<'a> {
    rig: &'a Rig,
    joints: Vec<[u16; 4]>,
    weights: Vec<[f32; 4]>,
}

type Influences = (Vec<([u32; 4], [u16; 4])>, usize);

fn quantised_skin(source: &Source, item: &str, skin: &Skin<'_>) -> Result<Influences, Refusal> {
    let mut held = Vec::with_capacity(skin.joints.len());
    let mut influences = 0;
    for (vertex, (joints, weights)) in skin.joints.iter().zip(&skin.weights).enumerate() {
        let total: f32 = weights.iter().sum();
        if !total.is_finite() || total <= 0.0 || weights.iter().any(|weight| *weight < 0.0) {
            return Err(source.refuse(
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
        let mut mapped = [0_u32; 4];
        for (lane, joint) in joints.iter().enumerate() {
            mapped[lane] = *skin.rig.of_skin.get(usize::from(*joint)).ok_or_else(|| {
                source.refuse(
                    "JointIndex",
                    format!("{item} vertex {vertex}"),
                    format!("joint {joint} of {}", skin.rig.of_skin.len()),
                )
            })?;
        }
        held.push((
            mapped,
            quantised.map(|weight| u16::try_from(weight).expect("weight_fits_u16")),
        ));
    }
    Ok((held, influences))
}

fn read_skin<'a, 's, F>(
    source: &Source,
    item: &str,
    reader: &gltf::mesh::Reader<'a, 's, F>,
    rig: &Rig,
    count: usize,
) -> Result<Influences, Refusal>
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    let missing = |name: &str| source.refuse("MissingAttribute", item, name);
    if reader.read_joints(1).is_some() || reader.read_weights(1).is_some() {
        return Err(source.refuse(
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
        return Err(source.refuse(
            "AttributeCount",
            item,
            "skin attributes differ in vertex count",
        ));
    }
    quantised_skin(source, item, &skin)
}

pub fn read_shape(
    source: &Source,
    item: &str,
    held: &gltf::Primitive<'_>,
    place: Mat4,
) -> Result<Shape, Refusal> {
    if held.mode() != Mode::Triangles {
        return Err(source.refuse("PrimitiveMode", item, format!("{:?}", held.mode())));
    }
    if held.morph_targets().next().is_some() {
        return Err(source.refuse("MorphTargets", item, "morph targets are refused"));
    }
    let reader = held.reader(|buffer| source.data(&buffer));
    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or_else(|| source.refuse("MissingAttribute", item, "POSITION"))?
        .map(|position| place.transform_point3(Vec3::from_array(position)).to_array())
        .collect();
    let indices: Vec<u32> = match reader.read_indices() {
        Some(indices) => indices.into_u32().collect(),
        None => (0..u32::try_from(positions.len()).expect("vertex_count_fits_u32")).collect(),
    };
    if indices.is_empty() || !indices.len().is_multiple_of(3) {
        return Err(source.refuse("Indices", item, format!("{} indices", indices.len())));
    }
    Ok(Shape { positions, indices })
}

pub fn read_part(
    source: &Source,
    item: &str,
    held: &gltf::Primitive<'_>,
    place: Mat4,
    rig: Option<&Rig>,
) -> Result<Part, Refusal> {
    let Shape { positions, indices } = read_shape(source, item, held, place)?;
    let material = held.material().index().ok_or_else(|| {
        source.refuse("NoMaterial", item, "a primitive needs a material with maps")
    })?;
    let reader = held.reader(|buffer| source.data(&buffer));
    let missing = |name: &str| source.refuse("MissingAttribute", item, name);
    let count = positions.len();
    let turn = Mat3::from_mat4(place).inverse().transpose();
    let normals: Vec<[f32; 3]> = reader
        .read_normals()
        .ok_or_else(|| missing("NORMAL"))?
        .map(|normal| {
            (turn * Vec3::from_array(normal))
                .normalize_or_zero()
                .to_array()
        })
        .collect();
    let uvs: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .ok_or_else(|| missing("TEXCOORD_0"))?
        .into_f32()
        .collect();
    let uv1: Option<Vec<[f32; 2]>> = reader
        .read_tex_coords(1)
        .map(|uvs| uvs.into_f32().collect());
    let colors: Option<Vec<[f32; 4]>> = reader
        .read_colors(0)
        .map(|colors| colors.into_rgba_f32().collect());
    if normals.len() != count
        || uvs.len() != count
        || uv1.as_ref().is_some_and(|uv1| uv1.len() != count)
        || colors.as_ref().is_some_and(|colors| colors.len() != count)
    {
        return Err(source.refuse("AttributeCount", item, "attributes differ in vertex count"));
    }
    let (skin, influences) = match rig {
        Some(rig) => {
            let (skin, influences) = read_skin(source, item, &reader, rig, count)?;
            (Some(skin), influences)
        }
        None => (None, 0),
    };
    let vertices = positions
        .into_iter()
        .zip(normals)
        .zip(uvs)
        .map(|((position, normal), uv)| MeshVertex {
            position,
            normal,
            uv,
        })
        .collect();
    Ok(Part {
        material,
        vertices,
        indices,
        uv1,
        colors,
        skin,
        influences,
    })
}

fn reordered<T: Copy>(held: usize, values: &[T], old_of: &[u32]) -> Option<Vec<T>> {
    (held != 0).then(|| order::permuted(values, old_of))
}

pub fn encode_mesh(source: &Source, item: &str, parts: &[Part]) -> Result<Built, Refusal> {
    let with_uv1 = parts.iter().filter(|part| part.uv1.is_some()).count();
    let with_skin = parts.iter().filter(|part| part.skin.is_some()).count();
    let with_colors = parts.iter().filter(|part| part.colors.is_some()).count();
    let mixed = |held: usize| held != 0 && held != parts.len();
    if mixed(with_uv1) || mixed(with_skin) || mixed(with_colors) {
        return Err(source.refuse(
            "MixedAttributes",
            item,
            "merged primitives differ in TEXCOORD_1, COLOR_0 or skin",
        ));
    }
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut uv1 = Vec::new();
    let mut colors = Vec::new();
    let mut skin = Vec::new();
    for part in parts {
        let base = u32::try_from(vertices.len()).expect("vertex_count_fits_u32");
        indices.extend(part.indices.iter().map(|index| index + base));
        vertices.extend_from_slice(&part.vertices);
        uv1.extend(part.uv1.iter().flatten().copied());
        colors.extend(part.colors.iter().flatten().copied());
        skin.extend(part.skin.iter().flatten().copied());
    }
    indices = order::triangles(&indices, vertices.len());
    let old_of = order::first_use(&mut indices, vertices.len());
    let (uv1, colors, skin) = (
        reordered(with_uv1, &uv1, &old_of),
        reordered(with_colors, &colors, &old_of),
        reordered(with_skin, &skin, &old_of),
    );
    vertices = order::permuted(&vertices, &old_of);
    let bytes = pack(&Unpacked {
        vertices: &vertices,
        indices: &indices,
        uv1: uv1.as_deref(),
        colors: colors.as_deref(),
        skin: skin.as_deref(),
    })
    .map_err(|error| source.refuse("MeshPack", item, error))?;
    Ok(Built {
        stem: String::new(),
        bytes: read_back(source, item, bytes)?,
        material: parts[0].material,
        vertices: vertices.len(),
        triangles: indices.len() / 3,
        influences: parts
            .iter()
            .map(|part| part.influences)
            .max()
            .unwrap_or_default(),
        colored: with_colors != 0,
        uv1: with_uv1 != 0,
    })
}

pub fn encode_shape(source: &Source, item: &str, shapes: &[Shape]) -> Result<Vec<u8>, Refusal> {
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    for held in shapes {
        let base = u32::try_from(positions.len()).expect("vertex_count_fits_u32");
        indices.extend(held.indices.iter().map(|index| index + base));
        positions.extend_from_slice(&held.positions);
    }
    let bytes =
        shape(&positions, &indices).map_err(|error| source.refuse("MeshPack", item, error))?;
    read_back(source, item, bytes)
}

fn read_back(source: &Source, item: &str, bytes: Vec<u8>) -> Result<Vec<u8>, Refusal> {
    MeshFile::read(Arc::from(bytes.as_slice()), item)
        .map_err(|error| source.refuse("MeshReadBack", item, error))?;
    Ok(bytes)
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

pub fn primitive_item(node: &gltf::Node<'_>, held: &gltf::Primitive<'_>) -> String {
    format!("{} primitive {}", node_label(node), held.index())
}

pub fn character(source: &Source, rig: &Rig, name: &str) -> Result<Vec<Built>, Refusal> {
    let place = Mat4::from_scale(Vec3::splat(rig.scale)) * rig.space.inverse();
    let mut built = Vec::new();
    for node in source.gltf.nodes() {
        let Some(mesh) = node.mesh() else { continue };
        if node.skin().is_none() {
            return Err(source.refuse(
                "UnskinnedMesh",
                node_label(&node),
                "a character file holds only skinned meshes",
            ));
        }
        for held in mesh.primitives() {
            let item = primitive_item(&node, &held);
            let part = read_part(source, &item, &held, place, Some(rig))?;
            built.push(encode_mesh(source, &item, &[part])?);
        }
    }
    if built.is_empty() {
        return Err(source.refuse("NoMesh", "nodes", "no mesh to cook"));
    }
    Ok(named(name, built))
}

fn rigid(place: Mat4, tolerance: f32) -> bool {
    let (scale, _, _) = place.to_scale_rotation_translation();
    place.determinant() > 0.0
        && (scale - Vec3::splat(scale.x)).abs().max_element() <= tolerance * scale.x
}

pub fn placed(
    source: &Source,
    node: &gltf::Node<'_>,
    scale: f32,
    tolerance: f32,
) -> Result<Mat4, Refusal> {
    if node.skin().is_some() {
        return Err(source.refuse(
            "SkinInStatic",
            node_label(node),
            "a static or map entry takes no skin",
        ));
    }
    let world = source.world(node.index());
    if !rigid(world, tolerance) {
        return Err(source.refuse(
            "NodeScale",
            node_label(node),
            "negative or non-uniform scale",
        ));
    }
    Ok(Mat4::from_scale(Vec3::splat(scale)) * world)
}

pub fn fixed(
    source: &Source,
    name: &str,
    scale: f32,
    tolerance: f32,
) -> Result<Vec<Built>, Refusal> {
    let mut built = Vec::new();
    for node in source.gltf.nodes() {
        let Some(mesh) = node.mesh() else { continue };
        let place = placed(source, &node, scale, tolerance)?;
        for held in mesh.primitives() {
            let item = primitive_item(&node, &held);
            let part = read_part(source, &item, &held, place, None)?;
            built.push(encode_mesh(source, &item, &[part])?);
        }
    }
    if built.is_empty() {
        return Err(source.refuse("NoMesh", "nodes", "no mesh to cook"));
    }
    Ok(named(name, built))
}

pub fn budget(
    source: &Source,
    code: &'static str,
    count: usize,
    budget: usize,
) -> Result<(), Refusal> {
    if count > budget {
        return Err(source.refuse(code, "budgets", format!("{count} > {budget}")));
    }
    Ok(())
}

pub fn counted(
    step: &mut Step<'_>,
    source: &Source,
    meshes: &[Built],
    (triangles, vertices): (usize, usize),
) -> Result<usize, Refusal> {
    let vertex_count: usize = meshes.iter().map(|mesh| mesh.vertices).sum();
    let triangle_count: usize = meshes.iter().map(|mesh| mesh.triangles).sum();
    let influences = meshes
        .iter()
        .map(|mesh| mesh.influences)
        .max()
        .unwrap_or_default();
    let colored: BTreeMap<&str, usize> = meshes
        .iter()
        .filter(|mesh| mesh.colored)
        .map(|mesh| (mesh.stem.as_str(), mesh.vertices))
        .collect();
    step.note(json!({"vertices": vertex_count, "triangles": triangle_count, "max_influences": influences, "color_stream_vertices": colored}));
    budget(source, "BudgetTriangles", triangle_count, triangles)?;
    budget(source, "BudgetVertices", vertex_count, vertices)?;
    Ok(influences)
}
