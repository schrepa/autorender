use engine_scene::{
    MeshSource, MeshStreams, SceneAlphaMode, SceneDocument, SceneMaterial, SceneMesh, SceneObject,
    ScenePlayer, SceneTransform, SceneVertex, encode_mesh_buffer_with,
};
use glam::{EulerRot, Mat3, Mat4, Quat, Vec3};
use gltf::{
    accessor::{DataType, Dimensions},
    mesh::Semantic,
};

use crate::{ImportError, ImportOptions, ImportedFile, unsupported};

/// Static models bake node transforms into their vertices; models with
/// animations or skins keep their nodes.
pub(crate) fn import(
    document: &gltf::Document,
    binary: &[u8],
    id: &str,
    options: &ImportOptions,
    output: &mut SceneDocument,
    files: &mut Vec<ImportedFile>,
) -> Result<(), ImportError> {
    if document.animations().next().is_some() || document.skins().next().is_some() {
        return hierarchy(document, binary, id, options, output, files);
    }
    let source = document.scenes().next().expect("validated single scene");
    let mut stack: Vec<_> = source.nodes().map(|node| (node, Mat4::IDENTITY)).collect();
    stack.reverse();
    while let Some((node, parent)) = stack.pop() {
        let path = format!("$.nodes[{}]", node.index());
        let local = Mat4::from_cols_array_2d(&node.transform().matrix());
        let matrix = parent * local;
        let linear = Mat3::from_mat4(matrix);
        let determinant = linear.determinant();
        if !matrix.is_finite()
            || !determinant.is_finite()
            || determinant.abs() < 0.000_000_01
            || matrix.x_axis.w != 0.0
            || matrix.y_axis.w != 0.0
            || matrix.z_axis.w != 0.0
            || matrix.w_axis.w != 1.0
        {
            return Err(ImportError::new(
                "invalid_transform",
                &path,
                "node world transforms must be finite, affine and nonsingular",
            ));
        }
        let normal_matrix = linear.inverse().transpose();
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                let (object, mesh, file) = import_primitive(
                    output,
                    binary,
                    id,
                    options,
                    &node,
                    &primitive,
                    matrix,
                    normal_matrix,
                    false,
                )?;
                output.meshes.push(mesh);
                files.push(file);
                output.entities.push(object);
            }
        }
        let start = stack.len();
        stack.extend(node.children().map(|child| (child, matrix)));
        stack[start..].reverse();
    }
    if output.entities.is_empty() {
        return Err(ImportError::new(
            "empty_model",
            "$.nodes",
            "the scene contains no mesh primitives",
        ));
    }
    Ok(())
}

/// A primitive's entity, mesh and mesh file, its vertices transformed by
/// `matrix`; with `skinned`, carrying the primitive's joints and weights.
/// Dense opaque primitives without extra streams become virtual meshes
/// as `options` say.
#[allow(clippy::too_many_arguments)]
fn import_primitive(
    output: &SceneDocument,
    binary: &[u8],
    id: &str,
    options: &ImportOptions,
    node: &gltf::Node<'_>,
    primitive: &gltf::Primitive<'_>,
    matrix: Mat4,
    normal_matrix: Mat3,
    skinned: bool,
) -> Result<(SceneObject, SceneMesh, ImportedFile), ImportError> {
    let mesh = node.mesh().expect("mesh nodes");
    let primitive_path = format!(
        "$.meshes[{}].primitives[{}]",
        mesh.index(),
        primitive.index()
    );
    let material_index = primitive.material().index().ok_or_else(|| {
        unsupported(
            format!("{primitive_path}.material"),
            "an explicit material is required",
        )
    })?;
    let material = &output.materials[material_index];
    let textured = material.texture.is_some()
        || material.metallic_roughness_texture.is_some()
        || material.normal_texture.is_some()
        || material.occlusion_texture.is_some()
        || material.emissive_texture.is_some();
    let (vertices, mut indices) = primitive_data(
        primitive,
        binary,
        matrix,
        normal_matrix,
        textured,
        skinned,
        &primitive_path,
    )?;
    if Mat3::from_mat4(matrix).determinant() < 0.0 {
        for triangle in indices.as_chunks_mut::<3>().0 {
            triangle.swap(1, 2);
        }
    }
    let object_id = format!("{id}.node{}.p{}", node.index(), primitive.index());
    let mesh_id = format!("{object_id}.mesh");
    // Vertex colors, second coordinates, joints and weights
    // ride in an EMB2 buffer's optional streams.
    let colors: Option<Vec<[f32; 4]>> = vertices.iter().map(|vertex| vertex.color).collect();
    let uv1: Option<Vec<[f32; 2]>> = vertices.iter().map(|vertex| vertex.uv1).collect();
    let joints: Option<Vec<[u16; 4]>> = vertices.iter().map(|vertex| vertex.joints).collect();
    let weights: Option<Vec<[f32; 4]>> = vertices.iter().map(|vertex| vertex.weights).collect();
    let name = format!("node{}.p{}", node.index(), primitive.index());
    let clustered = options
        .virtual_triangles
        .is_some_and(|triangles| indices.len() / 3 >= triangles as usize)
        && colors.is_none()
        && uv1.is_none()
        && joints.is_none()
        && opaque(material);
    let (file, source) = if clustered {
        let positions: Vec<[f32; 3]> = vertices.iter().map(|vertex| vertex.position).collect();
        let normals: Vec<[f32; 3]> = vertices.iter().map(|vertex| vertex.normal).collect();
        let uvs: Option<Vec<[f32; 2]>> =
            textured.then(|| vertices.iter().map(|vertex| vertex.uv).collect());
        let bytes = engine_vgbuild::build_virtual_mesh(&engine_vgbuild::Source {
            positions: &positions,
            normals: &normals,
            uvs: uvs.as_deref(),
            indices: &indices,
        })
        .map_err(|message| ImportError::new("invalid_mesh", &primitive_path, message))?;
        let file = ImportedFile::named(id, &name, "evg", bytes);
        let source = MeshSource::Virtual {
            uri: file.uri.clone(),
            fallback_triangles: None,
        };
        (file, source)
    } else {
        let file = ImportedFile::named(
            id,
            &name,
            "bin",
            encode_mesh_buffer_with(
                &vertices,
                &indices,
                MeshStreams {
                    colors: colors.as_deref(),
                    uv1: uv1.as_deref(),
                    joints: joints.as_deref(),
                    weights: weights.as_deref(),
                },
            ),
        );
        let source = MeshSource::Buffer {
            uri: file.uri.clone(),
        };
        (file, source)
    };
    let skin = node
        .skin()
        .filter(|_| joints.is_some())
        .map(|skin| format!("{id}.skin{}", skin.index()));
    let object = SceneObject {
        id: object_id,
        name: node_name(node, "Imported mesh"),
        mesh: Some(mesh_id.clone()),
        material: Some(format!("{id}.material{material_index}")),
        parent: None,
        prefab: None,
        collider: None,
        light: None,
        particles: None,
        tint: None,
        animation: None,
        skin,
        decal: None,
        reflection_probe: None,
        terrain: None,
        spline: None,
        floating: None,
        deformer: None,
        grounded: None,
        prefetch: None,
        nav_agent: None,
        transform: SceneTransform::default(),
        tags: vec!["imported".into()],
    };
    let mesh = SceneMesh {
        id: mesh_id,
        source,
    };
    Ok((object, mesh, file))
}

/// Whether `material` draws opaque: neither alpha-tested (virtual shadows
/// cast whole clusters) nor blended, see-through or transmissive.
fn opaque(material: &SceneMaterial) -> bool {
    material
        .alpha_mode
        .is_none_or(|mode| mode == SceneAlphaMode::Opaque)
        && material.opacity.is_none_or(|opacity| opacity >= 1.0)
        && material
            .transmission
            .is_none_or(|transmission| transmission <= 0.0)
}

fn node_name(node: &gltf::Node<'_>, fallback: &str) -> String {
    node.name()
        .filter(|name| name.len() <= 128 && !name.chars().any(char::is_control))
        .unwrap_or(fallback)
        .to_owned()
}

/// Animated and skinned models: one entity per node (`ID.nodeN`) with its
/// local transform, below one root (`ID.root`) that plays the first
/// animation; primitives are children (`ID.nodeN.pM`) in their node's space.
fn hierarchy(
    document: &gltf::Document,
    binary: &[u8],
    id: &str,
    options: &ImportOptions,
    output: &mut SceneDocument,
    files: &mut Vec<ImportedFile>,
) -> Result<(), ImportError> {
    let root = format!("{id}.root");
    output.entities.push(SceneObject {
        id: root.clone(),
        name: "Imported model".into(),
        mesh: None,
        material: None,
        parent: None,
        prefab: None,
        collider: None,
        light: None,
        particles: None,
        tint: None,
        animation: None,
        skin: None,
        decal: None,
        reflection_probe: None,
        terrain: None,
        spline: None,
        floating: None,
        deformer: None,
        grounded: None,
        prefetch: None,
        nav_agent: None,
        transform: SceneTransform::default(),
        tags: vec!["imported".into()],
    });
    let source = document.scenes().next().expect("validated single scene");
    let mut stack: Vec<_> = source.nodes().map(|node| (node, root.clone())).collect();
    stack.reverse();
    let mut primitives = 0;
    while let Some((node, parent)) = stack.pop() {
        let path = format!("$.nodes[{}]", node.index());
        let node_id = format!("{id}.node{}", node.index());
        output.entities.push(SceneObject {
            id: node_id.clone(),
            name: node_name(&node, "Imported node"),
            mesh: None,
            material: None,
            parent: Some(parent),
            prefab: None,
            collider: None,
            light: None,
            particles: None,
            tint: None,
            animation: None,
            skin: None,
            decal: None,
            reflection_probe: None,
            terrain: None,
            spline: None,
            floating: None,
            deformer: None,
            grounded: None,
            prefetch: None,
            nav_agent: None,
            transform: local_transform(&node, &path)?,
            tags: vec!["imported".into()],
        });
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                // Skinned vertices stay in bind space: joints place them.
                let (mut object, mesh, file) = import_primitive(
                    output,
                    binary,
                    id,
                    options,
                    &node,
                    &primitive,
                    Mat4::IDENTITY,
                    Mat3::IDENTITY,
                    node.skin().is_some(),
                )?;
                object.parent = Some(node_id.clone());
                output.meshes.push(mesh);
                files.push(file);
                output.entities.push(object);
                primitives += 1;
            }
        }
        let start = stack.len();
        stack.extend(node.children().map(|child| (child, node_id.clone())));
        stack[start..].reverse();
    }
    if primitives == 0 {
        return Err(ImportError::new(
            "empty_model",
            "$.nodes",
            "the scene contains no mesh primitives",
        ));
    }
    crate::animation::import(document, binary, id, output, files)?;
    if let Some(first) = output.animations.first() {
        output.entities[0].animation = Some(ScenePlayer {
            clip: first.id.clone(),
            speed: 1.0,
            looping: true,
            time: 0.0,
        });
    }
    Ok(())
}

/// A node's local transform as scene TRS: rotations as XYZ Euler degrees,
/// scales positive (mirrored nodes are unsupported in hierarchies).
fn local_transform(node: &gltf::Node<'_>, path: &str) -> Result<SceneTransform, ImportError> {
    let (translation, rotation, scale) = node.transform().decomposed();
    let rotation = Quat::from_array(rotation);
    let invalid = |message: &str| ImportError::new("invalid_transform", path, message.to_owned());
    if !rotation.is_finite() || rotation.length_squared() == 0.0 {
        return Err(invalid("node rotations must be finite and nonzero"));
    }
    let rotation = rotation.normalize();
    let (translation, scale) = (Vec3::from_array(translation), Vec3::from_array(scale));
    if !translation.is_finite() || translation.abs().max_element() > 1_000_000.0 {
        return Err(invalid(
            "node translations must be finite and within ±1000000",
        ));
    }
    if !scale.is_finite() || scale.min_element() < 0.0001 || scale.max_element() > 10_000.0 {
        return Err(invalid(
            "animated and skinned models need node scales between 0.0001 and 10000; mirrored nodes are unsupported",
        ));
    }
    // A matrix must be exactly a rotation, scale and translation.
    let original = Mat4::from_cols_array_2d(&node.transform().matrix());
    let rebuilt = Mat4::from_scale_rotation_translation(scale, rotation, translation);
    let tolerance = 0.0001
        * (1.0
            + original
                .abs()
                .to_cols_array()
                .iter()
                .fold(0.0, |a: f32, b| if *b > a { *b } else { a }));
    if (original - rebuilt)
        .to_cols_array()
        .iter()
        .any(|difference| difference.abs() > tolerance)
    {
        return Err(invalid(
            "node matrices of animated and skinned models must be rotations, scales and translations without shear",
        ));
    }
    let (x, y, z) = rotation.to_euler(EulerRot::XYZ);
    Ok(SceneTransform {
        translation: translation.to_array(),
        rotation_degrees: [x.to_degrees(), y.to_degrees(), z.to_degrees()],
        scale: scale.to_array(),
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn primitive_data(
    primitive: &gltf::Primitive<'_>,
    binary: &[u8],
    matrix: Mat4,
    normal_matrix: Mat3,
    textured: bool,
    skinned: bool,
    path: &str,
) -> Result<(Vec<SceneVertex>, Vec<u32>), ImportError> {
    if primitive.mode() != gltf::mesh::Mode::Triangles || primitive.morph_targets().next().is_some()
    {
        return Err(unsupported(
            path,
            "only indexed triangle primitives without morph targets are supported",
        ));
    }
    for (semantic, accessor) in primitive.attributes() {
        let dimensions = match semantic {
            Semantic::Positions | Semantic::Normals => Dimensions::Vec3,
            Semantic::TexCoords(0) => Dimensions::Vec2,
            // Normal maps derive their tangent frames; maps
            // read TEXCOORD_0 or TEXCOORD_1, and base colors COLOR_0,
            // skins JOINTS_0 and WEIGHTS_0,
            // whose formats the reader converts.
            Semantic::Tangents
            | Semantic::TexCoords(_)
            | Semantic::Colors(_)
            | Semantic::Joints(0)
            | Semantic::Weights(0) => continue,
            Semantic::Joints(_) | Semantic::Weights(_) => {
                return Err(unsupported(
                    format!("{path}.attributes"),
                    "at most four joints per vertex (JOINTS_0 and WEIGHTS_0) are supported",
                ));
            }
        };
        if accessor.data_type() != DataType::F32
            || accessor.dimensions() != dimensions
            || accessor.normalized()
        {
            return Err(unsupported(
                format!("{path}.attributes"),
                "vertex attributes must be unnormalized FLOAT VEC3 (position/normal) or FLOAT VEC2 (UV)",
            ));
        }
    }
    let positions = primitive.get(&Semantic::Positions).ok_or_else(|| {
        unsupported(
            format!("{path}.attributes.POSITION"),
            "POSITION is required",
        )
    })?;
    // Without normals, glTF asks for flat ones.
    let normals = primitive.get(&Semantic::Normals);
    let texcoords = primitive.get(&Semantic::TexCoords(0));
    if normals
        .as_ref()
        .is_some_and(|normals| normals.count() != positions.count())
        || texcoords
            .as_ref()
            .is_some_and(|uv| uv.count() != positions.count())
    {
        return Err(ImportError::new(
            "attribute_count",
            path,
            "vertex attribute counts must match",
        ));
    }
    if textured && texcoords.is_none() {
        return Err(unsupported(
            format!("{path}.attributes.TEXCOORD_0"),
            "textured primitives require TEXCOORD_0",
        ));
    }
    // Unindexed primitives draw their vertices in order.
    if let Some(accessor) = primitive.indices() {
        if accessor.dimensions() != Dimensions::Scalar
            || !matches!(
                accessor.data_type(),
                DataType::U8 | DataType::U16 | DataType::U32
            )
            || accessor.normalized()
            || accessor.count() % 3 != 0
            || accessor.view().is_some_and(|view| view.stride().is_some())
        {
            return Err(ImportError::new(
                "invalid_indices",
                format!("{path}.indices"),
                "indices must be tightly packed unsigned scalars in complete triangles",
            ));
        }
    } else if positions.count() % 3 != 0 {
        return Err(ImportError::new(
            "invalid_indices",
            format!("{path}.attributes.POSITION"),
            "unindexed primitives draw complete triangles",
        ));
    }
    let reader = primitive.reader(|buffer| {
        if buffer.index() == 0 {
            Some(binary)
        } else {
            None
        }
    });
    let position_data: Vec<Vec3> = reader
        .read_positions()
        .ok_or_else(|| ImportError::new("invalid_accessor", path, "cannot read positions"))?
        .map(Vec3::from)
        .collect();
    let normal_data: Option<Vec<Vec3>> = match normals {
        Some(_) => Some(
            reader
                .read_normals()
                .ok_or_else(|| ImportError::new("invalid_accessor", path, "cannot read normals"))?
                .map(Vec3::from)
                .collect(),
        ),
        None => None,
    };
    let uv_data: Option<Vec<[f32; 2]>> =
        reader.read_tex_coords(0).map(|uv| uv.into_f32().collect());
    let uv1_data: Option<Vec<[f32; 2]>> =
        reader.read_tex_coords(1).map(|uv| uv.into_f32().collect());
    let color_data: Option<Vec<[f32; 4]>> = reader
        .read_colors(0)
        .map(|colors| colors.into_rgba_f32().collect());
    // A skin's primitives carry both; without a skin glTF draws them rigid.
    let skin_data = if skinned {
        match (reader.read_joints(0), reader.read_weights(0)) {
            (Some(joints), Some(weights)) => Some((
                joints.into_u16().collect::<Vec<[u16; 4]>>(),
                weights.into_f32().collect::<Vec<[f32; 4]>>(),
            )),
            (None, None) => None,
            _ => {
                return Err(ImportError::new(
                    "attribute_count",
                    path,
                    "skinned primitives need both JOINTS_0 and WEIGHTS_0",
                ));
            }
        }
    } else {
        None
    };
    let (joint_data, weight_data) = match skin_data {
        Some((joints, weights)) => (Some(joints), Some(weights)),
        None => (None, None),
    };
    if uv1_data
        .as_ref()
        .is_some_and(|uv1| uv1.len() != position_data.len())
        || color_data
            .as_ref()
            .is_some_and(|colors| colors.len() != position_data.len())
        || joint_data
            .as_ref()
            .is_some_and(|joints| joints.len() != position_data.len())
        || weight_data
            .as_ref()
            .is_some_and(|weights| weights.len() != position_data.len())
    {
        return Err(ImportError::new(
            "attribute_count",
            path,
            "vertex attribute counts must match",
        ));
    }
    let mut indices: Vec<u32> = match primitive.indices() {
        Some(_) => reader
            .read_indices()
            .ok_or_else(|| ImportError::new("invalid_accessor", path, "cannot read indices"))?
            .into_u32()
            .collect(),
        None => (0..position_data.len() as u32).collect(),
    };
    if indices
        .iter()
        .any(|index| *index as usize >= position_data.len())
    {
        return Err(ImportError::new(
            "index_bounds",
            format!("{path}.indices"),
            "triangle index is outside the vertex array",
        ));
    }
    // Flat normals give each triangle its own three vertices.
    let (sources, local_normals): (Vec<u32>, Vec<Vec3>) = match &normal_data {
        Some(normals) => ((0..position_data.len() as u32).collect(), normals.clone()),
        None => {
            let mut sources = Vec::with_capacity(indices.len());
            let mut flat = Vec::with_capacity(indices.len());
            for triangle in indices.as_chunks::<3>().0 {
                let [a, b, c] = triangle.map(|index| position_data[index as usize]);
                let normal = (b - a).cross(c - a);
                // Degenerate triangles cover no pixels; any normal is valid.
                let normal = if normal.length_squared() > 0.0 {
                    normal
                } else {
                    Vec3::Y
                };
                sources.extend_from_slice(triangle);
                flat.extend([normal; 3]);
            }
            indices = (0..sources.len() as u32).collect();
            (sources, flat)
        }
    };
    let mut vertices = Vec::with_capacity(sources.len());
    for (&source, &normal) in sources.iter().zip(&local_normals) {
        let position = matrix.transform_point3(position_data[source as usize]);
        let normal = normal_matrix * normal;
        let uv = uv_data.as_ref().map_or([0.0; 2], |uv| uv[source as usize]);
        if !position.is_finite()
            || position.abs().max_element() > 1_000_000.0
            || !normal.is_finite()
            || normal.length_squared() < 0.000_000_000_001
            || uv.iter().any(|value| !value.is_finite())
        {
            return Err(ImportError::new(
                "invalid_vertex",
                path,
                "vertices must have finite bounded positions, nonzero finite normals and finite UVs",
            ));
        }
        let color = color_data.as_ref().map(|colors| colors[source as usize]);
        let uv1 = uv1_data.as_ref().map(|uv1| uv1[source as usize]);
        if color.is_some_and(|color| {
            !color
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        }) || uv1.is_some_and(|uv1| uv1.iter().any(|value| !value.is_finite()))
        {
            return Err(ImportError::new(
                "invalid_vertex",
                path,
                "vertex colors must be finite in [0, 1] and second UVs finite",
            ));
        }
        let joints = joint_data.as_ref().map(|joints| joints[source as usize]);
        let weights = weight_data.as_ref().map(|weights| weights[source as usize]);
        if weights.is_some_and(|weights| {
            !weights
                .iter()
                .all(|weight| weight.is_finite() && *weight >= 0.0)
                || weights.iter().sum::<f32>() <= 0.0
        }) {
            return Err(ImportError::new(
                "invalid_vertex",
                path,
                "joint weights must be finite, not negative and not all zero",
            ));
        }
        vertices.push(SceneVertex {
            position: position.to_array(),
            normal: normal.normalize().to_array(),
            uv,
            color,
            uv1,
            joints,
            weights,
        });
    }
    Ok((vertices, indices))
}
