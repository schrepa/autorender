use engine_scene::{
    MeshSource, MeshStreams, SceneDocument, SceneMesh, SceneObject, SceneTransform, SceneVertex,
    encode_mesh_buffer_with,
};
use glam::{Mat3, Mat4, Vec3};
use gltf::{
    accessor::{DataType, Dimensions},
    mesh::Semantic,
};

use crate::{ImportError, ImportedFile, unsupported};

pub(crate) fn import(
    document: &gltf::Document,
    binary: &[u8],
    id: &str,
    output: &mut SceneDocument,
    files: &mut Vec<ImportedFile>,
) -> Result<(), ImportError> {
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
                let (object, mesh, file) =
                    import_primitive(output, binary, id, &node, &primitive, matrix, normal_matrix)?;
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

fn import_primitive(
    output: &SceneDocument,
    binary: &[u8],
    id: &str,
    node: &gltf::Node<'_>,
    primitive: &gltf::Primitive<'_>,
    matrix: Mat4,
    normal_matrix: Mat3,
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
        &primitive_path,
    )?;
    if Mat3::from_mat4(matrix).determinant() < 0.0 {
        for triangle in indices.as_chunks_mut::<3>().0 {
            triangle.swap(1, 2);
        }
    }
    let object_id = format!("{id}.node{}.p{}", node.index(), primitive.index());
    let mesh_id = format!("{object_id}.mesh");
    let colors: Option<Vec<[f32; 4]>> = vertices.iter().map(|vertex| vertex.color).collect();
    let uv1: Option<Vec<[f32; 2]>> = vertices.iter().map(|vertex| vertex.uv1).collect();
    let name = format!("node{}.p{}", node.index(), primitive.index());
    let plain: Vec<engine_core::MeshVertex> =
        vertices.iter().map(|&vertex| vertex.into()).collect();
    let lods = engine_lodbuild::levels(&plain, &indices)
        .map_err(|message| ImportError::new("invalid_mesh", &primitive_path, message))?;
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
                lods: Some(&lods),
            },
        ),
    );
    let source = MeshSource::Buffer {
        uri: file.uri.clone(),
    };
    let object = SceneObject {
        id: object_id,
        name: node_name(node, "Imported mesh"),
        mesh: Some(mesh_id.clone()),
        material: Some(format!("{id}.material{material_index}")),
        transform: SceneTransform::default(),
        tags: vec!["imported".into()],
        ..SceneObject::default()
    };
    let mesh = SceneMesh {
        id: mesh_id,
        source,
    };
    Ok((object, mesh, file))
}

fn node_name(node: &gltf::Node<'_>, fallback: &str) -> String {
    node.name()
        .filter(|name| name.len() <= 128 && !name.chars().any(char::is_control))
        .unwrap_or(fallback)
        .to_owned()
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
            Semantic::Tangents
            | Semantic::TexCoords(_)
            | Semantic::Colors(_)
            | Semantic::Joints(_)
            | Semantic::Weights(_) => continue,
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
    if uv1_data
        .as_ref()
        .is_some_and(|uv1| uv1.len() != position_data.len())
        || color_data
            .as_ref()
            .is_some_and(|colors| colors.len() != position_data.len())
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
    let (sources, local_normals): (Vec<u32>, Vec<Vec3>) = match &normal_data {
        Some(normals) => ((0..position_data.len() as u32).collect(), normals.clone()),
        None => {
            let mut sources = Vec::with_capacity(indices.len());
            let mut flat = Vec::with_capacity(indices.len());
            for triangle in indices.as_chunks::<3>().0 {
                let [a, b, c] = triangle.map(|index| position_data[index as usize]);
                let normal = (b - a).cross(c - a);
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
        vertices.push(SceneVertex {
            position: position.to_array(),
            normal: normal.normalize().to_array(),
            uv,
            color,
            uv1,
        });
    }
    Ok((vertices, indices))
}
