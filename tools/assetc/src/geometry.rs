use engine_scene::{
    MeshSource, SceneDocument, SceneMesh, SceneObject, SceneTransform, SceneVertex,
    encode_mesh_buffer,
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
                    &primitive,
                    binary,
                    matrix,
                    normal_matrix,
                    textured,
                    &primitive_path,
                )?;
                if determinant < 0.0 {
                    for triangle in indices.as_chunks_mut::<3>().0 {
                        triangle.swap(1, 2);
                    }
                }
                let object_id = format!("{id}.node{}.p{}", node.index(), primitive.index());
                let mesh_id = format!("{object_id}.mesh");
                let file = ImportedFile::named(
                    id,
                    &format!("node{}.p{}", node.index(), primitive.index()),
                    "bin",
                    encode_mesh_buffer(&vertices, &indices),
                );
                output.meshes.push(SceneMesh {
                    id: mesh_id.clone(),
                    source: MeshSource::Buffer {
                        uri: file.uri.clone(),
                    },
                });
                files.push(file);
                output.entities.push(SceneObject {
                    id: object_id,
                    name: node
                        .name()
                        .filter(|name| name.len() <= 128 && !name.chars().any(char::is_control))
                        .unwrap_or("Imported mesh")
                        .to_owned(),
                    mesh: Some(mesh_id),
                    material: Some(format!("{id}.material{material_index}")),
                    parent: None,
                    prefab: None,
                    collider: None,
                    light: None,
                    particles: None,
                    transform: SceneTransform::default(),
                    tags: vec!["imported".into()],
                });
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
            // Normal maps derive their tangent frames (decision 031); maps
            // read TEXCOORD_0 only.
            Semantic::Tangents | Semantic::TexCoords(_) => continue,
            _ => {
                return Err(unsupported(
                    format!("{path}.attributes"),
                    "only POSITION, NORMAL, TANGENT and TEXCOORD attributes are supported",
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
    let accessor = primitive
        .indices()
        .ok_or_else(|| unsupported(format!("{path}.indices"), "indexed primitives are required"))?;
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
    let mut indices: Vec<u32> = reader
        .read_indices()
        .ok_or_else(|| ImportError::new("invalid_accessor", path, "cannot read indices"))?
        .into_u32()
        .collect();
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
        vertices.push(SceneVertex {
            position: position.to_array(),
            normal: normal.normalize().to_array(),
            uv,
        });
    }
    Ok((vertices, indices))
}
