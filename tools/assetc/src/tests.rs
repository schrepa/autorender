use engine_scene::{MeshSource, SceneAlphaMode, SceneMap, SceneMipmapFilter, SceneWrap};
use serde_json::{Value, json};

use crate::{Imported, import_glb, import_glb_prefab, source_sha256};

const FIXTURE: &[u8] = include_bytes!("../../../assets/source/garden-prop.glb");

fn changed(change: impl FnOnce(&mut Value)) -> Vec<u8> {
    let json_length = u32::from_le_bytes(FIXTURE[12..16].try_into().unwrap()) as usize;
    let mut json: Value = serde_json::from_slice(&FIXTURE[20..20 + json_length]).unwrap();
    change(&mut json);
    let mut json = serde_json::to_vec(&json).unwrap();
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }
    let binary_chunk = &FIXTURE[20 + json_length..];
    let mut bytes = b"glTF".to_vec();
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&((20 + json.len() + binary_chunk.len()) as u32).to_le_bytes());
    bytes.extend_from_slice(&(json.len() as u32).to_le_bytes());
    bytes.extend_from_slice(b"JSON");
    bytes.extend_from_slice(&json);
    bytes.extend_from_slice(binary_chunk);
    bytes
}

/// Positions, normals and UVs, and indices, of an imported mesh buffer.
fn mesh(imported: &Imported, index: usize) -> (Vec<[f32; 8]>, Vec<u32>) {
    let MeshSource::Buffer { uri } = &imported.document.meshes[index].source else {
        panic!("expected a mesh buffer");
    };
    let bytes = &imported
        .files
        .iter()
        .find(|file| file.uri == *uri)
        .unwrap()
        .bytes;
    assert_eq!(&bytes[..4], b"EMB1");
    let word = |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let (vertices, indices) = (word(4) as usize, word(8) as usize);
    let float = |offset: usize| f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    (
        (0..vertices)
            .map(|vertex| std::array::from_fn(|lane| float(12 + vertex * 32 + lane * 4)))
            .collect(),
        (0..indices)
            .map(|index| word(12 + vertices * 32 + index * 4))
            .collect(),
    )
}

#[test]
fn fixture_is_deterministic_and_writes_its_images_and_meshes_as_files() {
    let a = import_glb(FIXTURE, "garden").unwrap();
    let b = import_glb(FIXTURE, "garden").unwrap();
    let source = a.document.to_json().unwrap();
    assert_eq!(source, b.document.to_json().unwrap());
    assert_eq!(a.files, b.files);
    let document = &a.document;
    assert_eq!(document.version, engine_scene::SCENE_VERSION);
    assert!(
        document
            .entities
            .iter()
            .all(|entity| entity.collider.is_none())
    );
    assert_eq!(
        (
            document.meshes.len(),
            document.materials.len(),
            document.textures.len(),
            document.entities.len()
        ),
        (2, 2, 1, 2)
    );
    // The PNG is copied unchanged; the host decodes it. Files are named by
    // their content.
    let image = &a.files[0];
    assert_eq!(document.textures[0].uri.as_ref(), Some(&image.uri));
    assert_eq!(&image.bytes[..8], b"\x89PNG\r\n\x1a\n");
    let names: Vec<_> = a
        .files
        .iter()
        .map(|file| {
            let (name, hash) = file
                .uri
                .rsplit_once('.')
                .unwrap()
                .0
                .rsplit_once('.')
                .unwrap();
            assert_eq!(hash.len(), 12);
            name
        })
        .collect();
    assert_eq!(
        names,
        ["garden/image0", "garden/node1.p0", "garden/node1.p1"]
    );
    assert_eq!(
        document.entities[1].material.as_deref(),
        Some("garden.material1")
    );
    assert!(a.ignored_extensions.is_empty());
    assert_eq!(source_sha256(FIXTURE).len(), 64);
    engine_scene::SceneDocument::from_json(&source).unwrap();
}

#[test]
fn gltf_materials_map_one_to_one() {
    let imported = import_glb(
        &changed(|json| {
            let material = &mut json["materials"][0];
            material["pbrMetallicRoughness"]["metallicFactor"] = json!(0.75);
            material["pbrMetallicRoughness"]["roughnessFactor"] = json!(0.25);
            material["pbrMetallicRoughness"]["baseColorFactor"] = json!([1, 0.5, 0.25, 0.5]);
            material["pbrMetallicRoughness"]["metallicRoughnessTexture"] = json!({"index": 0});
            material["normalTexture"] = json!({"index": 0, "scale": 0.5});
            material["occlusionTexture"] = json!({"index": 0, "strength": 0.25});
            material["emissiveTexture"] = json!({"index": 0, "extensions":
                {"KHR_texture_transform": {"offset": [0.5, 0], "rotation": 0.25, "scale": [4, 2]}}});
            material["emissiveFactor"] = json!([1, 0.5, 0]);
            material["extensions"] = json!({"KHR_materials_emissive_strength": {"emissiveStrength": 4}});
            material["alphaMode"] = json!("MASK");
            material["alphaCutoff"] = json!(0.3);
            material["doubleSided"] = json!(true);
            json["samplers"][0]["wrapS"] = json!(33648);
            json["samplers"][0]["minFilter"] = json!(9984);
            json["extensionsUsed"] =
                json!(["KHR_texture_transform", "KHR_materials_emissive_strength", "EXT_other"]);
        }),
        "garden",
    )
    .unwrap();
    let material = &imported.document.materials[0];
    assert_eq!(material.color, [1.0, 0.5, 0.25, 1.0]);
    assert_eq!(material.opacity, Some(0.5));
    assert_eq!(material.alpha_mode, Some(SceneAlphaMode::Mask));
    assert_eq!(material.alpha_cutoff, Some(0.3));
    assert_eq!(material.double_sided, Some(true));
    let surface = material.surface.unwrap();
    assert_eq!((surface.roughness, surface.metalness), (0.25, 0.75));
    assert_eq!(material.normal_scale, Some(0.5));
    assert_eq!(material.occlusion_strength, Some(0.25));
    assert_eq!(material.emissive, Some([4.0, 2.0, 0.0]));
    let Some(SceneMap::Map(emissive)) = &material.emissive_texture else {
        panic!("expected a texture map");
    };
    assert_eq!(emissive.id, "garden.image0");
    assert_eq!(emissive.wrap_u, Some(SceneWrap::MirroredRepeat));
    assert_eq!(emissive.wrap_v, Some(SceneWrap::ClampToEdge));
    assert_eq!(emissive.mipmap_filter, Some(SceneMipmapFilter::Nearest));
    assert_eq!(
        (emissive.offset, emissive.rotation, emissive.scale),
        (Some([0.5, 0.0]), Some(0.25), Some([4.0, 2.0]))
    );
    assert!(material.metallic_roughness_texture.is_some() && material.normal_texture.is_some());
    assert_eq!(imported.ignored_extensions, ["EXT_other"]);
    imported.document.validate().unwrap();
}

#[test]
fn prefab_conversion_is_deterministic_without_placement_or_geometry_changes() {
    let flat = import_glb(FIXTURE, "garden").unwrap().document;
    let imported = import_glb_prefab(FIXTURE, "garden").unwrap();
    let prefab = &imported.document;
    assert_eq!(
        prefab.to_json().unwrap(),
        import_glb_prefab(FIXTURE, "garden")
            .unwrap()
            .document
            .to_json()
            .unwrap()
    );
    assert!(prefab.entities.is_empty());
    assert_eq!(prefab.prefabs.len(), 1);
    assert_eq!(prefab.prefabs[0].id, "garden");
    assert_eq!(prefab.prefabs[0].entities.len(), flat.entities.len());
    assert_eq!(prefab.prefabs[0].entities[0].id, "node1.p0");
    assert_eq!(
        serde_json::to_value(&flat.meshes).unwrap(),
        serde_json::to_value(&prefab.meshes).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&flat.textures).unwrap(),
        serde_json::to_value(&prefab.textures).unwrap()
    );
    for (template, original) in prefab.prefabs[0].entities.iter().zip(&flat.entities) {
        assert_eq!(template.mesh, original.mesh);
        assert_eq!(template.material, original.material);
        assert!(template.parent.is_none() && template.prefab.is_none());
    }
}

#[test]
fn nested_transforms_bake_positions_and_inverse_transpose_normals() {
    let document = import_glb(
        &changed(|json| {
            json["nodes"][0]["translation"] = json!([5, 2, -3]);
            json["nodes"][1]["scale"] = json!([2, 3, 4]);
        }),
        "garden",
    )
    .unwrap();
    let original = import_glb(FIXTURE, "garden").unwrap();
    let (vertices, _) = mesh(&document, 0);
    let (before, _) = mesh(&original, 0);
    for (actual, original) in vertices.iter().zip(&before) {
        let expected = [
            original[0] * 2.0 + 5.0,
            original[1] * 3.0 + 2.0,
            original[2] * 4.0 - 3.0,
        ];
        for (actual, expected) in actual[..3].iter().zip(expected) {
            assert!((actual - expected).abs() < 0.00001);
        }
        let normal =
            glam::Vec3::new(original[3], original[4], original[5]) / glam::Vec3::new(2.0, 3.0, 4.0);
        let actual = glam::Vec3::new(actual[3], actual[4], actual[5]);
        assert!((actual - normal.normalize()).length() < 0.00001);
    }
}

#[test]
fn mirrored_nodes_reverse_triangle_winding() {
    let document = import_glb(
        &changed(|json| json["nodes"][1]["scale"] = json!([-1, 1, 1])),
        "garden",
    )
    .unwrap();
    let (_, indices) = mesh(&document, 0);
    assert_eq!(&indices[..3], &[0, 2, 1]);
}

#[test]
fn primitives_without_normals_get_flat_ones() {
    let document = import_glb(
        &changed(|json| {
            let attributes = json["meshes"][0]["primitives"][0]["attributes"]
                .as_object_mut()
                .unwrap();
            attributes.remove("NORMAL");
        }),
        "garden",
    )
    .unwrap();
    let (vertices, indices) = mesh(&document, 0);
    // Each triangle gets its own three vertices, facing along its winding.
    assert_eq!(vertices.len(), indices.len());
    for triangle in indices.as_chunks::<3>().0 {
        let [a, b, c] = triangle.map(|index| {
            let vertex = vertices[index as usize];
            glam::Vec3::new(vertex[0], vertex[1], vertex[2])
        });
        let face = (b - a).cross(c - a);
        let normal = vertices[triangle[0] as usize];
        if face.length() > 1.0e-6 {
            let normal = glam::Vec3::new(normal[3], normal[4], normal[5]);
            assert!((normal - face.normalize()).length() < 1.0e-4);
        }
    }
}

#[test]
fn cycles_disconnected_nodes_and_excessive_depth_are_rejected() {
    let cycle = changed(|json| json["nodes"][1]["children"] = json!([0]));
    assert_eq!(
        import_glb(&cycle, "garden").unwrap_err().code,
        "invalid_node_tree"
    );
    let detached = changed(|json| json["nodes"].as_array_mut().unwrap().push(json!({})));
    assert_eq!(
        import_glb(&detached, "garden").unwrap_err().code,
        "unsupported_feature"
    );
    let deep = changed(|json| {
        json["nodes"] = Value::Array(
            (0..67)
                .map(|index| {
                    if index == 66 {
                        json!({"mesh":0})
                    } else {
                        json!({"children":[index+1]})
                    }
                })
                .collect(),
        );
    });
    assert_eq!(import_glb(&deep, "garden").unwrap_err().code, "depth_limit");
}

#[test]
fn invalid_accessor_ranges_and_unbounded_counts_fail_before_reading() {
    let bounds = changed(|json| json["accessors"][0]["byteOffset"] = json!(1_000_000));
    assert_eq!(
        import_glb(&bounds, "garden").unwrap_err().code,
        "accessor_bounds"
    );
    // No count budget: counts are bounded by the data they read.
    let count = changed(|json| json["accessors"][0]["count"] = json!(1_000_000));
    assert_eq!(
        import_glb(&count, "garden").unwrap_err().code,
        "accessor_bounds"
    );
    let view = changed(|json| json["bufferViews"][0]["byteLength"] = json!(100_000_000));
    assert_eq!(
        import_glb(&view, "garden").unwrap_err().code,
        "buffer_bounds"
    );
    let overflow = changed(|json| json["accessors"][1]["byteOffset"] = json!(u64::MAX - 3));
    assert_eq!(
        import_glb(&overflow, "garden").unwrap_err().code,
        "accessor_bounds"
    );
}

#[test]
fn unsupported_resources_texture_sets_and_required_extensions_are_explicit() {
    for change in [
        |json: &mut Value| {
            json["images"][0] = json!({"uri":"https://example.invalid/texture.png"});
        },
        |json: &mut Value| json["buffers"][0]["uri"] = json!("external.bin"),
        |json: &mut Value| {
            json["extensionsUsed"] = json!(["KHR_draco_mesh_compression"]);
            json["extensionsRequired"] = json!(["KHR_draco_mesh_compression"]);
        },
        |json: &mut Value| {
            json["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["texCoord"] = json!(1)
        },
    ] {
        assert_eq!(
            import_glb(&changed(change), "garden").unwrap_err().code,
            "unsupported_feature"
        );
    }
}

#[test]
fn default_samplers_import_as_plain_texture_ids() {
    // Repeating trilinear sampling is glTF's default and a plain ID's.
    let trilinear = import_glb(
        &changed(|json| json["samplers"][0] = json!({"minFilter": 9987})),
        "garden",
    )
    .unwrap();
    assert_eq!(
        trilinear.document.materials[0].texture,
        Some(SceneMap::Id("garden.image0".into()))
    );
    // The fixture clamps and samples its base level only.
    let fixture = import_glb(FIXTURE, "garden").unwrap();
    let Some(SceneMap::Map(map)) = &fixture.document.materials[0].texture else {
        panic!("expected a texture map");
    };
    assert_eq!(
        (map.wrap_u, map.wrap_v, map.mipmap_filter),
        (
            Some(SceneWrap::ClampToEdge),
            Some(SceneWrap::ClampToEdge),
            Some(SceneMipmapFilter::None)
        )
    );
}

#[test]
fn truncated_oversized_and_foreign_containers_fail() {
    assert_eq!(
        import_glb(&FIXTURE[..20], "garden").unwrap_err().code,
        "invalid_glb"
    );
    assert_eq!(
        import_glb(b"{\"asset\":{\"version\":\"2.0\"}}", "garden")
            .unwrap_err()
            .code,
        "invalid_glb"
    );
    assert_eq!(
        import_glb(FIXTURE, "../unsafe").unwrap_err().code,
        "invalid_id"
    );
}

#[test]
fn corrupt_geometry_and_singular_transforms_are_rejected() {
    let singular = changed(|json| json["nodes"][1]["scale"] = json!([0, 1, 1]));
    assert_eq!(
        import_glb(&singular, "garden").unwrap_err().code,
        "invalid_transform"
    );
    let mut nonfinite = FIXTURE.to_vec();
    let json_length = u32::from_le_bytes(FIXTURE[12..16].try_into().unwrap()) as usize;
    nonfinite[28 + json_length..32 + json_length].copy_from_slice(&f32::NAN.to_le_bytes());
    assert_eq!(
        import_glb(&nonfinite, "garden").unwrap_err().code,
        "invalid_vertex"
    );
}

#[test]
fn unsupported_versions_and_ambiguous_or_nonunit_transforms_are_rejected() {
    for change in [
        |json: &mut Value| json["asset"]["version"] = json!("3.0"),
        |json: &mut Value| json["asset"]["minVersion"] = json!("2.1"),
    ] {
        assert_eq!(
            import_glb(&changed(change), "garden").unwrap_err().code,
            "unsupported_feature"
        );
    }
    for change in [
        |json: &mut Value| json["nodes"][0]["rotation"] = json!([0, 0, 0, 0]),
        |json: &mut Value| {
            json["nodes"][0]["matrix"] = json!([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
            json["nodes"][0]["translation"] = json!([1, 2, 3]);
        },
    ] {
        assert_eq!(
            import_glb(&changed(change), "garden").unwrap_err().code,
            "invalid_transform"
        );
    }
}
