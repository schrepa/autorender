use engine_scene::MeshSource;
use serde_json::{Value, json};

use crate::{MAX_INPUT_BYTES, import_glb, import_glb_prefab, source_sha256};

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

#[test]
fn fixture_is_deterministic_self_contained_and_small() {
    let a = import_glb(FIXTURE, "garden").unwrap();
    let b = import_glb(FIXTURE, "garden").unwrap();
    let source = a.to_json().unwrap();
    assert_eq!(source, b.to_json().unwrap());
    assert_eq!(a.version, engine_scene::SCENE_VERSION);
    assert!(a.entities.iter().all(|entity| entity.collider.is_none()));
    assert!(source.len() < 150 * 1024);
    assert_eq!(
        (
            a.meshes.len(),
            a.materials.len(),
            a.textures.len(),
            a.entities.len()
        ),
        (2, 2, 1, 2)
    );
    assert_eq!(a.textures[0].width, 64);
    assert_eq!(a.entities[1].material.as_deref(), Some("garden.material1"));
    assert_eq!(source_sha256(FIXTURE).len(), 64);
    engine_scene::SceneDocument::from_json(&source).unwrap();
}

#[test]
fn prefab_conversion_is_deterministic_without_placement_or_geometry_changes() {
    let flat = import_glb(FIXTURE, "garden").unwrap();
    let prefab = import_glb_prefab(FIXTURE, "garden").unwrap();
    assert_eq!(
        prefab.to_json().unwrap(),
        import_glb_prefab(FIXTURE, "garden")
            .unwrap()
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
    let (
        MeshSource::Inline { vertices, .. },
        MeshSource::Inline {
            vertices: before, ..
        },
    ) = (&document.meshes[0].source, &original.meshes[0].source)
    else {
        panic!("expected baked mesh");
    };
    for (actual, original) in vertices.iter().zip(before) {
        let expected = [
            original.position[0] * 2.0 + 5.0,
            original.position[1] * 3.0 + 2.0,
            original.position[2] * 4.0 - 3.0,
        ];
        for (actual, expected) in actual.position.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 0.00001);
        }
        let normal = glam::Vec3::from_array(original.normal) / glam::Vec3::new(2.0, 3.0, 4.0);
        assert!((glam::Vec3::from_array(actual.normal) - normal.normalize()).length() < 0.00001);
    }
}

#[test]
fn mirrored_nodes_reverse_triangle_winding() {
    let document = import_glb(
        &changed(|json| json["nodes"][1]["scale"] = json!([-1, 1, 1])),
        "garden",
    )
    .unwrap();
    let MeshSource::Inline { indices, .. } = &document.meshes[0].source else {
        panic!("expected inline mesh");
    };
    assert_eq!(&indices[..3], &[0, 2, 1]);
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
    let count = changed(|json| json["accessors"][0]["count"] = json!(1_000_000));
    assert_eq!(
        import_glb(&count, "garden").unwrap_err().code,
        "count_limit"
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
fn unsupported_material_sampler_and_external_resources_are_explicit() {
    for change in [
        |json: &mut Value| {
            json["materials"][0]["pbrMetallicRoughness"]["metallicFactor"] = json!(1)
        },
        |json: &mut Value| json["materials"][0]["alphaMode"] = json!("BLEND"),
        |json: &mut Value| json["samplers"][0]["wrapS"] = json!(10497),
        |json: &mut Value| json["samplers"][0]["minFilter"] = json!(9984),
        |json: &mut Value| {
            json["images"][0] = json!({"uri":"https://example.invalid/texture.png"});
        },
        |json: &mut Value| json["buffers"][0]["uri"] = json!("external.bin"),
        |json: &mut Value| {
            json["meshes"][0]["primitives"][0]["extensions"] =
                json!({"KHR_draco_mesh_compression":{}})
        },
    ] {
        assert_eq!(
            import_glb(&changed(change), "garden").unwrap_err().code,
            "unsupported_feature"
        );
    }
}

#[test]
fn trilinear_samplers_import_like_linear_ones() {
    // LINEAR_MIPMAP_LINEAR states how the renderer samples every texture.
    let trilinear = import_glb(
        &changed(|json| json["samplers"][0]["minFilter"] = json!(9987)),
        "garden",
    )
    .unwrap();
    let linear = import_glb(FIXTURE, "garden").unwrap();
    assert_eq!(
        serde_json::to_value(&trilinear.textures).unwrap(),
        serde_json::to_value(&linear.textures).unwrap()
    );
}

#[test]
fn truncated_oversized_and_foreign_containers_fail() {
    assert_eq!(
        import_glb(&FIXTURE[..20], "garden").unwrap_err().code,
        "invalid_glb"
    );
    assert_eq!(
        import_glb(&vec![0; MAX_INPUT_BYTES + 1], "garden")
            .unwrap_err()
            .code,
        "input_too_large"
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
